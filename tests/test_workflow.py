import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient
from app.engine import Engine
from app.query import update_conditions


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "photos"
        self.source.mkdir()
        self.engine = Engine(self.root / "state")

    def tearDown(self):
        self.temporary.cleanup()

    def make_photo(self, filename, colour="green"):
        path = self.source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), colour).save(path)
        return path

    def test_incremental_index_and_source_integrity(self):
        a = self.make_photo("tree.png")
        self.make_photo("nested/car.jpg", "blue")
        before = hashlib.sha256(a.read_bytes()).hexdigest()
        library = self.engine.add_library(str(self.source))
        first = self.engine.index_library(library["id"])
        self.assertEqual(first["processed"], 2)
        second = self.engine.index_library(library["id"])
        self.assertEqual((second["processed"], second["skipped"]), (0, 2))
        self.assertEqual(before, hashlib.sha256(a.read_bytes()).hexdigest())
        self.make_photo("new.png", "red")
        third = self.engine.index_library(library["id"])
        self.assertEqual((third["processed"], third["skipped"]), (1, 2))
        a.unlink()
        fourth = self.engine.index_library(library["id"])
        self.assertEqual(fourth["removed"], 1)
        self.assertEqual(next(x for x in self.engine.libraries() if x["id"] == library["id"])["count"], 2)

    def test_unavailable_source_does_not_prune(self):
        self.make_photo("kept.png")
        library = self.engine.add_library(str(self.source))
        self.engine.index_library(library["id"])
        self.source.rename(self.root / "temporarily-unplugged")
        failed = self.engine.index_library(library["id"])
        self.assertEqual(failed["status"], "error")
        self.assertEqual(next(x for x in self.engine.libraries() if x["id"] == library["id"])["count"], 1)

    def test_bad_images_are_reported(self):
        self.make_photo("good.png")
        (self.source / "broken.png").write_text("not an image")
        library = self.engine.add_library(str(self.source))
        job = self.engine.index_library(library["id"])
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["processed"], 1)
        self.assertEqual(job["errors"][0]["file"], "broken.png")
        response = self.engine.search(library["id"])
        self.assertEqual(response["total"], 1)

    def test_demo_refinement_preferences_and_reset(self):
        first = self.engine.search("demo", "有树的照片")
        second = self.engine.search("demo", "旁边还有车", first["session"]["id"])
        third = self.engine.search("demo", "可能是白色的", first["session"]["id"])
        self.assertGreater(first["total"], second["total"])
        self.assertEqual(second["total"], 2)
        self.assertEqual(third["total"], 2)
        self.assertEqual(third["results"][0]["id"], "demo-01")
        self.assertEqual(next(c for c in third["session"]["conditions"] if c["concept"] == "white")["kind"], "prefer")
        reset = self.engine.search("demo", session_id=first["session"]["id"], mode="reset")
        self.assertEqual((reset["total"], reset["session"]["conditions"]), (12, []))

    def test_bilingual_demo_and_remove_condition(self):
        english = self.engine.search("demo", "trees and cars")
        chinese = self.engine.search("demo", "有树和车")
        self.assertEqual([x["id"] for x in english["results"]], [x["id"] for x in chinese["results"]])
        car = next(c for c in chinese["session"]["conditions"] if c["concept"] == "car")
        expanded = self.engine.search("demo", session_id=chinese["session"]["id"], remove_id=car["id"])
        self.assertGreater(expanded["total"], chinese["total"])

    def test_metadata_search_does_not_claim_to_see_pixels(self):
        self.make_photo("IMG_001.png", "green")
        library = self.engine.add_library(str(self.source))
        self.engine.index_library(library["id"])
        searched = self.engine.search(library["id"], "树")
        self.assertEqual(searched["mode"], "metadata")
        self.assertEqual(searched["total"], 0)
        self.assertIn("不识别画面", searched["note"])
        uncertain = self.engine.search(library["id"], "大概是2020年", mode="replace")
        self.assertEqual(uncertain["total"], 1)
        self.assertIsNone(uncertain["results"][0]["captured"])

    def test_symlinks_cannot_expose_outside_files(self):
        external = self.root / "secret.png"
        Image.new("RGB", (20, 20)).save(external)
        (self.source / "linked.png").symlink_to(external)
        library = self.engine.add_library(str(self.source))
        self.assertEqual(self.engine.index_library(library["id"])["total"], 0)
        real = self.make_photo("photo.png")
        self.engine.index_library(library["id"])
        item = self.engine.search(library["id"])["results"][0]
        real.unlink()
        real.symlink_to(external)
        with self.assertRaises(ValueError):
            self.engine.asset_file(item["id"])

    def test_api_requires_local_origin_and_page_token(self):
        from app import main
        previous = main.engine
        main.engine = self.engine
        try:
            client = TestClient(main.app)
            state = client.get("/api/state").json()
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.post("/api/search", json={"library_id": "demo"}).status_code, 403)
            valid = {"X-PhotoRecall-Token": state["token"]}
            response = client.post("/api/search", json={"library_id": "demo", "message": "树和车"}, headers=valid)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["total"], 2)
            forbidden = {**valid, "Origin": "https://other.example"}
            self.assertEqual(client.post("/api/search", json={"library_id": "demo"}, headers=forbidden).status_code, 403)
            self.assertEqual(client.get("/api/state", headers={"Host": "other.example"}).status_code, 403)
            self.assertEqual(client.get("/api/assets/not-a-real-id/original").status_code, 400)
        finally:
            main.engine = previous


if __name__ == "__main__":
    unittest.main()
