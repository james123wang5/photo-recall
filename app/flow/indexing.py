"""Resumable layered index writer. Originals are opened read-only.

File scanning is separate from visual/OCR/description/person processing. Each
component has independent fingerprints and failure state; an empty OCR result
does not exclude an asset from any visual channel.
"""
import json
import threading
import time
import uuid
import numpy as np
from PIL import Image, ImageOps
from ..hybrid import observation_text
from ..providers import ollama
from .domain import digest
from .store import encode
from .tools import crop_image


class FlowIndexer:
    def __init__(self, flow):
        self.flow, self.e = flow, flow.e

    def start(self, library_id, components):
        if self.e.library(library_id)["is_demo"]:
            raise ValueError("请选择真实照片库")
        if not components or not set(components) <= {"visual", "ocr", "description", "people"}:
            raise ValueError("索引组件不正确")
        with self.e.job_lock:
            if any(j.get("status") in {"running", "scanning"} for j in self.e.jobs.values()):
                raise ValueError("已有建库任务，请先暂停或等待完成")
            job = {"id": uuid.uuid4().hex, "library_id": library_id, "task": "flow-index", "status": "running",
                "total": 0, "done": 0, "processed": 0, "skipped": 0, "removed": 0, "errors": [],
                "current": "准备分层索引", "started": time.time(), "cancel": threading.Event()}
            self.e.jobs[library_id] = job
            threading.Thread(target=self.build, args=(job, components), daemon=True, name="flow-index").start()
            return self.e.public_job(job)

    def _vector(self, asset, provider, version, image, region):
        key = encode(region)
        with self.e.db() as con:
            old = con.execute("SELECT 1 FROM flow_vectors WHERE asset_id=? AND fingerprint=? AND version=? AND region=?",
                              (asset["id"], asset["fingerprint"], version, key)).fetchone()
        if old:
            return
        vector = np.asarray(provider.image(image), dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("图像向量格式不正确")
        self.flow.tools.original(asset)
        with self.e.db() as con:
            con.execute("INSERT OR REPLACE INTO flow_vectors VALUES (?,?,?,?,?,?)", (
                asset["id"], asset["fingerprint"], version, key, vector.tobytes(), vector.size))

    def _people(self, asset, image):
        tool = self.flow.tools.faces
        for index, face in enumerate(tool.faces(image)):
            vector = np.asarray(face.normed_embedding, dtype=np.float32)
            with self.e.db() as con:
                rows = con.execute("SELECT f.person_id,f.vector FROM flow_faces f JOIN assets a ON a.id=f.asset_id AND a.fingerprint=f.fingerprint WHERE f.version=?", (tool.version,)).fetchall()
                similarities = [(float(np.frombuffer(r["vector"], dtype=np.float32) @ vector), r["person_id"]) for r in rows]
                best = max(similarities, default=(-1, ""))
                pid = best[1] if best[0] >= .5 else uuid.uuid4().hex
                con.execute("INSERT OR IGNORE INTO flow_people VALUES (?, '')", (pid,))
                box = [float(v) / (image.width if i % 2 == 0 else image.height) for i, v in enumerate(face.bbox.tolist())]
                con.execute("INSERT OR REPLACE INTO flow_faces VALUES (?,?,?,?,?,?,?)", (
                    asset["id"], asset["fingerprint"], str(index), pid, vector.tobytes(), encode(box), tool.version))

    def build(self, job, components):
        config = self.flow.store.config()
        primary = self.flow.recall.primary(config)
        siglip = self.e.provider
        description_version = self.flow.tools.vision_signature(config["vision_model"]) + ":flow-describe-v1"
        assets = self.flow.catalog(job["library_id"])
        lanes = [c for c in ("main_visual", "visual", "ocr", "description", "text_vectors", "people")
                 if c in components or c == "main_visual" and "visual" in components
                 or c == "text_vectors" and "description" in components]
        job["total"] = len(assets) * len(lanes)
        job["photo_total"] = len(assets)
        try:
            # All visual assets become searchable before optional language work.
            for component in lanes:
                version = {"main_visual": primary.version, "visual": siglip.name + ":regions4-v1",
                    "ocr": self.e.hybrid.ocr.version,
                    "description": description_version,
                    "text_vectors": self.e.hybrid.embedding.version + ":" + description_version,
                    "people": self.flow.tools.faces.version}[component]
                for asset in assets:
                    if not self.e.hybrid._wait(job):
                        job["status"] = "paused"
                        return
                    job.update(current={"main_visual": "主图像索引", "visual": "互补图像与局部", "ocr": "照片文字", "description": "画面描述", "text_vectors": "描述向量", "people": "人物线索"}[component] + " · " + asset["name"], stage=component)
                    with self.e.db() as con:
                        old = con.execute("SELECT status FROM flow_index_state WHERE asset_id=? AND component=? AND fingerprint=? AND version=?", (
                            asset["id"], component, asset["fingerprint"], version)).fetchone()
                    if old and old[0] in {"ready", "empty"}:
                        job["skipped"] += 1
                        job["done"] += 1
                        continue
                    self.flow.store.index_state(asset, component, version, "processing")
                    try:
                        path = self.flow.tools.original(asset)
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                        empty = False
                        if component == "main_visual":
                            if not primary.installed:
                                raise RuntimeError("所选主召回权重缺失，请先准备本地模型")
                            self._vector(asset, primary, primary.version, image, "full")
                        elif component == "visual":
                            if not siglip.installed:
                                raise RuntimeError("SigLIP 权重缺失，请先准备本地模型")
                            self._vector(asset, siglip, siglip.name, image, "full")
                            for box in ([0, 0, .7, .7], [.3, 0, 1, .7], [0, .3, .7, 1], [.3, .3, 1, 1]):
                                self._vector(asset, siglip, siglip.name, crop_image(image, box), box)
                        elif component == "ocr":
                            result = self.e.hybrid.ocr.read(path)
                            text = "\n".join(line["text"] for line in result.get("lines", []) if line.get("confidence", 0) >= .3)
                            self.e.hybrid.save(asset, "ocr", result["version"], result, text)
                            empty = not bool(text)
                        elif component == "description":
                            result = ollama.describe(path, model=config["vision_model"], keep_alive=60)
                            text = observation_text(result)
                            if not text:
                                raise ValueError("画面描述为空")
                            self.e.hybrid.save(asset, "caption", version, result, text)
                        elif component == "text_vectors":
                            with self.e.db() as con:
                                observation = con.execute("SELECT text,version FROM observations WHERE asset_id=? AND fingerprint=? AND kind='caption'",
                                    (asset["id"], asset["fingerprint"])).fetchone()
                            if not observation:
                                raise ValueError("该照片尚无有效描述，描述向量可稍后续建")
                            vector = self.e.hybrid.embedding.encode([observation["text"]])[0]
                            with self.e.db() as con:
                                con.execute("UPDATE observations SET vector=?,vector_version=? WHERE asset_id=? AND fingerprint=? AND kind='caption' AND version=?", (
                                    vector.tobytes(), self.e.hybrid.embedding.version, asset["id"], asset["fingerprint"], observation["version"]))
                        else:
                            if not config["people_enabled"]:
                                raise ValueError("请在设置中明确启用人物组件后建立人物索引")
                            self._people(asset, image)
                        self.flow.tools.original(asset)
                        self.flow.store.index_state(asset, component, version, "empty" if empty else "ready")
                        job["processed"] += 1
                    except Exception as exc:
                        self.flow.store.index_state(asset, component, version, "error", str(exc)[:400])
                        job["errors"].append({"file": asset["name"], "message": str(exc)[:250]})
                        job["error_count"] = job.get("error_count", 0) + 1
                        job["errors"] = job["errors"][-30:]
                    job["done"] += 1
            job.update(status="complete", current="本轮分层索引处理完成")
        except Exception as exc:
            job.update(status="error", current=str(exc)[:400])
        finally:
            self.e.hybrid.ocr.close()
            job["finished"] = time.time()
