import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .metadata import OfflinePlaces, gps_coordinates, captured_time
from .query import CONCEPTS, contains, semantic_query, update_conditions
from .providers.siglip import SiglipProvider
from .providers.detector import Detector
from .providers import ollama
from .providers.resources import inference
from .retrieval import visual_stats, semantic_scores, condition_terms, count_evidence, inspection_grade, normalize_inspection, rank_key, DESCRIPTION_FIELDS
from .matching import match_decision
from .pipeline import RetrievalPipeline
from .hybrid import HybridIndex

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class Engine:
    def __init__(self, data_dir=None):
        self.data = Path(data_dir or os.environ.get("PHOTORECALL_DATA", ROOT / "data"))
        self.data.mkdir(parents=True, exist_ok=True)
        self.thumbs = self.data / "thumbnails"
        self.thumbs.mkdir(exist_ok=True)
        self.db_path = self.data / "library.sqlite3"
        self.provider = SiglipProvider(self.data / "models" / "siglip2")
        self.detector = Detector(self.data)
        self.jobs = {}
        self.job_lock = threading.Lock()
        self.inspect_lock = threading.Lock()
        self.verifications = {}
        with self.db() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS libraries (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
              is_demo INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS assets (
              id TEXT PRIMARY KEY, library_id TEXT NOT NULL, path TEXT NOT NULL, name TEXT NOT NULL,
              fingerprint TEXT, mtime_ns INTEGER, size INTEGER, width INTEGER, height INTEGER,
              captured TEXT, thumbnail TEXT, tags TEXT DEFAULT '[]', description TEXT DEFAULT '',
              embedding BLOB, model TEXT, error TEXT,
              UNIQUE(library_id, path));
            CREATE INDEX IF NOT EXISTS assets_library ON assets(library_id);
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY, library_id TEXT NOT NULL, conditions TEXT NOT NULL DEFAULT '[]',
              messages TEXT NOT NULL DEFAULT '[]', updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS inspections (
              asset_id TEXT, fingerprint TEXT, query TEXT, result TEXT, created REAL,
              PRIMARY KEY(asset_id, fingerprint, query));
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS query_plans (key TEXT PRIMARY KEY, result TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS annotations (
              asset_id TEXT PRIMARY KEY, fingerprint TEXT, model TEXT, result TEXT, embedding BLOB, created REAL);
            CREATE TABLE IF NOT EXISTS visual_facts (
              asset_id TEXT, fingerprint TEXT, entity TEXT, version TEXT, result TEXT,
              PRIMARY KEY(asset_id,fingerprint,entity,version));
            CREATE TABLE IF NOT EXISTS detections (
              asset_id TEXT, fingerprint TEXT, version TEXT, result TEXT,
              PRIMARY KEY(asset_id,fingerprint,version));
            """)
            for table, column, definition in [("assets", "visual_stats", "TEXT"), ("sessions", "search_plan", "TEXT DEFAULT '{}'"),
                                              ("annotations", "field_vectors", "BLOB"),
                                              *[("assets", c, t) for c, t in [("latitude", "REAL"), ("longitude", "REAL"), ("place_names", "TEXT"), ("metadata_version", "TEXT")]]]:
                if column not in [row[1] for row in con.execute(f"PRAGMA table_info({table})")]:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self.pipeline = RetrievalPipeline(self)
        self.hybrid = HybridIndex(self)
        with self.db() as con:
            con.execute("INSERT OR IGNORE INTO settings VALUES ('hybrid_search','true')")
            con.execute("INSERT OR IGNORE INTO settings VALUES ('retrieval_vnext', ?)",
                (json.dumps(self.pipeline.main.installed and self.pipeline.reranker.installed),))
        self.seed_demo()

    @contextmanager
    def db(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def seed_demo(self):
        manifest = ROOT / "demo" / "manifest.json"
        if not manifest.exists():
            return
        with self.db() as con:
            con.execute("INSERT OR IGNORE INTO libraries VALUES (?, ?, ?, 1, ?)",
                        ("demo", "记忆练习册", str(ROOT / "demo"), time.time()))
            con.execute("UPDATE libraries SET path=? WHERE id='demo' AND is_demo=1", (str(ROOT / "demo"),))
            for item in json.loads(manifest.read_text()):
                con.execute("""INSERT OR REPLACE INTO assets
                  (id,library_id,path,name,fingerprint,width,height,captured,thumbnail,tags,description)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                  (item["id"], "demo", str(ROOT / "demo" / item["file"]), item["title"], "demo-v1",
                   800, 600, item["date"], item["file"], json.dumps(item["tags"]), item["description"]))

    def setting(self, key, default=False):
        with self.db() as con:
            row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def enable_semantic(self, enabled):
        if enabled:
            with inference(self.provider):
                self.provider._load()
        with self.db() as con:
            con.execute("INSERT OR REPLACE INTO settings VALUES ('semantic', ?)", (json.dumps(enabled),))

    def libraries(self):
        with self.db() as con:
            rows = con.execute("""SELECT l.*, COUNT(a.id) AS count,
                SUM(CASE WHEN a.embedding IS NOT NULL AND a.error IS NULL THEN 1 ELSE 0 END) AS semantic_count,
                SUM(CASE WHEN a.error IS NOT NULL THEN 1 ELSE 0 END) AS error_count,
                (SELECT COUNT(*) FROM annotations n JOIN assets aa ON n.asset_id=aa.id
                 WHERE aa.library_id=l.id AND n.fingerprint=aa.fingerprint) AS detailed_count
                FROM libraries l LEFT JOIN assets a ON a.library_id=l.id
                GROUP BY l.id ORDER BY l.is_demo DESC, l.created""").fetchall()
        libraries = [dict(x) for x in rows]
        for library in libraries:
            library["retrieval_index"] = self.pipeline.counts(library["id"])
            library["hybrid_index"] = self.hybrid.coverage(library["id"])
        return libraries

    def library(self, library_id):
        with self.db() as con:
            row = con.execute("SELECT * FROM libraries WHERE id=?", (library_id,)).fetchone()
        if not row:
            raise ValueError("照片库不存在")
        return dict(row)

    def add_library(self, folder):
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("文件夹不存在，请填写电脑上的完整路径")
        if root == Path(root.anchor) or root == Path.home():
            raise ValueError("请选择具体的照片文件夹，避免扫描整个磁盘或个人主目录")
        key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        with self.db() as con:
            con.execute("INSERT OR IGNORE INTO libraries VALUES (?, ?, ?, 0, ?)",
                        (key, root.name, str(root), time.time()))
        return self.library(key)

    def start_index(self, library_id, scan_only=False):
        library = self.library(library_id)
        if library["is_demo"]:
            raise ValueError("演示库已准备好，无需扫描")
        with self.job_lock:
            old = self.jobs.get(library_id)
            if old and old["status"] in {"running", "scanning"}:
                return self.public_job(old)
            job = {"id": uuid.uuid4().hex, "library_id": library_id, "status": "scanning",
                   "total": 0, "done": 0, "processed": 0, "skipped": 0, "removed": 0,
                   "errors": [], "current": "正在清点文件", "started": time.time(),
                   "cancel": threading.Event(), "scan_only": scan_only}
            self.jobs[library_id] = job
            threading.Thread(target=self.index_library, args=(library_id, job), daemon=True).start()
        return self.public_job(job)

    @staticmethod
    def public_job(job):
        return {k: v for k, v in job.items() if k != "cancel"}

    def index_library(self, library_id, job=None):
        library = self.library(library_id)
        root = Path(library["path"])
        job = job or {"id": "test", "library_id": library_id, "status": "scanning", "total": 0,
                      "done": 0, "processed": 0, "skipped": 0, "removed": 0, "errors": [],
                      "current": "", "cancel": threading.Event()}
        try:
            if not root.is_dir():
                raise ValueError("照片文件夹无法访问，请检查硬盘是否连接")
            files = []
            walk_errors = []
            for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_errors.append):
                dirs[:] = [d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink()
                           and (Path(directory) / d).resolve() != self.data.resolve()]
                if job["cancel"].is_set():
                    job["status"] = "paused"
                    return job
                for name in names:
                    p = Path(directory) / name
                    if p.suffix.lower() in EXTENSIONS and not p.is_symlink():
                        files.append(p)
            if walk_errors:
                raise ValueError("部分目录无法读取，请检查文件夹权限；本次没有清理旧索引")
            job.update(total=len(files), status="running")
            use_ai = self.setting("semantic") and not job.get("scan_only")
            if use_ai:
                with inference(self.provider):
                    self.provider._load()
            expected_model = self.provider.name if use_ai else None
            places = OfflinePlaces()
            places.load()
            seen = set()
            for path in sorted(files):
                if job["cancel"].is_set():
                    job["status"] = "paused"
                    break
                key = hashlib.sha256((library_id + str(path)).encode()).hexdigest()[:24]
                seen.add(key)
                job["current"] = str(path.relative_to(root))
                try:
                    st = path.stat()
                    with self.db() as con:
                        old = con.execute("SELECT * FROM assets WHERE id=?", (key,)).fetchone()
                    if (old and old["mtime_ns"] == st.st_mtime_ns and old["size"] == st.st_size
                            and not old["error"] and (not use_ai or old["model"] == expected_model)
                            and (self.thumbs / old["thumbnail"]).exists()):
                        # Refresh metadata independently; preserve vectors and fingerprint.
                        with Image.open(path) as source:
                            exif = source.getexif()
                            lat, lon = gps_coordinates(exif)
                            captured = captured_time(exif)
                        with self.db() as con:
                            con.execute("UPDATE assets SET latitude=?,longitude=?,place_names=?,metadata_version=?,captured=COALESCE(?,captured) WHERE id=?",
                                (lat, lon, json.dumps(places.reverse(lat, lon), ensure_ascii=False), places.version, captured, key))
                        job["skipped"] += 1
                        job["done"] += 1
                        continue
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    with Image.open(path) as source:
                        width, height = source.size
                        exif = source.getexif()
                        captured = captured_time(exif)
                        lat, lon = gps_coordinates(exif)
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        image.thumbnail((768, 768))
                        thumb = key + ".jpg"
                        temporary = self.thumbs / (thumb + ".tmp")
                        image.save(temporary, format="JPEG", quality=85)
                        temporary.replace(self.thumbs / thumb)
                        vector = self.provider.image(image) if use_ai else None
                    with self.db() as con:
                        con.execute("""INSERT OR REPLACE INTO assets
                        (id,library_id,path,name,fingerprint,mtime_ns,size,width,height,captured,thumbnail,embedding,model)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (key, library_id, str(path), path.name, digest.hexdigest(),
                            st.st_mtime_ns, st.st_size, width, height, captured, thumb,
                            vector.astype(np.float32).tobytes() if vector is not None else None, expected_model))
                        con.execute("UPDATE assets SET latitude=?,longitude=?,place_names=?,metadata_version=? WHERE id=?",
                            (lat, lon, json.dumps(places.reverse(lat, lon), ensure_ascii=False), places.version, key))
                        con.execute("UPDATE assets SET visual_stats=? WHERE id=?", (json.dumps(visual_stats(self.thumbs / thumb)), key))
                    job["processed"] += 1
                except Exception as exc:
                    job["errors"].append({"file": str(path.relative_to(root)), "message": str(exc)[:250]})
                    with self.db() as con:
                        con.execute("""INSERT INTO assets (id,library_id,path,name,error) VALUES (?,?,?,?,?)
                          ON CONFLICT(id) DO UPDATE SET error=excluded.error""", (key, library_id, str(path), path.name, str(exc)[:250]))
                job["done"] += 1
            if job["status"] != "paused":
                # Only prune after a complete scan; interruption and unavailable disks preserve old records.
                with self.db() as con:
                    old_ids = con.execute("SELECT id,thumbnail FROM assets WHERE library_id=?", (library_id,)).fetchall()
                    for row in old_ids:
                        if row["id"] not in seen:
                            con.execute("DELETE FROM assets WHERE id=?", (row["id"],))
                            con.execute("DELETE FROM inspections WHERE asset_id=?", (row["id"],))
                            con.execute("DELETE FROM annotations WHERE asset_id=?", (row["id"],))
                            for table in ("feature_vectors", "rerank_cache", "detections", "visual_facts"):
                                con.execute(f"DELETE FROM {table} WHERE asset_id=?", (row["id"],))
                            if row["thumbnail"]:
                                (self.thumbs / row["thumbnail"]).unlink(missing_ok=True)
                            job["removed"] += 1
                job["status"] = "complete"
                job["current"] = "索引已更新"
                if not job.get("scan_only") and self.setting("retrieval_vnext") and self.pipeline.main.installed and not job["errors"]:
                    job.update(task="retrieval", file_processed=job["processed"], file_skipped=job["skipped"],
                               done=0, processed=0, skipped=0, status="running")
                    self.pipeline.index_library(library_id, job)
                if (use_ai and job["id"]!="test" and job["status"]=="complete" and not job["errors"]
                        and self.setting("hybrid_search",True)):
                    job.update(task="hybrid",done=0,total=0,processed=0,skipped=0,status="running",
                        current="继续建立画面描述与照片文字索引")
                    self.hybrid.build(library_id,job)
        except Exception as exc:
            job["status"] = "error"
            job["errors"].append({"file": "", "message": str(exc)[:250]})
        finally:
            job["finished"] = time.time()
        return job

    def session(self, library_id, session_id=None):
        self.library(library_id)
        with self.db() as con:
            row = con.execute("SELECT * FROM sessions WHERE id=? AND library_id=?", (session_id, library_id)).fetchone() if session_id else None
            if not row:
                session_id = uuid.uuid4().hex
                con.execute("INSERT INTO sessions (id,library_id,conditions,messages,updated) VALUES (?, ?, '[]', '[]', ?)", (session_id, library_id, time.time()))
                row = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return {"id": row["id"], "library_id": row["library_id"], "conditions": json.loads(row["conditions"]),
                "messages": json.loads(row["messages"]), "plan": json.loads(row["search_plan"] or "{}")}

    def save_session(self, session):
        with self.db() as con:
            con.execute("UPDATE sessions SET conditions=?,messages=?,search_plan=?,updated=? WHERE id=?", (
                json.dumps(session["conditions"], ensure_ascii=False), json.dumps(session["messages"][-30:], ensure_ascii=False),
                json.dumps(session.get("plan", {}), ensure_ascii=False),
                time.time(), session["id"]))

    def parse_conditions(self, previous, message):
        plain = [{k: c[k] for k in ["label", "kind", "concept", "slot", "english", "constraint"] if k in c} for c in previous]
        key = hashlib.sha256(json.dumps([plain, message, "plan-v6"], ensure_ascii=False).encode()).hexdigest()
        with self.db() as con:
            cached = con.execute("SELECT result FROM query_plans WHERE key=?", (key,)).fetchone()
        parsed = json.loads(cached[0]) if cached else ollama.plan_search(plain, message)
        if not parsed and not any(word in message for word in ["取消", "撤销", "remove", "clear"]):
            raise RuntimeError("没有解析出可靠条件")
        if not cached:
            with self.db() as con:
                con.execute("INSERT OR REPLACE INTO query_plans VALUES (?,?,?)", (key, json.dumps(parsed, ensure_ascii=False), time.time()))
        # Keep an untouched clause when the planner omitted it. This is a general
        # coverage safeguard, not a vocabulary list or a photo-specific ranking rule.
        for clause in re.split(r"[，,。；;\n]+", message):
            clause = clause.strip()
            if len(clause) < 3 or re.search(r"取消|撤销|撤回|去掉|记错|改成|不限定|remove|instead|actually", clause, re.I):
                continue
            if any(clause in c.get("evidence", "") or clause in c["label"] for c in parsed):
                continue
            remainder = clause
            for c in parsed:
                for text in (c["label"], c.get("evidence", "")):
                    if text:
                        remainder = remainder.replace(text, "")
            meaningful = re.sub(r"[\s的了在和与有是照片里中]+", "", remainder)
            if len(meaningful) < 3:
                continue
            uncertain = bool(re.search(r"可能|好像|大概|不确定|maybe|perhaps|possibly", clause, re.I))
            parsed.append({"label": clause, "english": clause, "concept": "detail", "slot": "detail",
                "kind": "prefer" if uncertain else "include", "soft": uncertain, "evidence": clause,
                "coverage_fallback": True})
        for c in parsed:
            old = next((p for p in previous if p["label"] == c["label"] and p["kind"] == c["kind"]), None)
            c["id"] = old["id"] if old else uuid.uuid4().hex[:10]
        return parsed

    def ensure_stats(self, assets):
        for asset in assets:
            if not asset.get("visual_stats"):
                try:
                    stats = json.dumps(visual_stats(self.thumbs / asset["thumbnail"]))
                    asset["visual_stats"] = stats
                    with self.db() as con:
                        con.execute("UPDATE assets SET visual_stats=? WHERE id=?", (stats, asset["id"]))
                except (OSError, ValueError):
                    pass

    def search(self, library_id, message="", session_id=None, mode="append", remove_id=None, limit=100):
        with self.hybrid.foreground():
            return self._search_impl(library_id,message,session_id,mode,remove_id,limit)

    def _search_impl(self, library_id, message="", session_id=None, mode="append", remove_id=None, limit=100):
        self.hybrid.last_search=time.monotonic()
        started = time.time()
        library = self.library(library_id)
        session = self.session(library_id, session_id)
        old_job = self.verifications.get(session["id"])
        if old_job and self.setting("retrieval_vnext"):
            old_job["cancel"].set()
        use_planner = bool(not library["is_demo"] and self.setting("semantic"))
        migrating = bool(use_planner and session["messages"] and (session["plan"].get("version") != 4 or
                         (session["plan"].get("parser") == "fallback" and session["plan"].get("retry_after", 0) < time.time())))
        # Restore lost cues from old conversation history once, rather than retaining the old lossy chips.
        if migrating and not remove_id and mode != "reset":
            history = list(dict.fromkeys(m["text"] for m in session["messages"] if m["role"] == "user"))
            try:
                session["conditions"] = self.parse_conditions([], "\n".join(history))
                session["plan"] = {"version": 4, "parser": "qwen"}
            except (RuntimeError, OSError, ValueError) as exc:
                recovered = []
                for text in history:
                    recovered = update_conditions(recovered, text)
                session["conditions"] = recovered
                session["plan"] = {"version": 4, "parser": "fallback", "warning": str(exc)[:180], "retry_after": time.time() + 60}
        if remove_id:
            session["conditions"] = [x for x in session["conditions"] if x["id"] != remove_id]
        elif message.strip():
            previous = session["conditions"] if mode == "append" else []
            if use_planner:
                try:
                    session["conditions"] = self.parse_conditions(previous, message)
                    session["plan"] = {"version": 4, "parser": "qwen"}
                except (RuntimeError, OSError, ValueError) as exc:
                    session["conditions"] = update_conditions(previous, message)
                    session["plan"] = {"version": 4, "parser": "fallback", "warning": str(exc)[:180], "retry_after": time.time() + 60}
            else:
                session["conditions"] = update_conditions(previous, message)
            session["messages"].append({"role": "user", "text": message})
        elif mode == "reset":
            session["conditions"] = []
            session["messages"] = []
            session["plan"] = {"version": 4}
        self.save_session(session)
        query = semantic_query(session["conditions"])
        inspection_key = self.inspection_key(session["conditions"])
        with self.db() as con:
            assets = [dict(x) for x in con.execute("SELECT * FROM assets WHERE library_id=? AND error IS NULL", (library_id,))]
            annotations = {x["asset_id"]: dict(x) for x in con.execute("SELECT n.* FROM annotations n JOIN assets a ON n.asset_id=a.id WHERE a.library_id=? AND n.fingerprint=a.fingerprint", (library_id,))}
            cached_inspections = {x["asset_id"]: json.loads(x["result"]) for x in con.execute("SELECT i.* FROM inspections i JOIN assets a ON i.asset_id=a.id WHERE a.library_id=? AND i.fingerprint=a.fingerprint AND i.query=?", (library_id, inspection_key))}
        if self.setting("retrieval_vnext") and self.setting("semantic") and not library["is_demo"]:
            return self.pipeline.search(library, session, assets, cached_inspections, limit, started)
        has_vectors = any(x["embedding"] and x["model"] == self.provider.name for x in assets)
        use_semantic = bool(query and has_vectors and self.setting("semantic"))
        semantic_assets = [a for a in assets if a["embedding"] and a["model"] == self.provider.name]
        semantic_lookup = {}
        if use_semantic:
            self.ensure_stats(semantic_assets)
            scores, score_reasons = semantic_scores(semantic_assets, session["conditions"], self.provider, annotations)
            semantic_lookup = {a["id"]: (float(score), why) for a, score, why in zip(semantic_assets, scores, score_reasons)}
        results = []
        for asset in assets:
            reasons = []
            score = 0.0
            tags = json.loads(asset["tags"] or "[]")
            if library["is_demo"]:
                failed = False
                for condition in session["conditions"]:
                    concept, kind = condition["concept"], condition["kind"]
                    if concept == "detail":
                        continue
                    matches = ((asset["captured"] or "").startswith(condition["label"]) if concept == "year" else concept in tags)
                    if (kind == "include" and not matches) or (kind == "exclude" and matches):
                        failed = True
                        break
                    if matches and kind != "exclude":
                        score += 1 if kind == "include" else .4
                        reasons.append(condition["label"])
                if failed:
                    continue
                engine = "demo"
            elif use_semantic:
                if not asset["embedding"] or asset["model"] != self.provider.name:
                    continue
                score, why = semantic_lookup[asset["id"]]
                engine = "semantic"
                reasons = why + ["完整描述与逐项线索检索，细节可进一步核对"]
            else:
                # Honest fallback: only names and paths, never pretend to see image pixels.
                searchable = (asset["name"] + " " + str(Path(asset["path"]).relative_to(library["path"]))).lower()
                for condition in session["conditions"]:
                    aliases = CONCEPTS.get(condition["concept"], [condition["label"].lower()])
                    matched = any(contains(searchable, a) for a in aliases)
                    if condition["concept"] == "year":
                        matched = (asset["captured"] or "").startswith(condition["label"])
                    if condition["kind"] == "exclude" and matched:
                        score -= 1
                    elif matched:
                        score += 1
                        reasons.append("文件信息：" + condition["label"])
                if query and score <= 0 and any(c["kind"] == "include" for c in session["conditions"]):
                    continue
                engine = "metadata"
            # A reliable explicit date can filter; uncertain dates keep unknown assets.
            excluded = False
            for condition in session["conditions"]:
                if condition["concept"] == "year" and condition["kind"] == "include" and not condition.get("soft") and not library["is_demo"]:
                    if not (asset["captured"] or "").startswith(condition["label"]):
                        excluded = True
            if excluded:
                continue
            results.append({"id": asset["id"], "name": asset["name"], "path": asset["path"],
                "width": asset["width"], "height": asset["height"], "captured": asset["captured"],
                "description": asset["description"], "thumbnail": "/api/assets/" + asset["id"] + "/thumbnail",
                "score": round(score, 5), "reasons": reasons, "engine": engine})
        for result in results:
            if result["id"] in cached_inspections:
                result["inspection"] = cached_inspections[result["id"]]
                result.update(inspection_grade(result["inspection"], session["conditions"]))
        results.sort(key=rank_key)
        total = len(results)
        response_mode = "demo" if library["is_demo"] else ("semantic" if use_semantic else "metadata")
        verification = self.start_verification(session, results[:min(limit, 1000)]) if use_semantic else {"enabled": False, "status": "idle"}
        note = {"demo": "演示结果来自预设标签，用于体验流程；不代表真实 AI 搜索效果。",
                "metadata": "AI 尚未启用，当前仅搜索文件名、路径与已有拍摄日期，不识别画面内容。",
                "semantic": "全库召回后自动逐项核对明确条件；主要结果只展示已核对符合的照片。待核对与无法判断的候选可单独查看。"}[response_mode]
        if verification.get("status") == "unavailable":
            note = "本地看图模型未就绪，当前仅提供相似候选，尚未核对明确条件。请启动 Ollama。"
        if session["plan"].get("warning"):
            note += " 对话模型暂未完成解析，已保留原文线索。"
        return {"session": session, "results": results[:max(1, min(limit, 1000))], "total": total,
                "mode": response_mode, "note": note, "verification": verification, "elapsed_ms": round((time.time() - started) * 1000),
                "message": (f"已检索 {total} 张照片，正在自动核对明确条件。" if verification.get("enabled") else f"找到 {total} 张候选照片。") if total else "暂未找到候选，可以移除条件或开始新的搜索。"}

    def asset_file(self, asset_id, thumbnail=False):
        with self.db() as con:
            row = con.execute("SELECT a.*, l.path AS root, l.is_demo FROM assets a JOIN libraries l ON l.id=a.library_id WHERE a.id=?", (asset_id,)).fetchone()
        if not row:
            raise ValueError("照片不存在")
        if row["is_demo"]:
            path = Path(row["path"])
        elif thumbnail:
            if not row["thumbnail"]:
                raise ValueError("缩略图尚未生成")
            path = self.thumbs / row["thumbnail"]
        else:
            path = Path(row["path"])
        allowed_root = self.thumbs.resolve() if thumbnail and not row["is_demo"] else Path(row["root"]).resolve()
        if path.is_symlink() or not path.resolve().is_relative_to(allowed_root) or not path.is_file():
            raise ValueError("原文件无法访问，或已移到照片库外，请更新索引")
        return path, dict(row)

    @staticmethod
    def inspection_key(conditions):
        plain = [{k: c[k] for k in ["label", "kind", "english", "soft", "constraint"] if k in c} for c in conditions]
        return "inspect-v5:" + hashlib.sha256(json.dumps(plain, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def verification_state(self, session_id):
        job = self.verifications.get(session_id)
        if not job:
            return {"enabled": False, "status": "idle"}
        if job.get("pipeline") == "vnext":
            return self.pipeline.public_job(job)
        return {k: job[k] for k in ["enabled", "key", "status", "stage", "prefilter_done", "prefilter_total", "done", "total", "confirmed", "errors", "results", "remaining"]}

    def start_verification(self, session, candidates):
        key = self.inspection_key(session["conditions"])
        required = any(c["kind"] != "prefer" and not c.get("soft") and c.get("concept") != "year" for c in session["conditions"])
        with self.job_lock:
            old = self.verifications.get(session["id"])
            if old and old["key"] == key and required:
                return self.verification_state(session["id"])
            if old:
                old["cancel"].set()
            if not required:
                self.verifications.pop(session["id"], None)
                return {"enabled": False, "status": "idle"}
            if ollama.MODEL not in ollama.installed_models():
                self.verifications.pop(session["id"], None)
                return {"enabled": False, "status": "unavailable"}
            confirmed = sum(x.get("inspection_tier") == 2 for x in candidates)
            pending = [x["id"] for x in candidates if not x.get("inspection")]
            subset = pending[:48]
            job = {"enabled": True, "key": key, "status": "running", "done": 0, "total": len(subset),
                   "stage": "verification", "prefilter_done": 0, "prefilter_total": 0, "conditions": session["conditions"],
                   "confirmed": confirmed, "errors": [], "results": [], "remaining": len(pending), "cancel": threading.Event()}
            self.verifications[session["id"]] = job
            if confirmed >= 6 or not subset:
                job["status"] = "complete"
            else:
                threading.Thread(target=self.verify_search, args=(session["id"], pending, job), daemon=True).start()
            return self.verification_state(session["id"])

    def verify_search(self, session_id, candidates, job):
        try:
            # Scan the whole current candidate pool cheaply before limiting expensive vision calls.
            if self.detector.ready and any(c.get("constraint") and c["kind"] != "prefer" and not c.get("soft") for c in job["conditions"]):
                job["stage"] = "quantity"
                job["prefilter_total"] = len(candidates)
                passing = []
                for asset_id in candidates:
                    if job["cancel"].is_set():
                        job["status"] = "paused"
                        return
                    try:
                        with self.db() as con:
                            asset = dict(con.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone())
                        detected = self.detect_asset(asset)
                        facts = {c["constraint"]["entity"]: {"count": None, **detected.get(c["constraint"]["entity"], {}),
                                 "evidence": "本地目标检测的可见目标数量下限。"} for c in job["conditions"] if c.get("constraint")}
                        checked = count_evidence(job["conditions"], facts)
                        labels = condition_terms(job["conditions"])
                        contradiction = any(checked.get(term, {}).get("status") == "contradicted" for term, c in zip(labels, job["conditions"]) if c["kind"] != "prefer" and not c.get("soft"))
                        if contradiction:
                            inspection = {"summary": "数量条件不符合，已排除。", "observed_counts": facts,
                                          "conditions": [checked.get(term, {"term": term, "status": "unknown", "evidence": "尚未核对"}) for term in labels]}
                            result = {"id": asset_id, "inspection": inspection, **inspection_grade(inspection, job["conditions"])}
                            with self.db() as con:
                                con.execute("INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?)", (asset_id, asset["fingerprint"], job["key"], json.dumps(inspection, ensure_ascii=False), time.time()))
                            job["results"].append(result)
                            job["remaining"] -= 1
                        else:
                            passing.append(asset_id)
                    except (OSError, ValueError, RuntimeError, TypeError) as exc:
                        job["errors"].append(str(exc)[:200])
                        passing.append(asset_id)
                    job["prefilter_done"] += 1
                candidates = passing
                job["stage"] = "verification"
            job["total"] = min(48, len(candidates))
            for asset_id in candidates[:48]:
                if job["cancel"].is_set():
                    job["status"] = "paused"
                    return
                try:
                    result = self.inspect_candidates(session_id, [asset_id], expected_key=job["key"])[0]
                except ValueError as exc:
                    if str(exc) == "搜索条件已改变":
                        job["status"] = "paused"
                        return
                    job["errors"].append(str(exc)[:200])
                except (RuntimeError, OSError) as exc:
                    job["errors"].append(str(exc)[:200])
                    # Do not retry a broken local model against dozens of photos.
                    job["status"] = "error"
                    return
                else:
                    if job["cancel"].is_set():
                        job["status"] = "paused"
                        return
                    job["results"].append(result)
                    job["confirmed"] += result["inspection_tier"] == 2
                job["done"] += 1
                job["remaining"] -= 1
                if job["confirmed"] >= 6:
                    break
            job["status"] = "complete"
        except Exception as exc:
            job["errors"].append(str(exc)[:200])
            job["status"] = "error"

    def inspect_candidates(self, session_id, asset_ids, expected_key=None):
        with self.db() as con:
            row = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise ValueError("搜索会话不存在")
        conditions = json.loads(row["conditions"])
        query = semantic_query(conditions)
        if not query:
            raise ValueError("请先描述想找的照片")
        results = []
        cache_key = self.inspection_key(conditions)
        if expected_key is not None and cache_key != expected_key:
            raise ValueError("搜索条件已改变")
        with self.inspect_lock:
            for asset_id in asset_ids[:8]:
                path, asset = self.asset_file(asset_id)
                if asset["library_id"] != row["library_id"]:
                    raise ValueError("候选照片不属于当前照片库")
                if asset["is_demo"]:
                    raise ValueError("演示插画无需 AI 核对，请用真实图片测试")
                with self.db() as con:
                    cache = con.execute("SELECT result FROM inspections WHERE asset_id=? AND fingerprint=? AND query=?", (asset_id, asset["fingerprint"], cache_key)).fetchone()
                labels = condition_terms(conditions)
                if cache:
                    result = json.loads(cache[0])
                else:
                    # Measure counts independently of the requested answer, and reuse across searches.
                    entities = list(dict.fromkeys(c["constraint"]["entity"] for c in conditions if c.get("constraint")))
                    facts = {}
                    with self.db() as con:
                        for entity in entities:
                            stored = con.execute("SELECT result FROM visual_facts WHERE asset_id=? AND fingerprint=? AND entity=? AND version='count-v3'", (asset_id, asset["fingerprint"], entity)).fetchone()
                            if stored:
                                facts[entity] = json.loads(stored[0])
                    missing = [entity for entity in entities if entity not in facts or
                               (facts[entity].get("count") is None and not facts[entity].get("measured"))]
                    if missing:
                        detections = self.detect_asset(asset) if self.detector.ready else {}
                        bounds = {entity: {"count": None, **detections.get(entity, {}), "evidence": "本地目标检测提供可见目标的位置与数量下限。"} for entity in missing}
                        provisional = count_evidence(conditions, bounds)
                        decisive_failure = any(provisional.get(term, {}).get("status") == "contradicted" for term, c in zip(labels, conditions) if c["kind"] != "prefer" and not c.get("soft"))
                        sufficient_bounds = all(provisional.get(term, {}).get("status") in {"matched", "contradicted"} for term, c in zip(labels, conditions) if c.get("constraint"))
                        if decisive_failure or sufficient_bounds:
                            measured = bounds
                        else:
                            measured = ollama.count_entities(path, missing)
                            for entity, fact in measured.items():
                                fact["measured"] = True
                                bound = detections.get(entity, {})
                                fact.update(bound)
                                if fact.get("count") is not None and fact["count"] < fact.get("lower_bound", 0) + fact.get("possible_extra", 0):
                                    fact["count"] = None
                                    fact["evidence"] = "看图计数与目标检测存在冲突或额外疑似目标，保留数量下限并标记不确定。"
                        facts.update(measured)
                        with self.db() as con:
                            for entity, fact in measured.items():
                                if fact.get("count") is None and not fact.get("lower_bound"):
                                    continue
                                con.execute("INSERT OR REPLACE INTO visual_facts VALUES (?,?,?,?,?)", (asset_id, asset["fingerprint"], entity, "count-v3", json.dumps(fact, ensure_ascii=False)))
                    counted = count_evidence(conditions, facts)
                    failed_count = any(counted.get(term, {}).get("status") == "contradicted" for term, c in zip(labels, conditions) if c["kind"] != "prefer" and not c.get("soft"))
                    if failed_count:
                        result = {"summary": "数量条件不符合，已排除。", "conditions": [counted.get(term, {"term": term, "status": "unknown", "evidence": "数量条件已不符合，未继续核对此项"}) for term in labels]}
                    else:
                        result = ollama.inspect(path, query, labels)
                        result["conditions"] = [counted[item["term"]] if item["term"] in counted and
                            (counted[item["term"]]["status"] != "matched" or item["status"] == "matched") else item for item in result["conditions"]]
                    result["observed_counts"] = facts
                    with self.db() as con:
                        con.execute("INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?)", (asset_id, asset["fingerprint"], cache_key, json.dumps(result, ensure_ascii=False), time.time()))
                result = normalize_inspection(result)
                update={"id": asset_id, "inspection": result, **inspection_grade(result, conditions), "cached": bool(cache)}
                update.update(match_decision(update,conditions))
                results.append(update)
        self.pipeline.publish_inspections(session_id,cache_key,results,conditions)
        return sorted(results, key=lambda x: -x["inspection_score"])

    def detect_asset(self, asset):
        with self.db() as con:
            cached = con.execute("SELECT result FROM detections WHERE asset_id=? AND fingerprint=? AND version=?", (asset["id"], asset["fingerprint"], self.detector.version)).fetchone()
        if cached:
            return json.loads(cached[0])
        # Original resolution retains background figures lost in small thumbnails.
        path, _ = self.asset_file(asset["id"])
        detected = self.detector.detect(path)
        with self.db() as con:
            con.execute("INSERT OR REPLACE INTO detections VALUES (?,?,?,?)", (asset["id"], asset["fingerprint"], self.detector.version, json.dumps(detected)))
        return detected

    def start_analysis(self, library_id):
        library = self.library(library_id)
        if library["is_demo"]:
            raise ValueError("请选择真实照片库")
        if ollama.MODEL not in ollama.installed_models():
            raise RuntimeError("请先启动 Ollama")
        with self.job_lock:
            old = self.jobs.get(library_id)
            if old and old["status"] in {"running", "scanning"}:
                raise ValueError("当前照片库已有任务，请先暂停或等待完成")
            job = {"id": uuid.uuid4().hex, "library_id": library_id, "task": "analysis", "status": "running",
                   "total": 0, "done": 0, "processed": 0, "skipped": 0, "removed": 0, "errors": [],
                   "current": "正在准备详细画面分析", "cancel": threading.Event()}
            self.jobs[library_id] = job
            threading.Thread(target=self.analyze_library, args=(library_id, job), daemon=True).start()
        return self.public_job(job)

    def analyze_library(self, library_id, job):
        try:
            with self.db() as con:
                rows = [dict(x) for x in con.execute("SELECT a.*,n.fingerprint AS analyzed_fingerprint,n.field_vectors AS analyzed_fields FROM assets a LEFT JOIN annotations n ON a.id=n.asset_id WHERE a.library_id=? AND a.error IS NULL", (library_id,))]
            job["total"] = len(rows)
            for asset in rows:
                if job["cancel"].is_set():
                    job["status"] = "paused"
                    break
                job["current"] = asset["name"]
                if asset["analyzed_fingerprint"] == asset["fingerprint"] and (asset["analyzed_fields"] is not None or not self.setting("semantic")):
                    job["skipped"] += 1
                    job["done"] += 1
                    continue
                try:
                    path, _ = self.asset_file(asset["id"], thumbnail=True)
                    result = ollama.describe(path)
                    caption = result.get("caption", "")
                    if not isinstance(caption, str) or not caption:
                        raise RuntimeError("画面分析没有返回有效描述")
                    vector = self.provider.text(caption) if self.setting("semantic") else None
                    field_vectors = None
                    if vector is not None:
                        field_vectors = np.stack([self.provider.text(result[field]) if isinstance(result.get(field), str) and result[field].lower().strip() not in {"unknown", "none", "unclear", ""}
                                                  else np.zeros_like(vector) for field in DESCRIPTION_FIELDS]).astype(np.float32).tobytes()
                    with self.db() as con:
                        # Avoid attaching analysis to a file that changed while the model was running.
                        current = con.execute("SELECT fingerprint FROM assets WHERE id=?", (asset["id"],)).fetchone()
                        if current and current[0] == asset["fingerprint"]:
                            con.execute("INSERT OR REPLACE INTO annotations (asset_id,fingerprint,model,result,embedding,created,field_vectors) VALUES (?,?,?,?,?,?,?)", (asset["id"], asset["fingerprint"], ollama.MODEL + ":describe-v2", json.dumps(result, ensure_ascii=False), vector.astype(np.float32).tobytes() if vector is not None else None, time.time(), field_vectors))
                            job["processed"] += 1
                except Exception as exc:
                    job["errors"].append({"file": asset["name"], "message": str(exc)[:250]})
                job["done"] += 1
                time.sleep(.05)
            if job["status"] != "paused":
                job["status"] = "complete"
                job["current"] = "详细画面索引已更新"
        except Exception as exc:
            job["status"] = "error"
            job["errors"].append({"file": "", "message": str(exc)[:250]})
