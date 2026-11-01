"""Versioned image indices, full-library recall, bounded reranking and visible uncertainty."""
import hashlib
import json
import threading
import time
import uuid
import numpy as np
from PIL import Image, ImageOps
from .providers.multimodal import WemmProvider, CoreReranker
from .retrieval import condition_terms, count_evidence, inspection_grade, normalize_inspection, percentile
from .ranking import evidence_order
from .matching import match_decision, match_summary


def query_text(conditions, relaxed=False, english=False):
    parts = []
    for c in conditions:
        if relaxed and (c.get("soft") or c["kind"] == "prefer"):
            continue
        label = c.get("english", c["label"]) if english else c["label"]
        if c["kind"] == "exclude":
            label = ("without " if english else "不含 ") + label
        elif not relaxed and (c.get("soft") or c["kind"] == "prefer"):
            label = ("possibly " if english else "可能 ") + label
        parts.append(label)
    return ("A photograph showing " if english else "照片：") + "; ".join(parts) if parts else "A photograph"


def fuse(rankings, limit=500,grouped=False,caption_coverage=1):
    scores, sources = {}, {}
    family_scores={}
    image_channels={}
    for name, ids in rankings.items():
        family="caption" if name.startswith("描述") else "ocr" if name=="照片文字" else "metadata" if name=="拍摄日期" else "image"
        for rank, asset_id in enumerate(dict.fromkeys(ids)):
            if grouped:
                if family=="image":
                    image_channels.setdefault(name,{})[asset_id]=1/(60+rank+1)
                    sources.setdefault(asset_id,[]).append(name)
                    continue
                key=(family,asset_id)
                weight=caption_coverage if family=="caption" else 1
                family_scores[key]=max(family_scores.get(key,0),weight/(60+rank+1))
            else:
                scores[asset_id] = scores.get(asset_id, 0) + 1 / (60 + rank + 1)
            sources.setdefault(asset_id, []).append(name)
    if grouped:
        for (_,aid),value in family_scores.items():scores[aid]=scores.get(aid,0)+value
        if image_channels:
            anchor="完整描述" if "完整描述" in image_channels else "整图补漏" if "整图补漏" in image_channels else next(iter(image_channels))
            primary=image_channels[anchor]
            supporting=[values for name,values in image_channels.items() if name!=anchor]
            image_ids=set().union(*(values.keys() for values in image_channels.values()))
            for aid in image_ids:
                main=primary.get(aid,0)
                auxiliary=sum(values.get(aid,0) for values in supporting)/len(supporting) if supporting else main
                # Main full-query retrieval is the anchor; a single crop cannot win an equal first-place vote.
                credit=main+.25*(auxiliary-main)
                scores[aid]=scores.get(aid,0)+credit
    ordered = sorted(scores, key=lambda aid: (-scores[aid], aid))[:limit]
    return ordered, scores, sources


def select_candidates(rankings, order, budget):
    # Reserve entry points before fusion truncation, rather than repeating top-8-only inspection.
    selected = list(order[:min(16, budget)])
    quota = max(1, budget // max(1, len(rankings)) // 8)
    for ids in rankings.values():
        for aid in ids[:quota]:
            if aid not in selected:
                selected.append(aid)
    selected = selected[:budget]
    # Audit the whole remaining pool; do not spend all exploration on ranks 32–80.
    tail = order[32:]
    if tail and len(selected)<budget:
        count=min(max(1,budget//4),budget-len(selected),len(tail))
        for i in np.linspace(0,len(tail)-1,count,dtype=int):
            aid=tail[i]
            if aid not in selected:
                selected.append(aid)
    for aid in order:
        if len(selected)>=budget:
            break
        if aid not in selected:
            selected.append(aid)
    return selected


def merge_reranked(order, ranked, lookup):
    """CORE supports retrieval; its score is not a calibrated binary pool gate."""
    positions = {aid: i for i, aid in enumerate(order)}
    def key(aid):
        score = ranked.get(aid)
        adjustment = 0 if score is None else 12 * (1 - 2 * max(0, min(1, score)))
        return positions[aid] + adjustment, positions[aid]
    return evidence_order(sorted(order, key=key), lookup)


def similar_candidates(assets, positives, remaining, main_rows):
    """Query-independent visual neighbors are candidates, never inherited matches."""
    vectors = {r["asset_id"]:np.frombuffer(r["vector"],dtype=np.float32) for r in main_rows}
    if not vectors:
        vectors = {a["id"]:np.frombuffer(a["embedding"],dtype=np.float32) for a in assets if a.get("embedding")}
    seeds = [vectors[aid] for aid in positives if aid in vectors]
    candidates = [aid for aid in remaining if aid in vectors]
    if not seeds or not candidates:
        return []
    similarities = np.stack([vectors[aid] for aid in candidates]) @ np.stack(seeds).T
    return [candidates[i] for i in np.argsort(-similarities.max(axis=1),kind="stable")]


def regions(image):
    w, h = image.size
    # Overlap retains small gestures near quadrant boundaries; coordinates are reproducible.
    return [(0, 0, int(.7*w), int(.7*h)), (int(.3*w), 0, w, int(.7*h)),
            (0, int(.3*h), int(.7*w), h), (int(.3*w), int(.3*h), w, h)]


class RetrievalPipeline:
    def __init__(self, engine):
        self.engine = engine
        self.main = WemmProvider(engine.data / "models/wemm-2b")
        self.reranker = CoreReranker(engine.data / "models/core-reranker-2b")
        self.local_version = engine.provider.name + ":regions4:v1"
        with engine.db() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS feature_vectors (
                    asset_id TEXT, fingerprint TEXT, version TEXT, region INTEGER,
                    coordinates TEXT, vector BLOB, dimension INTEGER, created REAL,
                    PRIMARY KEY(asset_id,version,region));
                CREATE TABLE IF NOT EXISTS rerank_cache (
                    asset_id TEXT, fingerprint TEXT, version TEXT, query TEXT, score REAL, elapsed_ms REAL,
                    PRIMARY KEY(asset_id,fingerprint,version,query));
                CREATE TABLE IF NOT EXISTS search_runs (
                    id TEXT PRIMARY KEY, session_id TEXT, query TEXT, report TEXT, created REAL);
                CREATE INDEX IF NOT EXISTS vectors_version ON feature_vectors(version);
                CREATE TABLE IF NOT EXISTS text_vectors (
                    version TEXT, text TEXT, vector BLOB, dimension INTEGER,
                    PRIMARY KEY(version,text));
            """)

    def counts(self, library_id):
        with self.engine.db() as con:
            rows = con.execute("""SELECT v.version,v.region,COUNT(*) AS n FROM feature_vectors v
                JOIN assets a ON a.id=v.asset_id AND a.fingerprint=v.fingerprint
                WHERE a.library_id=? AND a.error IS NULL GROUP BY v.version,v.region""", (library_id,)).fetchall()
            total = con.execute("SELECT COUNT(*) FROM assets WHERE library_id=? AND error IS NULL", (library_id,)).fetchone()[0]
        return {"main": sum(r["n"] for r in rows if r["version"] == self.main.version and r["region"] == -1),
                "local": sum(r["n"] for r in rows if r["version"] == self.local_version), "total": total}

    def text_vector(self, provider, version, text):
        with self.engine.db() as con:
            cached = con.execute("SELECT vector FROM text_vectors WHERE version=? AND text=?", (version,text)).fetchone()
        if cached:
            return np.frombuffer(cached[0], dtype=np.float32).copy()
        vector = provider.text(text).astype(np.float32)
        with self.engine.db() as con:
            con.execute("INSERT OR REPLACE INTO text_vectors VALUES (?,?,?,?)", (version,text,vector.tobytes(),len(vector)))
        return vector

    def start_index(self, library_id):
        e = self.engine
        if e.library(library_id)["is_demo"]:
            raise ValueError("请选择真实照片库")
        if not self.main.installed:
            raise RuntimeError("主检索模型尚未完整下载")
        with e.job_lock:
            old = e.jobs.get(library_id)
            if old and old["status"] in {"running", "scanning"}:
                raise ValueError("已有索引任务，请等待或暂停后继续")
            job = {"id": uuid.uuid4().hex, "library_id": library_id, "task": "retrieval", "status": "running",
                "total": 0, "done": 0, "processed": 0, "skipped": 0, "removed": 0, "errors": [],
                "current": "准备新版视觉索引", "started": time.time(), "cancel": threading.Event()}
            e.jobs[library_id] = job
            threading.Thread(target=self.index_library, args=(library_id, job), daemon=True).start()
        return e.public_job(job)

    def index_library(self, library_id, job):
        e = self.engine
        try:
            with e.db() as con:
                assets = [dict(r) for r in con.execute("SELECT * FROM assets WHERE library_id=? AND error IS NULL", (library_id,))]
            # Establish inexpensive small-object coverage before the slower main model.
            job["total"] = len(assets) * 2
            for version in [self.local_version, self.main.version]:
                for asset in assets:
                    if job["cancel"].is_set():
                        job["status"] = "paused"
                        return
                    job["current"] = ("整图索引 · " if version == self.main.version else "局部补漏 · ") + asset["name"]
                    if version == self.main.version and e.detector.ready:
                        try:
                            e.detect_asset(asset)
                        except Exception as exc:
                            job["errors"].append({"file":asset["name"],"message":"目标位置未完成："+str(exc)[:180]})
                    with e.db() as con:
                        present = {r[0] for r in con.execute("SELECT region FROM feature_vectors WHERE asset_id=? AND fingerprint=? AND version=?",
                            (asset["id"], asset["fingerprint"], version))}
                    required = {-1} if version == self.main.version else {0, 1, 2, 3}
                    if required <= present:
                        job["skipped"] += 1
                        job["done"] += 1
                        continue
                    try:
                        path, _ = e.asset_file(asset["id"])
                        stat = path.stat()
                        if stat.st_size != asset["size"] or stat.st_mtime_ns != asset["mtime_ns"]:
                            raise RuntimeError("原文件已变化，请先更新文件夹索引")
                        with Image.open(path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                        boxes = [None] if version == self.main.version else regions(image)
                        features = []
                        for region, box in enumerate(boxes):
                            region = -1 if box is None else region
                            if region in present:
                                continue
                            if job["cancel"].is_set():
                                break
                            vector = self.main.image(image) if box is None else e.provider.image(image.crop(box))
                            features.append((region, box, vector))
                        if path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != stat.st_size:
                            raise RuntimeError("照片在索引过程中改变，未保存结果")
                        with e.db() as con:
                            current = con.execute("SELECT fingerprint FROM assets WHERE id=?", (asset["id"],)).fetchone()
                            if current and current[0] == asset["fingerprint"]:
                                for region, box, vector in features:
                                    con.execute("INSERT OR REPLACE INTO feature_vectors VALUES (?,?,?,?,?,?,?,?)",
                                        (asset["id"], asset["fingerprint"], version, region, json.dumps(box),
                                         vector.astype(np.float32).tobytes(), len(vector), time.time()))
                        job["processed"] += 1
                    except Exception as exc:
                        job["errors"].append({"file": asset["name"], "message": str(exc)[:250]})
                        # A model-wide incompatibility must not repeat on the entire library.
                        if not isinstance(exc, (OSError, ValueError)):
                            raise
                    job["done"] += 1
            job["status"] = "complete"
            job["current"] = "新版视觉索引已建立"
        except Exception as exc:
            job["status"] = "error"
            job["errors"].append({"file": "", "message": str(exc)[:250]})
        finally:
            self.main.unload()
            e.provider.unload()
            job["finished"] = time.time()

    def _vector_ranking(self, assets, vector, field="embedding", top=200):
        compatible = [a for a in assets if a.get(field) and len(a[field]) == len(vector)*4]
        if not compatible:
            return []
        matrix = np.stack([np.frombuffer(a[field], dtype=np.float32) for a in compatible])
        scores = matrix @ vector
        return [compatible[i]["id"] for i in np.argsort(-scores, kind="stable")[:top]]

    def condition_ranking(self, rows, conditions):
        """A missing required visual cue should not be averaged away by easy cues.

        These relative similarities support recall, never certify a numeric or
        spatial condition. Explicit contradictions still come from visual evidence.
        """
        hard = [c for c in conditions if c["kind"] == "include" and not c.get("soft")
                and c.get("concept") != "year" and not c.get("constraint")]
        if len(hard) < 2:
            return []
        matrix = np.stack([np.frombuffer(r["vector"],dtype=np.float32) for r in rows])
        cue_scores = []
        for c in hard:
            vector = self.text_vector(self.main,self.main.version,"照片："+c["label"])
            cue_scores.append(percentile(matrix @ vector))
        values = np.stack(cue_scores)
        # Rank-based combination avoids mixing cosine scales between different prompts.
        combined = .7 * values.min(axis=0) + .3 * values.mean(axis=0)
        return [rows[i]["id"] for i in np.argsort(-combined,kind="stable")[:500]]

    def recall(self, assets, conditions, use_main=False):
        e = self.engine
        rankings = {}
        query = query_text(conditions)
        english = query_text(conditions, english=True)
        valid = {a["id"] for a in assets}
        fallback = [a for a in assets if a.get("model") == e.provider.name]
        if fallback and e.provider.installed:
            rankings["整图补漏"] = self._vector_ranking(fallback, self.text_vector(e.provider,e.provider.name,english), top=500)
        with e.db() as con:
            rows = [dict(r) for r in con.execute("""SELECT v.*,a.id FROM feature_vectors v JOIN assets a
                ON a.id=v.asset_id AND a.fingerprint=v.fingerprint WHERE a.library_id=? AND a.error IS NULL""",
                (assets[0]["library_id"],))] if assets else []
        rows = [r for r in rows if r["asset_id"] in valid]
        main_rows = [r for r in rows if r["version"] == self.main.version and r["region"] == -1]
        locals_ = [r for r in rows if r["version"] == self.local_version]
        # Different windows of one photograph aggregate by best similarity, not separate photo IDs.
        # An incomplete additional index unfairly boosts only the indexed prefix
        # in rank fusion. Keep full-library recall until that source covers it.
        if locals_ and len({r["asset_id"] for r in locals_}) == len(valid):
            key_cues = [c.get("english", c["label"]) for c in conditions if c["kind"] != "exclude" and c.get("concept") != "year"]
            for cue in key_cues[-3:]:
                vector = self.text_vector(e.provider,e.provider.name,"A photograph showing " + cue)
                compatible = [row for row in locals_ if row["dimension"] == len(vector)]
                best = {}
                if compatible:
                    similarities = np.stack([np.frombuffer(row["vector"],dtype=np.float32) for row in compatible]) @ vector
                    for row,value in zip(compatible,similarities):
                        best[row["asset_id"]] = max(float(value), best.get(row["asset_id"], -1))
                rankings["局部：" + cue[:45]] = sorted(best, key=lambda aid: -best[aid])[:500]
        # Group all SigLIP text work before WeMM, avoiding SigLIP→WeMM→SigLIP reloads.
        if use_main and len(main_rows) == len(valid) and main_rows:
            rankings["完整描述"] = self._vector_ranking(main_rows, self.text_vector(self.main,self.main.version,query), "vector", 500)
            combined = self.condition_ranking(main_rows,conditions)
            if combined:
                rankings["必要线索综合"] = combined
            relaxed = query_text(conditions, relaxed=True)
            if relaxed != query and relaxed != "A photograph":
                rankings["放宽不确定线索"] = self._vector_ranking(main_rows, self.text_vector(self.main,self.main.version,relaxed), "vector", 500)
        # Explicit dates add an entry; fuzzy dates never remove missing-EXIF images.
        for c in conditions:
            if c.get("concept") == "year":
                rankings["拍摄日期"] = [a["id"] for a in assets if (a["captured"] or "").startswith(c["label"])]
        hybrid=e.setting("hybrid_search",True)
        caption_coverage=1
        if hybrid:
            rankings.update(e.hybrid.rankings(assets,conditions,semantic=use_main))
            if assets:
                caption_coverage=min(1,e.hybrid.coverage(assets[0]["library_id"])["caption"]/max(1,len(assets)))
        rankings = {name: ids for name, ids in rankings.items() if ids}
        if not rankings:
            rankings["尚未建立画面索引"] = [a["id"] for a in assets]
        return rankings, fuse(rankings,grouped=hybrid,caption_coverage=caption_coverage)

    def search(self, library, session, assets, cached_inspections, limit, started):
        e = self.engine
        old = e.verifications.get(session["id"])
        if old:
            old["cancel"].set()
        if not session["conditions"]:
            rankings = {"浏览照片": [a["id"] for a in sorted(assets, key=lambda a: a["captured"] or "", reverse=True)]}
            order, scores, sources = fuse(rankings, limit=1000)
        else:
            rankings, (order, scores, sources) = self.recall(assets, session["conditions"])
        results = self.results(assets, order, scores, sources, cached_inspections, session["conditions"])
        key = e.inspection_key(session["conditions"])
        job = {"enabled": bool(session["conditions"]), "key": key, "pipeline": "vnext", "status": "running",
            "stage": "recall", "prefilter_done": 0, "prefilter_total": 0, "done": 0, "total": 0,
            "confirmed": 0, "errors": [], "results": results, "remaining": len(order), "rankings": rankings,
            "recall_count": len(order), "indexed": self.counts(library["id"]), "elapsed_ms": 0,
            "cancel": threading.Event(), "query": query_text(session["conditions"]), "started": time.time()}
        job["revision"]=0
        job["conditions"]=session["conditions"]
        job["hybrid"]=bool(e.setting("hybrid_search",True))
        job["observation_coverage"]=e.hybrid.coverage(library["id"])
        if session["conditions"]:
            e.verifications[session["id"]] = job
            threading.Thread(target=self.refine, args=(session, assets, cached_inspections, job), daemon=True).start()
        else:
            job["status"] = "idle"
            e.verifications.pop(session["id"], None)
        return {"session": session, "results": results[:limit], "total": len(assets), "mode": "semantic",
            "note": "先展示全库召回候选，再逐步精排。相关性分数不是匹配概率；未核对和无法判断的照片仍可查看。",
            "verification": self.public_job(job), "elapsed_ms": round((time.time()-started)*1000),
            "message": f"已读取 {len(assets)} 张照片的索引，正在按完整线索整理候选。" if session["conditions"] else f"照片库有 {len(assets)} 张照片。"}

    @staticmethod
    def public_job(job):
        public = {k: v for k, v in job.items() if k not in {"cancel", "rankings", "query", "started"}}
        public["match_summary"] = match_summary(job.get("results", []))
        return public

    def results(self, assets, order, scores, sources, inspections, conditions):
        lookup = {a["id"]: a for a in assets}
        results = []
        for rank, aid in enumerate(order):
            a = lookup[aid]
            r = {k: a[k] for k in ["id", "name", "path", "width", "height", "captured", "description"]}
            r.update(thumbnail="/api/assets/"+aid+"/thumbnail", score=float(scores.get(aid, 0)),
                rank_order=rank, reasons=["候选来源：" + "、".join(sources.get(aid, []))], engine="hybrid")
            if aid in inspections:
                r["inspection"] = normalize_inspection(inspections[aid])
                r.update(inspection_grade(inspections[aid], conditions))
            r.update(match_decision(r, conditions))
            results.append(r)
        return results

    def publish_inspections(self,session_id,key,updates,conditions=None):
        e=self.engine
        with e.job_lock:
            job=e.verifications.get(session_id)
            if not job or job.get("pipeline")!="vnext" or job.get("key")!=key:return
            lookup={r["id"]:dict(r) for r in job["results"]}
            for update in updates:
                if update["id"] in lookup:
                    lookup[update["id"]].update(update,inspection_conflict=update.get("inspection_tier")==0)
            conditions=conditions if conditions is not None else job.get("conditions",[])
            for result in lookup.values():result.update(match_decision(result,conditions))
            base=job.get("base_order")
            base=base or [r["id"] for r in job["results"]]
            order=evidence_order(base,lookup)
            for rank,aid in enumerate(order):lookup[aid]["rank_order"]=rank
            job.update(results=[lookup[aid] for aid in order],revision=job.get("revision",0)+1,
                confirmed=sum(r.get("inspection_tier")==2 for r in lookup.values()))

    def ranking_snapshot(self,session_id):
        with self.engine.job_lock:
            job=self.engine.verifications.get(session_id)
            if not job or job.get("pipeline")!="vnext":return None
            return {"key":job["key"],"revision":job.get("revision",0),"ids":[r["id"] for r in job["results"]]}

    def refine(self, session, assets, inspections, job, budget=None):
        e = self.engine
        if e.setting("hybrid_search",True) and not job.get("force_core"):
            return e.hybrid.refine(self,session,assets,inspections,job)
        adaptive = budget is None
        try:
            recall_started = time.time()
            rankings, (order, scores, sources) = self.recall(assets, session["conditions"], use_main=True)
            if job["cancel"].is_set():
                job["status"] = "paused"
                return
            job["recall_ms"] = round((time.time()-recall_started)*1000)
            job["first_preview_ready"] = False
            job["recall_count"] = len(order)
            job["rankings"] = rankings
            results = self.results(assets, order, scores, sources, inspections, session["conditions"])
            lookup = {r["id"]: r for r in results}
            asset_lookup = {a["id"]: a for a in assets}
            with e.db() as con:
                detections = {(r["asset_id"],r["fingerprint"]):json.loads(r["result"]) for r in con.execute(
                    "SELECT asset_id,fingerprint,result FROM detections WHERE version=?", (e.detector.version,))}
            for result in results:
                asset = asset_lookup[result["id"]]
                self.apply_cached_counts(result, asset, session["conditions"], detections.get((asset["id"],asset["fingerprint"]),{}))
            eligible = [aid for aid in order if not lookup[aid].get("fact_conflict")]
            eligible_set = set(eligible)
            filtered_rankings = {name: [aid for aid in ids if aid in eligible_set] for name, ids in rankings.items()}
            job["base_order"] = list(order)
            initial_order = evidence_order(order, lookup)
            for rank,aid in enumerate(initial_order):
                lookup[aid]["rank_order"] = rank
            results = [lookup[aid] for aid in initial_order]
            job["results"] = results
            budget = min(budget or int(e.setting("rerank_budget", 64)), len(eligible))
            main_query = query_text(session["conditions"], relaxed=True, english=True)
            if main_query == "A photograph":
                main_query = query_text(session["conditions"], english=True)
            query_hash = hashlib.sha256(main_query.encode()).hexdigest()
            with e.db() as con:
                cached_scores = {(r["asset_id"],r["fingerprint"]):float(r["score"]) for r in con.execute(
                    "SELECT asset_id,fingerprint,score FROM rerank_cache WHERE version=? AND query=?",(self.reranker.version,query_hash))}
            selected = select_candidates(filtered_rankings, eligible, budget) if budget else []
            # Expansion retains previous evidence and spends new capacity on new images.
            previous = ([aid for aid in eligible if (aid,asset_lookup[aid]["fingerprint"]) in cached_scores]
                        if job.get("expanding") else [])
            selected = list(dict.fromkeys(previous+selected+eligible))[:budget]
            job.update(stage="rerank", total=len(selected), remaining=len(order))
            if not self.reranker.installed:
                raise RuntimeError("精排模型尚未准备，当前保留全库召回结果")
            ranked = {}
            job.update(cache_hits=0,inference_ms=0,automatic_extra=0)
            queue = list(selected)
            expanded = False
            batch_size = max(1,min(2,int(e.setting("rerank_batch_size",2))))
            while queue:
                if job["cancel"].is_set():
                    job["status"] = "paused"
                    break
                batch = [queue.pop(0) for _ in range(min(batch_size,len(queue)))]
                missing = [aid for aid in batch if (aid,asset_lookup[aid]["fingerprint"]) not in cached_scores]
                before = time.time()
                if missing:
                    paths = [str(e.asset_file(aid)[0]) for aid in missing]
                    values = ([self.reranker.score(main_query,paths[0])] if len(paths)==1
                              else self.reranker.score_many(main_query,paths))
                    elapsed = (time.time()-before)*1000
                    job["inference_ms"] += round(elapsed)
                    with e.db() as con:
                        for aid,score in zip(missing,values):
                            cached_scores[(aid,asset_lookup[aid]["fingerprint"])] = float(score)
                            con.execute("INSERT OR REPLACE INTO rerank_cache VALUES (?,?,?,?,?,?)", (aid,
                                asset_lookup[aid]["fingerprint"],self.reranker.version,query_hash,float(score),elapsed/len(missing)))
                if job["cancel"].is_set():
                    job["status"]="paused"
                    break
                for aid in batch:
                    if aid not in missing:
                        job["cache_hits"] += 1
                    score = cached_scores[(aid,asset_lookup[aid]["fingerprint"])]
                    ranked[aid] = score
                    lookup[aid].update(rerank_score=score,reranked=True,
                        relevance_evidence="positive" if score>=.5 else "weak")
                    job["done"] += 1
                visible = merge_reranked(order,ranked,lookup)
                # Store the CORE baseline so later manual checks do not compound
                # previous inspection offsets each time a batch is published.
                clean_lookup={aid:{} for aid in order}
                job["base_order"] = merge_reranked(order,ranked,clean_lookup)
                for rank,aid in enumerate(visible):
                    lookup[aid]["rank_order"] = rank
                job["results"] = [lookup[aid] for aid in visible]
                job["remaining"] = len(order)-job["done"]
                job["elapsed_ms"] = round((time.time()-job["started"])*1000)
                if not job["first_preview_ready"] and job["done"]>=min(16,len(selected)):
                    job["first_preview_ready"] = True
                    job["first_preview_ms"] = job["elapsed_ms"]
                # One bounded extension: investigate neighbors of positive evidence and
                # preserve other retrieval entries. No asset gets a match from its neighbor.
                if not queue and adaptive and not expanded and not job["cancel"].is_set():
                    expanded = True
                    positives = [aid for aid in ranked if ranked[aid]>=.5]
                    if len(positives)>=max(4,budget//4) or len(positives)<4:
                        remaining = [aid for aid in eligible if aid not in ranked]
                        cap = max(budget,min(128,int(e.setting("rerank_auto_max",96))))
                        extra_budget = min(32,cap-len(ranked),len(remaining))
                        if extra_budget>0:
                            with e.db() as con:
                                main_rows = [dict(r) for r in con.execute("""SELECT v.* FROM feature_vectors v JOIN assets a
                                    ON a.id=v.asset_id AND a.fingerprint=v.fingerprint
                                    WHERE a.library_id=? AND v.version=? AND v.region=-1 AND a.error IS NULL""",
                                    (assets[0]["library_id"],self.main.version))]
                            if len(main_rows)!=len(assets):main_rows=[]
                            neighbors = similar_candidates(assets,positives,remaining,main_rows)[:extra_budget//2]
                            remaining_set = set(remaining)
                            other_rankings = {name:[aid for aid in ids if aid in remaining_set] for name,ids in filtered_rankings.items()}
                            broader = select_candidates(other_rankings,remaining,extra_budget)
                            queue = list(dict.fromkeys(neighbors+broader+remaining))[:extra_budget]
                            job["automatic_extra"] = len(queue)
                            job["total"] += len(queue)
            if job["status"] == "running":
                job["status"] = "complete"
            # Detailed inspection is a bounded explicit action, never an unbounded automatic loop.
            job["stage"] = "complete"
            job["confirmed"] = sum(r.get("inspection_tier") == 2 for r in job["results"])
        except Exception as exc:
            job["status"] = "error"
            job["errors"].append(str(exc)[:300])
        finally:
            self.main.unload()
            self.reranker.unload()
            job["elapsed_ms"] = round((time.time()-job["started"])*1000)
            with e.db() as con:
                report = {"status": job["status"], "models": [self.main.version,self.reranker.version],
                    "indexed": job["indexed"], "candidate_ids": list(job.get("rankings", {}).values()),
                    "ranked_ids": [r["id"] for r in job["results"]], "reranked": job["done"],
                    "elapsed_ms": job["elapsed_ms"], "first_preview_ms":job.get("first_preview_ms"),
                    "inference_ms":job.get("inference_ms"),"automatic_extra":job.get("automatic_extra",0),"errors": job["errors"]}
                con.execute("INSERT INTO search_runs VALUES (?,?,?,?,?)", (uuid.uuid4().hex, session["id"],job["query"],json.dumps(report,ensure_ascii=False),time.time()))

    def apply_cached_counts(self, result, asset, conditions, detection=None):
        e = self.engine
        if detection is None:
            with e.db() as con:
                cached = con.execute("SELECT result FROM detections WHERE asset_id=? AND fingerprint=? AND version=?",
                    (asset["id"],asset["fingerprint"],e.detector.version)).fetchone()
            detection = json.loads(cached[0]) if cached else {}
        facts = {key: {"count": None, **value, "evidence": "已缓存检测提供的可见数量下限"}
                 for key, value in detection.items()}
        checked = count_evidence(conditions, facts)
        for term,c in zip(condition_terms(conditions),conditions):
            if c.get("concept") == "year" and c["kind"] != "prefer" and not c.get("soft"):
                matches = bool(asset["captured"] and asset["captured"].startswith(c["label"]))
                status = "unknown" if not asset["captured"] else ("matched" if matches else "contradicted")
                if c["kind"] == "exclude" and status != "unknown":
                    status = "contradicted" if matches else "matched"
                checked[term] = {"term":term,"status":status,"evidence":"记录的拍摄时间："+(asset["captured"] or "未知")}
        if any(item["status"] == "contradicted" for term,item in checked.items()
               if any(t == term and c["kind"] != "prefer" and not c.get("soft") for t,c in zip(condition_terms(conditions),conditions))):
            inspection = {"summary": "检测到的数量与明确条件冲突", "conditions": [checked.get(term,
                {"term": term,"status":"unknown","evidence":"尚未核对"}) for term in condition_terms(conditions)]}
            result["inspection"] = inspection
            result.update(inspection_grade(inspection, conditions))
            result["fact_conflict"]=True
        result.update(match_decision(result, conditions))

    def expand(self, session_id, budget):
        old = self.engine.verifications.get(session_id)
        if not old or old.get("pipeline") != "vnext":
            raise ValueError("请先进行新版搜索")
        if old["status"] == "running":
            raise ValueError("当前精排尚在进行，可先暂停")
        with self.engine.db() as con:
            s = con.execute("SELECT library_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not s:
                raise ValueError("搜索会话不存在")
            assets = [dict(r) for r in con.execute("SELECT * FROM assets WHERE library_id=? AND error IS NULL", (s[0],))]
        session = self.engine.session(s[0], session_id)
        key = self.engine.inspection_key(session["conditions"])
        with self.engine.db() as con:
            inspections = {r["asset_id"]: json.loads(r["result"]) for r in con.execute(
                """SELECT i.* FROM inspections i JOIN assets a ON a.id=i.asset_id AND a.fingerprint=i.fingerprint
                WHERE a.library_id=? AND i.query=?""", (s[0],key))}
        old.update(status="running",stage="recall",done=0,errors=[],cancel=threading.Event(),started=time.time(),expanding=True,force_core=True,hybrid=False)
        threading.Thread(target=self.refine,args=(session,assets,inspections,old,budget),daemon=True).start()
        return self.public_job(old)
