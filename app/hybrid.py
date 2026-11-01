"""Offline observations + OCR, multilingual text retrieval, bounded local checks.

No query-dependent facts are propagated to neighboring photos. Raw OCR and model
descriptions keep separate provenance and never become mandatory filters.
"""
import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
import numpy as np
from .providers import ollama
from .providers.ocr import LocalOCR
from .providers.text_embedding import TextEmbedding
from .retrieval import condition_terms, inspection_grade
from .ranking import evidence_order


def tokens(text):
    text=str(text).casefold()
    words=re.findall(r"[a-z0-9]+",text)
    for run in re.findall(r"[\u3400-\u9fff]+",text):
        words.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run)==1:words.append(run)
    return list(dict.fromkeys(words))


def observation_text(result):
    return "\n".join(f"{key}: {value.strip()}" for key,value in result.items()
        if isinstance(value,str) and value.strip().lower() not in {"","unknown","none","unclear"})[:8000]


class HybridIndex:
    def __init__(self,engine):
        self.e=engine
        from .engine import ROOT
        self.ocr=LocalOCR(ROOT)
        self.embedding=TextEmbedding()
        self.last_search=0
        self.foreground_count=0
        self.foreground_lock=threading.Lock()
        self.build_lock=threading.Lock()
        with engine.db() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS observations (
                    asset_id TEXT, fingerprint TEXT, kind TEXT, version TEXT, payload TEXT, text TEXT,
                    vector BLOB, vector_version TEXT, created REAL, PRIMARY KEY(asset_id,kind));
                CREATE VIRTUAL TABLE IF NOT EXISTS observation_fts USING fts5(
                    asset_id UNINDEXED,kind UNINDEXED,content,tokenize='unicode61');
                CREATE TABLE IF NOT EXISTS hybrid_query_vectors (
                    version TEXT,query TEXT,vector BLOB,PRIMARY KEY(version,query));
                CREATE TABLE IF NOT EXISTS observation_failures (
                    asset_id TEXT,kind TEXT,error TEXT,attempts INTEGER,updated REAL,PRIMARY KEY(asset_id,kind));
            """)

    def coverage(self,library_id):
        with self.e.db() as con:
            rows=con.execute("""SELECT o.kind,COUNT(*) AS n,SUM(CASE WHEN o.vector_version=? THEN 1 ELSE 0 END) AS vectors
                FROM observations o JOIN assets a ON a.id=o.asset_id AND a.fingerprint=o.fingerprint
                WHERE a.library_id=? AND a.error IS NULL GROUP BY o.kind""",(self.embedding.version,library_id)).fetchall()
        info={"caption":0,"ocr":0,"text_vectors":0}
        for r in rows:
            info[r["kind"]]=r["n"]
            if r["kind"]=="caption":info["text_vectors"]=r["vectors"] or 0
        return info

    def save(self,asset,kind,version,payload,text):
        with self.e.db() as con:
            current=con.execute("SELECT fingerprint FROM assets WHERE id=?",(asset["id"],)).fetchone()
            if not current or current[0]!=asset["fingerprint"]:return
            con.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,NULL,NULL,?)",
                (asset["id"],asset["fingerprint"],kind,version,json.dumps(payload,ensure_ascii=False),text,time.time()))
            con.execute("DELETE FROM observation_fts WHERE asset_id=? AND kind=?",(asset["id"],kind))
            con.execute("INSERT INTO observation_fts VALUES (?,?,?)",(asset["id"],kind," ".join(tokens(text))))
            con.execute("DELETE FROM observation_failures WHERE asset_id=? AND kind=?",(asset["id"],kind))

    def describe_safely(self,path,model):
        errors=[]
        for candidate in dict.fromkeys([model,ollama.MODEL]):
            try:
                payload=ollama.describe(path,model=candidate,keep_alive=60)
                if not isinstance(payload.get("caption"),str) or not payload["caption"].strip():
                    raise ValueError("没有返回有效画面描述")
                payload["provenance"]={"model":candidate,"fallback":candidate!=model}
                return payload
            except (RuntimeError,ValueError) as exc:
                errors.append(candidate+"："+str(exc)[:160])
        raise ValueError("该图描述失败，保留原图检索："+"；".join(errors))

    def start(self,library_id):
        e=self.e
        if e.library(library_id)["is_demo"]:raise ValueError("请选择真实照片库")
        with e.job_lock:
            if any(j.get("status") in {"running","scanning"} for j in e.jobs.values()):
                raise ValueError("已有建库任务，请等待或暂停后继续")
            job={"id":uuid.uuid4().hex,"library_id":library_id,"task":"hybrid","status":"running",
                "total":0,"done":0,"processed":0,"skipped":0,"removed":0,"errors":[],
                "current":"准备本地描述、OCR与文字向量","started":time.time(),"cancel":threading.Event()}
            e.jobs[library_id]=job
            threading.Thread(target=self.build,args=(library_id,job),daemon=True).start()
        return e.public_job(job)

    def _wait(self,job):
        # Online work has priority between images, never interrupt an unsafe GPU operation.
        while (self.foreground_count or time.monotonic()-self.last_search<3 or any(
                j.get("status")=="running" and not j["cancel"].is_set() for j in self.e.verifications.values())):
            if job["cancel"].wait(.3):return False
        return not job["cancel"].is_set()

    @contextmanager
    def foreground(self):
        with self.foreground_lock:self.foreground_count+=1
        try:yield
        finally:
            with self.foreground_lock:
                self.foreground_count-=1
                self.last_search=time.monotonic()

    def embed_pending(self,library_id,job):
        with self.e.db() as con:
            rows=[dict(r) for r in con.execute("""SELECT o.* FROM observations o JOIN assets a
                ON a.id=o.asset_id AND a.fingerprint=o.fingerprint WHERE a.library_id=? AND o.kind='caption'
                AND (o.vector IS NULL OR o.vector_version IS NULL OR o.vector_version!=?)""",
                (library_id,self.embedding.version))]
        for offset in range(0,len(rows),12):
            if not self._wait(job):return False
            batch=rows[offset:offset+12];job['current']=f'建立文字向量 · {offset+len(batch)} / {len(rows)}'
            vectors=self.embedding.encode([r['text'] for r in batch])
            with self.e.db() as con:
                for row,vector in zip(batch,vectors):
                    con.execute("UPDATE observations SET vector=?,vector_version=? WHERE asset_id=? AND kind='caption' AND fingerprint=? AND version=?",
                        (vector.tobytes(),self.embedding.version,row['asset_id'],row['fingerprint'],row['version']))
            job['processed']+=len(batch);job['done']+=len(batch)
            job['embedded_this_run']=job.get('embedded_this_run',0)+len(batch)
        return True

    def build(self,library_id,job):
        e=self.e
        try:
            with self.build_lock:
                with e.db() as con:
                    assets=[dict(r) for r in con.execute("SELECT * FROM assets WHERE library_id=? AND error IS NULL",(library_id,))]
                model=ollama.description_model()
                version=model+":describe-original1536-v3"
                job["total"]=len(assets)*3
                job["photo_total"]=len(assets)
                # Visual meaning must become searchable before optional OCR.
                for kind in ["caption","ocr"]:
                    if kind=="ocr":
                        if not self.embed_pending(library_id,job):job["status"]="paused";return
                        job["visual_ready"]=True
                    job["stage"]=kind
                    for asset in assets:
                        if not self._wait(job):job["status"]="paused";return
                        job["current"]=("识别文字 · " if kind=="ocr" else "理解画面 · ")+asset["name"]
                        expected=self.ocr.version if kind=="ocr" else version
                        with e.db() as con:
                            old=con.execute("SELECT version FROM observations WHERE asset_id=? AND fingerprint=? AND kind=?",
                                (asset["id"],asset["fingerprint"],kind)).fetchone()
                        # Preserve valid previous OCR, including empty results;
                        # swapping the preferred engine is not a reason to redo it.
                        if old and (kind=="ocr" or old[0]==expected):
                            job["skipped"]+=1;job["done"]+=1;continue
                        try:
                            path,_=e.asset_file(asset["id"])
                            before=path.stat()
                            if before.st_size!=asset["size"] or before.st_mtime_ns!=asset["mtime_ns"]:
                                raise ValueError("原图已变化，请先更新照片文件索引")
                            if kind=="ocr":
                                payload=self.ocr.read(path)
                                text="\n".join(line["text"] for line in payload["lines"] if line.get("confidence",0)>=.3)
                                expected=payload["version"]
                            else:
                                payload=self.describe_safely(path,model)
                                if not str(payload.get("caption","")).strip():raise RuntimeError("没有返回有效画面描述")
                                text=observation_text(payload)
                            after=path.stat()
                            if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
                                raise ValueError("原图在处理时变化，未保存结果")
                            self.save(asset,kind,expected,payload,text)
                            if kind=="caption":
                                with e.db() as con:
                                    con.execute("INSERT OR REPLACE INTO annotations (asset_id,fingerprint,model,result,embedding,created,field_vectors) VALUES (?,?,?,?,NULL,?,NULL)",
                                        (asset["id"],asset["fingerprint"],version,json.dumps(payload,ensure_ascii=False),time.time()))
                            job["processed"]+=1
                            if kind=="caption":
                                job["caption_since_embed"]=job.get("caption_since_embed",0)+1
                                if job["caption_since_embed"]>=32:
                                    if not self.embed_pending(library_id,job):job["status"]="paused";return
                                    job["caption_since_embed"]=0
                        except Exception as exc:
                            job["errors"].append({"file":asset["name"],"message":kind+"："+str(exc)[:240]})
                            with e.db() as con:
                                con.execute("""INSERT INTO observation_failures VALUES (?,?,?,1,?)
                                    ON CONFLICT(asset_id,kind) DO UPDATE SET error=excluded.error,
                                    attempts=attempts+1,updated=excluded.updated""",(asset["id"],kind,str(exc)[:500],time.time()))
                            if kind!="ocr" and not isinstance(exc,(FileNotFoundError,ValueError)):
                                raise
                        job["done"]+=1
                # Publish semantic observations in batches during captioning; finish any remainder.
                with e.db() as con:
                    existing=con.execute("""SELECT COUNT(*) FROM observations o JOIN assets a ON a.id=o.asset_id
                        AND a.fingerprint=o.fingerprint WHERE a.library_id=? AND o.kind='caption'
                        AND o.vector IS NOT NULL AND o.vector_version=?""",(library_id,self.embedding.version)).fetchone()[0]
                reused=max(0,existing-job.get("embedded_this_run",0))
                job["skipped"]+=reused;job["done"]+=reused
                if not self.embed_pending(library_id,job):job["status"]="paused";return
                job.update(status="complete",current="画面描述、照片文字与混合索引已更新")
        except Exception as exc:
            job["status"]="error";job["errors"].append({"file":"","message":str(exc)[:300]})
        finally:
            self.ocr.close()
            job["finished"]=time.time()

    def rankings(self,assets,conditions,semantic=True):
        if not assets:return {}
        words=" ".join(c.get("english",c["label"]) for c in conditions if c["kind"]!="exclude")
        chinese=" ".join(c["label"] for c in conditions if c["kind"]!="exclude")
        terms=tokens(words+" "+chinese)[:64]
        rankings={}
        with self.e.db() as con:
            for kind,label in [("caption","描述关键词"),("ocr","照片文字")]:
                if terms:
                    expression=" OR ".join('"'+t+'"' for t in terms)
                    rows=con.execute("""SELECT f.asset_id,bm25(observation_fts) AS score FROM observation_fts f
                        JOIN assets a ON a.id=f.asset_id JOIN observations o ON o.asset_id=a.id AND o.kind=f.kind
                        AND o.fingerprint=a.fingerprint WHERE observation_fts MATCH ? AND f.kind=?
                        AND a.library_id=? AND a.error IS NULL ORDER BY score LIMIT 500""",
                        (expression,kind,assets[0]["library_id"])).fetchall()
                    if rows:rankings[label]=[r["asset_id"] for r in rows]
            rows=[dict(r) for r in con.execute("""SELECT o.* FROM observations o JOIN assets a ON a.id=o.asset_id
                AND a.fingerprint=o.fingerprint WHERE a.library_id=? AND o.kind='caption' AND o.vector_version=?""",
                (assets[0]["library_id"],self.embedding.version))]
        if semantic and rows and words:
            query=chinese or words
            with self.e.db() as con:
                cached=con.execute("SELECT vector FROM hybrid_query_vectors WHERE version=? AND query=?",(self.embedding.version,query)).fetchone()
            if cached:vector=np.frombuffer(cached[0],dtype=np.float32)
            else:
                vector=self.embedding.encode([query],query=True)[0]
                with self.e.db() as con:
                    con.execute("INSERT OR REPLACE INTO hybrid_query_vectors VALUES (?,?,?)",(self.embedding.version,query,vector.tobytes()))
            matrix=np.stack([np.frombuffer(r["vector"],dtype=np.float32) for r in rows])
            rankings["描述语义"]=[rows[i]["asset_id"] for i in np.argsort(-(matrix@vector),kind="stable")[:500]]
        return rankings

    def refine(self,pipeline,session,assets,inspections,job):
        e=self.e
        try:
            rankings,(base,scores,sources)=pipeline.recall(assets,session["conditions"],use_main=True)
            if job["cancel"].is_set():job["status"]="paused";return
            results=pipeline.results(assets,base,scores,sources,inspections,session["conditions"])
            lookup={r["id"]:r for r in results};asset_lookup={a["id"]:a for a in assets}
            with e.db() as con:
                detections={(r["asset_id"],r["fingerprint"]):json.loads(r["result"]) for r in con.execute(
                    "SELECT asset_id,fingerprint,result FROM detections WHERE version=?",(e.detector.version,))}
            for r in results:
                asset=asset_lookup[r["id"]]
                pipeline.apply_cached_counts(r,asset,session["conditions"],detections.get((r["id"],asset["fingerprint"]),{}))
            order=evidence_order(base,lookup)
            for rank,aid in enumerate(order):lookup[aid]["rank_order"]=rank
            with e.job_lock:
                job.update(results=[lookup[aid] for aid in order],recall_count=len(order),remaining=len(order),
                    base_order=list(base),rankings=rankings,revision=job.get("revision",0)+1,
                    first_preview_ready=True,first_preview_ms=round((time.time()-job["started"])*1000),stage="local-check")
            hard=[c for c in session["conditions"] if c["kind"]!="prefer" and not c.get("soft") and c.get("concept")!="year"]
            # A single scene clause can still contain an action and location; do not silently skip it.
            count=min(8,int(e.setting("hybrid_verify_budget",4))) if hard and e.setting("hybrid_auto_verify",True) else 0
            eligible=[aid for aid in order if not lookup[aid].get("fact_conflict")]
            selected=eligible[:min(2,count)]
            for ids in rankings.values():
                if len(selected)>=count:break
                candidate=next((aid for aid in ids if aid in lookup and aid not in selected and aid in eligible[:96]),None)
                if candidate:selected.append(candidate)
            selected=list(dict.fromkeys(selected+eligible))[:max(0,count)]
            job["total"]=len(selected)
            key=e.inspection_key(session["conditions"])
            for aid in selected:
                if job["cancel"].is_set():job["status"]="paused";return
                e.inspect_candidates(session["id"],[aid],expected_key=key)
                if job["cancel"].is_set():job["status"]="paused";return
                job.update(done=job["done"]+1,elapsed_ms=round((time.time()-job["started"])*1000))
            job.update(status="complete",stage="complete",remaining=len(order)-job["done"],
                confirmed=sum(r.get("inspection_tier")==2 for r in job["results"]))
        except ValueError as exc:
            job["status"]="paused" if str(exc)=="搜索条件已改变" else "error"
            if job["status"]=="error":job["errors"].append(str(exc)[:300])
        except Exception as exc:
            job["status"]="error";job["errors"].append(str(exc)[:300])
        finally:
            pipeline.main.unload();e.provider.unload()
            job["elapsed_ms"]=round((time.time()-job["started"])*1000)
            with e.db() as con:
                report={"mode":"hybrid-v2","status":job["status"],"coverage":job.get("observation_coverage"),
                    "ranked_ids":[r["id"] for r in job["results"]],"base_order":job.get("base_order",[]),
                    "checked":job["done"],"revision":job.get("revision",0),"elapsed_ms":job["elapsed_ms"],"errors":job["errors"]}
                con.execute("INSERT INTO search_runs VALUES (?,?,?,?,?)",(uuid.uuid4().hex,session["id"],
                    "；".join(c["label"] for c in session["conditions"]),json.dumps(report,ensure_ascii=False),time.time()))
