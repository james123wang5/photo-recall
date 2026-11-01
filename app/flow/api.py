"""HTTP contracts for the full workflow. No model or image is run at import."""
from typing import Literal
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from .service import FlowService
from .indexing import FlowIndexer


class Reference(BaseModel):
    asset_id: str
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)


class SearchRequest(BaseModel):
    library_id: str
    message: str = Field(default="", max_length=2000)
    session_id: str | None = None
    expected_revision: int = Field(default=0, ge=0)
    mode: Literal["find_one", "set", "all"] = "find_one"
    replace: bool = False
    remove_id: str | None = None
    reference: Reference | None = None
    exclude_id: str | None = None
    action: Literal["search", "recover", "reset"] = "search"
    expression: dict | None = None
    scope: list[str] | None = Field(default=None, max_length=100000)
    person_id: str | None = None


class ResumeRequest(BaseModel):
    asset_ids: list[str] | None = Field(default=None, max_length=8)
    regions: dict[str, list[list[float]]] | None = None
    retry_failed: bool = False


class IndexRequest(BaseModel):
    components: list[Literal["visual", "ocr", "description", "people"]] = Field(default_factory=lambda: ["visual"])


class NameRequest(BaseModel):
    name: str = Field(max_length=80)


def router_for(engine):
    router = APIRouter(prefix="/api/flow")
    flow = FlowService(engine)
    indexer = FlowIndexer(flow)

    @router.get("/config")
    def config():
        cfg = flow.store.config()
        primary, reranker = flow.recall.primary(cfg), flow.recall.reranker(cfg)
        return {"config": cfg, "models": {"primary": {"name": primary.name, "ready": primary.installed, "version": primary.version},
            "reranker": {"name": reranker.name, "ready": reranker.installed, "version": reranker.version},
            "siglip": {"name": engine.provider.name, "ready": engine.provider.installed},
            "vision": {"name": cfg["vision_model"], "signature": flow.tools.vision_signature(cfg["vision_model"])},
            "ocr": engine.hybrid.ocr.status(), "grounding_ready": flow.tools.grounding.installed,
            "faces_ready": flow.tools.faces.installed}}

    @router.post("/config")
    def configure(values: dict):
        return {"config": flow.configure(values)}

    @router.post("/search")
    def search(body: SearchRequest):
        values = body.model_dump()
        return flow.start(**values)

    @router.get("/runs/{run_id}")
    def run(run_id: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=60, ge=1, le=200),
            filter_by: str = "all", expected_ranking: int | None = None):
        return flow.view(run_id, offset, limit, filter_by, expected_ranking)

    @router.post("/runs/{run_id}/pause")
    def pause(run_id: str):
        return flow.pause(run_id)

    @router.post("/runs/{run_id}/continue")
    def resume(run_id: str, body: ResumeRequest):
        return flow.resume(run_id, body.asset_ids, body.regions, body.retry_failed)

    @router.get("/runs/{run_id}/actions")
    def actions(run_id: str, after: int = Query(default=0, ge=0)):
        flow.store.get_run(run_id)
        return {"actions": flow.store.actions(run_id, after)}

    @router.get("/libraries/{library_id}/sessions")
    def sessions(library_id: str):
        engine.library(library_id)
        with engine.db() as con:
            rows = con.execute("SELECT id,current_run,payload,updated FROM flow_sessions WHERE library_id=? ORDER BY updated DESC LIMIT 20", (library_id,)).fetchall()
        import json
        return {"sessions": [{"id": r["id"], "current_run": r["current_run"], "updated": r["updated"],
                               "query": json.loads(r["payload"])} for r in rows]}

    @router.post("/libraries/{library_id}/scan")
    def scan(library_id: str):
        return engine.start_index(library_id, scan_only=True)

    @router.post("/libraries/{library_id}/index")
    def index(library_id: str, body: IndexRequest):
        return indexer.start(library_id, body.components)

    @router.get("/libraries/{library_id}/coverage")
    def coverage(library_id: str):
        engine.library(library_id)
        with engine.db() as con:
            rows = con.execute("""SELECT s.component,s.status,COUNT(*) AS count FROM flow_index_state s
                JOIN assets a ON a.id=s.asset_id AND a.fingerprint=s.fingerprint WHERE a.library_id=? GROUP BY s.component,s.status""", (library_id,)).fetchall()
        return {"components": [dict(r) for r in rows], "legacy": engine.pipeline.counts(library_id),
                "observations": engine.hybrid.coverage(library_id)}

    @router.get("/libraries/{library_id}/people")
    def people(library_id: str):
        engine.library(library_id)
        with engine.db() as con:
            rows = con.execute("""SELECT p.id,p.name,COUNT(DISTINCT f.asset_id) AS count,MIN(f.asset_id) AS example
                FROM flow_people p JOIN flow_faces f ON f.person_id=p.id JOIN assets a ON a.id=f.asset_id
                AND a.fingerprint=f.fingerprint WHERE a.library_id=? GROUP BY p.id ORDER BY count DESC""", (library_id,)).fetchall()
        return {"people": [dict(r) for r in rows], "note": "相似聚类可能出错；名称仅由用户提供"}

    @router.post("/people/{person_id}/name")
    def name(person_id: str, body: NameRequest):
        with engine.db() as con:
            if not con.execute("SELECT 1 FROM flow_people WHERE id=?", (person_id,)).fetchone():
                raise ValueError("人物组不存在")
            con.execute("UPDATE flow_people SET name=? WHERE id=?", (body.name.strip(), person_id))
        return {"ok": True}

    return router, flow
