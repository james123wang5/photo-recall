import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine, ROOT
from .providers import ollama

app = FastAPI(title="PhotoRecall", version="0.6.0")
engine = Engine()
TOKEN = secrets.token_urlsafe(24)


@app.middleware("http")
async def local_requests(request: Request, call_next):
    host = request.url.hostname
    if host not in {"localhost", "127.0.0.1", "testserver"}:
        return JSONResponse({"detail": "此版本仅接受本机访问"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and (urlparse(origin).hostname not in {"localhost", "127.0.0.1"} or urlparse(origin).netloc != request.url.netloc):
        return JSONResponse({"detail": "请求来源不匹配"}, status_code=403)
    if request.method == "POST" and request.headers.get("x-photorecall-token") != TOKEN:
        return JSONResponse({"detail": "页面已过期，请刷新后重试"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
    return response


@app.exception_handler(ValueError)
async def validation_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(RuntimeError)
async def provider_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "flow.html")


@app.get("/legacy")
def legacy_home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/state")
def state():
    return {"token": TOKEN, "libraries": engine.libraries(), "semantic_enabled": engine.setting("semantic"),
            "semantic_installed": engine.provider.installed, "version": "0.6.0", "local_only": True,
            "retrieval_vnext": engine.setting("retrieval_vnext"),"hybrid_search":engine.setting("hybrid_search",True)}


@app.get("/api/models")
def models():
    installed = ollama.installed_models()
    return {"hybrid":{"enabled":engine.setting("hybrid_search",True),"ocr":engine.hybrid.ocr.status(),
                "description_model":ollama.description_model(),"text_embedding":engine.hybrid.embedding.model,
                "text_embedding_ready":engine.hybrid.embedding.installed,"verify_budget":engine.setting("hybrid_verify_budget",4)},
            "siglip": {"installed": engine.provider.installed, "enabled": engine.setting("semantic"), "name": engine.provider.name},
            "ollama": {"installed_models": installed, "ready": ollama.MODEL in installed, "name": ollama.MODEL},
            "detector": {"ready": engine.detector.ready, "name": "YOLO26 nano"},
            "planner": {"ready": ollama.TEXT_MODEL in installed, "name": ollama.TEXT_MODEL},
            "retrieval": {"enabled": engine.setting("retrieval_vnext"),
                "main": {"installed": engine.pipeline.main.installed, "name": engine.pipeline.main.name},
                "reranker": {"installed": engine.pipeline.reranker.installed, "name": engine.pipeline.reranker.name},
                "rerank_budget": engine.setting("rerank_budget", 64)}}


class FolderInput(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


@app.post("/api/libraries")
def add_library(body: FolderInput):
    return engine.add_library(body.path)


@app.post("/api/pick-folder")
def pick_folder():
    if sys.platform != "darwin":
        raise ValueError("此系统请直接粘贴文件夹路径")
    try:
        result = subprocess.run(["osascript", "-e", 'POSIX path of (choose folder with prompt "选择要索引的照片文件夹")'], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise ValueError("选择超时，请重试")
    return {"path": result.stdout.strip() if result.returncode == 0 else "", "cancelled": result.returncode != 0}


@app.post("/api/libraries/{library_id}/index")
def start_index(library_id: str):
    return engine.start_index(library_id)


@app.post("/api/libraries/{library_id}/analysis")
def start_analysis(library_id: str):
    return engine.hybrid.start(library_id)


@app.post("/api/libraries/{library_id}/hybrid-index")
def hybrid_index(library_id: str):
    return engine.hybrid.start(library_id)


@app.post("/api/libraries/{library_id}/retrieval-index")
def retrieval_index(library_id: str):
    return engine.pipeline.start_index(library_id)


@app.get("/api/libraries/{library_id}/job")
def job(library_id: str):
    engine.library(library_id)
    found = engine.jobs.get(library_id)
    return engine.public_job(found) if found else {"status": "idle"}


@app.post("/api/libraries/{library_id}/pause")
def pause(library_id: str):
    found = engine.jobs.get(library_id)
    if found:
        found["cancel"].set()
    return {"ok": True}


class SearchInput(BaseModel):
    library_id: str
    session_id: str | None = None
    message: str = Field(default="", max_length=1000)
    mode: str = "append"
    remove_id: str | None = None
    limit: int = Field(default=200, ge=1, le=1000)


@app.post("/api/search")
def search(body: SearchInput):
    if body.mode not in {"append", "replace", "reset"}:
        raise ValueError("未知搜索方式")
    return engine.search(body.library_id, body.message, body.session_id, body.mode, body.remove_id, body.limit)


@app.get("/api/assets/{asset_id}/thumbnail")
def thumbnail(asset_id: str):
    path, _ = engine.asset_file(asset_id, thumbnail=True)
    return FileResponse(path)


@app.get("/api/assets/{asset_id}/original")
def original(asset_id: str):
    path, _ = engine.asset_file(asset_id)
    return FileResponse(path)


@app.post("/api/assets/{asset_id}/reveal")
def reveal(asset_id: str):
    path, asset = engine.asset_file(asset_id)
    if asset["is_demo"]:
        raise ValueError("这是演示插画")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=True)
    return {"path": str(path)}


class InspectionInput(BaseModel):
    session_id: str
    asset_ids: list[str] = Field(max_length=8)


@app.post("/api/inspect")
def inspect_candidates(body: InspectionInput):
    results=engine.inspect_candidates(body.session_id, body.asset_ids)
    return {"results":results,"ranking":engine.pipeline.ranking_snapshot(body.session_id)}


@app.get("/api/sessions/{session_id}/verification")
def verification(session_id: str):
    return engine.verification_state(session_id)


@app.post("/api/sessions/{session_id}/verification/pause")
def pause_verification(session_id: str):
    found = engine.verifications.get(session_id)
    if found:
        found["cancel"].set()
    return {"ok": True}


class RerankInput(BaseModel):
    budget: int = Field(default=128, ge=8, le=256)


@app.post("/api/sessions/{session_id}/rerank")
def expand_rerank(session_id: str, body: RerankInput):
    return engine.pipeline.expand(session_id, body.budget)


class SettingsInput(BaseModel):
    semantic: bool | None = None
    retrieval_vnext: bool | None = None
    hybrid_search: bool | None = None
    hybrid_auto_verify: bool | None = None


@app.post("/api/settings")
def settings(body: SettingsInput):
    if body.semantic is not None:engine.enable_semantic(body.semantic)
    with engine.db() as con:
        for key,value in [("hybrid_search",body.hybrid_search),("hybrid_auto_verify",body.hybrid_auto_verify)]:
            if value is not None:con.execute("INSERT OR REPLACE INTO settings VALUES (?,?)",(key,"true" if value else "false"))
    if body.retrieval_vnext is not None:
        if body.retrieval_vnext and not (engine.pipeline.main.installed and engine.pipeline.reranker.installed):
            raise RuntimeError("新版模型尚未完整准备")
        with engine.db() as con:
            con.execute("INSERT OR REPLACE INTO settings VALUES ('retrieval_vnext', ?)", ("true" if body.retrieval_vnext else "false",))
    return {"ok": True, "semantic_enabled": engine.setting("semantic")}


from .flow.api import router_for
flow_router, flow = router_for(engine)
app.include_router(flow_router)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
