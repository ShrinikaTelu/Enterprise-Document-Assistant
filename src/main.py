"""FastAPI layer. Startup ingests ./docs into the in-memory store and
serves the visualizer UI (static/index.html) at /."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import logger
from .graph import app_graph, provider, store


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        n = store.ingest_dir()
        logger.info("ready: provider=%s chunks=%d", provider.name, n)
    except Exception as e:
        logger.error("Failed to ingest docs: %s", e)
        # Continue anyway - API will work but with empty store
    yield


app = FastAPI(title="Self-RAG Document Assistant", version="2.0",
              lifespan=lifespan)


class Ask(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[str]
    used_web_fallback: bool
    grounded: bool
    provider: str
    trace: list[dict]


@app.get("/health")
def health():
    return {"ok": True, "provider": provider.name, "chunks": len(store.chunks)}


@app.post("/ask", response_model=Answer)
def ask(body: Ask):
    try:
        final = app_graph.invoke({"question": body.question})
    except Exception as exc:
        logger.exception("graph failure")
        raise HTTPException(500, f"pipeline error: {exc}") from exc
    return Answer(
        answer=final.get("generation", ""),
        sources=final.get("sources", []),
        used_web_fallback=final.get("used_web_fallback", False),
        grounded=final.get("grounded", True),
        provider=provider.name,
        trace=final.get("trace", []),
    )


_static = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
