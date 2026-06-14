"""Test suite — runs entirely on FakeProvider (FAKE_LLM=1), no network.

Covers: chunking/search math, every graph route (generate path, web
fallback path), and the API contract.
"""
import os

os.environ["FAKE_LLM"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from src.graph import app_graph, store  # noqa: E402
from src.main import app  # noqa: E402


def setup_module():
    store.ingest_dir("docs")


def test_ingest_and_search():
    assert len(store.chunks) >= 3
    hits = store.search("Q3 revenue", k=2)
    assert hits and any("4.2" in h.text for h in hits)


def test_graph_generate_path():
    out = app_graph.invoke({"question": "What was the Q3 2025 revenue figure?"})
    assert out["generation"]
    assert out["used_web_fallback"] is False
    assert out["sources"]


def test_graph_web_fallback_path():
    out = app_graph.invoke({"question": "zzqx unrelatedterm plutonium recipes"})
    assert out["used_web_fallback"] is True
    assert out["sources"] == ["google_search"]
    assert [t["node"] for t in out["trace"]] == ["retrieve", "grade_docs", "web_search"]


def test_api_contract():
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["ok"] and h["chunks"] > 0
        r = client.post("/ask", json={"question": "remote work days for engineers?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"answer", "sources", "used_web_fallback",
                             "grounded", "provider", "trace"}
        assert [t["node"] for t in body["trace"]][0] == "retrieve"
