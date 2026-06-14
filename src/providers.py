"""LLM + embedding providers behind one small interface.

GeminiProvider talks to the Gemini API over REST (httpx, no SDK churn) and
implements REAL Google Search grounding via the google_search tool.
FakeProvider is deterministic — it powers the test suite and FAKE_LLM mode.

Swapping in Vertex AI for an enterprise deployment means implementing these
three methods against the Vertex endpoints; the graph never changes.
"""
from __future__ import annotations

import hashlib
import json
import re

import httpx
import numpy as np

from .config import EMBED_MODEL, FAKE_LLM, FAST_MODEL, GEMINI_API_KEY, LLM_MODEL

BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider:
    name = "gemini"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        with httpx.Client(timeout=30) as client:
            for text in texts:
                r = client.post(
                    f"{BASE}/models/{EMBED_MODEL}:embedContent",
                    params={"key": GEMINI_API_KEY},
                    json={"content": {"parts": [{"text": text[:8000]}]}},
                )
                r.raise_for_status()
                vecs.append(r.json()["embedding"]["values"])
        return np.array(vecs, dtype=np.float32)

    def complete(self, prompt: str, fast: bool = False) -> str:
        model = FAST_MODEL if fast else LLM_MODEL
        r = httpx.post(
            f"{BASE}/models/{model}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def complete_with_search(self, prompt: str) -> str:
        """Grounded generation — the model actually calls Google Search."""
        r = httpx.post(
            f"{BASE}/models/{LLM_MODEL}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],
            },
            timeout=60,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


class FakeProvider:
    """Deterministic stand-in: hashed bag-of-words embeddings, templated
    answers. Good enough to exercise every graph path in tests."""

    name = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        dim = 64
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    def complete(self, prompt: str, fast: bool = False) -> str:
        if "grade" in prompt.lower() and "yes" in prompt.lower():
            # Relevance grader: overlap heuristic on the embedded doc/question.
            m = re.search(r"DOCUMENT:\n(.*?)\nQUESTION:\n(.*)", prompt, re.S)
            if m:
                doc_words = set(re.findall(r"[a-z]{4,}", m.group(1).lower()))
                q_words = set(re.findall(r"[a-z]{4,}", m.group(2).lower()))
                return json.dumps({"relevant": bool(doc_words & q_words)})
            return json.dumps({"relevant": True})
        if "supported by the context" in prompt.lower():
            return json.dumps({"grounded": True})
        m = re.search(r"Context:\n(.*?)\n\nQuestion", prompt, re.S)
        snippet = (m.group(1)[:180] if m else "no context")
        return f"[fake answer based on context: {snippet}...]"

    def complete_with_search(self, prompt: str) -> str:
        return "[fake web-grounded answer]"


def get_provider():
    return FakeProvider() if (FAKE_LLM or not GEMINI_API_KEY) else GeminiProvider()
