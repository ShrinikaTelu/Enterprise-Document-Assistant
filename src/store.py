"""In-memory vector store: numpy cosine similarity over document chunks.

Replaces the previous Vertex AI Vector Search dependency for the default
mode — at portfolio/document scale (hundreds of chunks), exact cosine search
in numpy is simpler, free, and instant. The retrieval interface is the seam:
an enterprise deployment swaps this module for Vertex AI Vector Search and
the graph is untouched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import CHUNK_CHARS, DOCS_DIR, TOP_K, logger


@dataclass
class Chunk:
    doc: str
    text: str


class VectorStore:
    def __init__(self, provider) -> None:
        self.provider = provider
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def ingest_dir(self, path: str = DOCS_DIR) -> int:
        if not os.path.exists(path):
            logger.warning("Docs directory not found: %s", path)
            return 0
        
        texts: list[Chunk] = []
        for fname in sorted(os.listdir(path)):
            if not fname.endswith((".md", ".txt")):
                continue
            try:
                with open(os.path.join(path, fname), encoding="utf-8") as fh:
                    raw = fh.read()
                for chunk in self._split(raw):
                    texts.append(Chunk(doc=fname, text=chunk))
            except Exception as e:
                logger.warning("Failed to read %s: %s", fname, e)
                continue
        
        if not texts:
            logger.warning("No documents found in %s", path)
            return 0
            
        self.chunks = texts
        vecs = self.provider.embed([c.text for c in texts])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.matrix = vecs / np.maximum(norms, 1e-9)
        logger.info("Ingested %d chunks from %d files", len(texts),
                    len({c.doc for c in texts}))
        return len(texts)

    @staticmethod
    def _split(text: str) -> list[str]:
        """Paragraph-aware splitting with a max chunk size."""
        out, buf = [], ""
        for para in text.split("\n\n"):
            if len(buf) + len(para) > CHUNK_CHARS and buf:
                out.append(buf.strip())
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            out.append(buf.strip())
        return out

    def search(self, query: str, k: int = TOP_K) -> list[Chunk]:
        if self.matrix is None or not self.chunks:
            return []
        q = self.provider.embed([query])[0]
        q = q / max(np.linalg.norm(q), 1e-9)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in top]
