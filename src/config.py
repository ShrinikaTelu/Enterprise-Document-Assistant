"""Configuration. One required secret: GEMINI_API_KEY (free tier works).

Set FAKE_LLM=1 to run the whole system with a deterministic fake provider —
used by the test suite and useful for offline development.
"""
import logging
import os
from pathlib import Path

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FAKE_LLM = os.environ.get("FAKE_LLM", "0") == "1"

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
FAST_MODEL = os.environ.get("FAST_MODEL", "gemini-2.0-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")

# Resolve docs directory relative to project root
_ROOT = Path(__file__).parent.parent
DOCS_DIR = os.environ.get("DOCS_DIR", str(_ROOT / "docs"))
TOP_K = int(os.environ.get("TOP_K", "4"))
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "900"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("self-rag")
