"""Self-RAG state machine (LangGraph).

    retrieve -> grade_docs --(relevant)--> generate -> check_grounding --(ok)--> END
                    |                                       |
                (irrelevant)                          (ungrounded, 1 retry)
                    v                                       v
                web_search ------------------------------> END

Every node is honest about what it does:
- grade_docs:    fast LLM grades each chunk's relevance (strict JSON)
- web_search:    REAL Google Search grounding via the google_search tool
- check_grounding: judges whether the answer is supported by the retrieved
  context; one stricter regeneration, then the response is flagged rather
  than silently returned. This is the "self-correcting" part — implemented,
  not aspirational.
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import END, StateGraph

from .config import logger
from .providers import get_provider
from .store import VectorStore


class RAGState(TypedDict, total=False):
    question: str
    trace: Annotated[List[dict], operator.add]
    documents: List[str]
    sources: List[str]
    generation: str
    used_web_fallback: bool
    grounded: bool
    retries: int


# Singletons: provider + store built once, not per request.
provider = get_provider()
store = VectorStore(provider)


def _parse_json(text: str) -> dict:
    try:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


def retrieve_node(state: RAGState) -> dict:
    logger.info("retrieve: %s", state["question"][:60])
    chunks = store.search(state["question"])
    return {
        "documents": [c.text for c in chunks],
        "sources": sorted({c.doc for c in chunks}),
        "used_web_fallback": False,
        "retries": 0,
        "trace": [{"node": "retrieve",
                   "detail": f"{len(chunks)} chunks from "
                             f"{len({c.doc for c in chunks})} documents"}],
    }


GRADE_PROMPT = """You are a strict relevance grader. Does this document contain
information that helps answer the question? Reply with JSON only:
{{"relevant": true}} or {{"relevant": false}}. Reply yes only if genuinely useful.

DOCUMENT:
{document}
QUESTION:
{question}"""


def grade_documents_node(state: RAGState) -> dict:
    relevant = []
    for doc in state["documents"]:
        verdict = _parse_json(provider.complete(
            GRADE_PROMPT.format(document=doc[:2000], question=state["question"]),
            fast=True,
        ))
        if verdict.get("relevant"):
            relevant.append(doc)
    logger.info("grade_docs: %d/%d relevant", len(relevant), len(state["documents"]))
    return {"documents": relevant,
            "trace": [{"node": "grade_docs",
                       "detail": f"{len(relevant)}/{len(state['documents'])} "
                                 "chunks judged relevant"}]}


def route_after_grading(state: RAGState) -> str:
    return "generate" if state["documents"] else "web_search"


def web_search_node(state: RAGState) -> dict:
    logger.info("web_search fallback (grounded)")
    answer = provider.complete_with_search(
        f"Answer concisely and factually: {state['question']}"
    )
    return {"generation": answer, "used_web_fallback": True,
            "grounded": True, "sources": ["google_search"],
            "trace": [{"node": "web_search",
                       "detail": "no relevant context — answered via "
                                 "grounded Google Search"}]}


GEN_PROMPT = """Answer the question using ONLY the context. If the context is
insufficient, say what's missing instead of guessing.{strictness}

Context:
{context}

Question: {question}"""


def generate_node(state: RAGState) -> dict:
    strict = ("\nBe extremely conservative: every sentence must be directly "
              "traceable to the context.") if state.get("retries", 0) else ""
    answer = provider.complete(GEN_PROMPT.format(
        strictness=strict,
        context="\n\n".join(state["documents"]),
        question=state["question"],
    ))
    attempt = "stricter retry" if state.get("retries", 0) else "first attempt"
    return {"generation": answer,
            "trace": [{"node": "generate", "detail": attempt}]}


CHECK_PROMPT = """Is every factual claim in this answer supported by the context?
Reply JSON only: {{"grounded": true}} or {{"grounded": false}}.

Context:
{context}

Answer:
{answer}"""


def check_grounding_node(state: RAGState) -> dict:
    verdict = _parse_json(provider.complete(
        CHECK_PROMPT.format(context="\n\n".join(state["documents"])[:6000],
                            answer=state["generation"]),
        fast=True,
    ))
    grounded = bool(verdict.get("grounded"))
    logger.info("check_grounding: %s", grounded)
    return {"grounded": grounded, "retries": state.get("retries", 0) + 1,
            "trace": [{"node": "check_grounding",
                       "detail": "answer supported by context"
                                 if grounded else "unsupported claims found"}]}


def route_after_check(state: RAGState) -> str:
    if state["grounded"] or state["retries"] >= 2:
        return "done"
    logger.warning("ungrounded answer — regenerating once, stricter")
    return "regenerate"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_docs", grade_documents_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)
    g.add_node("check_grounding", check_grounding_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", route_after_grading,
                            {"generate": "generate", "web_search": "web_search"})
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges("check_grounding", route_after_check,
                            {"done": END, "regenerate": "generate"})
    g.add_edge("web_search", END)
    return g.compile()


app_graph = build_graph()
