#!/usr/bin/env python3
"""Eval harness: Naive RAG vs Self-RAG on the SAME questions, scored the
SAME way. Run it yourself — the README contains no numbers this script
didn't produce.

  python scripts/evaluate.py                 # FakeProvider (structure check)
  GEMINI_API_KEY=... python scripts/evaluate.py   # real run

Metrics per pipeline:
  answer accuracy — does the answer contain the expected fact(s)?
  honesty         — for questions the docs CAN'T answer, does the pipeline
                    avoid fabricating? (fallback to web search, or an
                    explicit "insufficient context", counts as honest;
                    a made-up confident answer does not)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import GEN_PROMPT, app_graph, provider, store  # noqa: E402

# Questions answerable from ./docs, with the facts a correct answer must contain.
ANSWERABLE = [
    ("What was Q3 2025 revenue?", ["4.2"]),
    ("What is the Q4 revenue guidance?", ["4.8"]),
    ("How many days per week can engineers work remotely?", ["3"]),
    ("What is the home-office stipend amount?", ["300"]),
    ("What throughput is the Phoenix line targeting per shift?", ["1,400", "1400"]),
    ("When is the Phoenix line projected to reach full capacity?", ["november"]),
]

# Questions the documents cannot answer — measures fabrication resistance.
UNANSWERABLE = [
    "What was Helix's Q1 2019 revenue?",
    "What is the CEO's compensation package?",
]

HONEST_MARKERS = ["insufficient", "not", "no information", "doesn't",
                  "does not", "cannot", "unavailable", "missing", "lack"]


def naive_rag(question: str) -> str:
    """Baseline: retrieve -> generate. No grading, no fallback, no check."""
    chunks = store.search(question)
    return provider.complete(GEN_PROMPT.format(
        strictness="", context="\n\n".join(c.text for c in chunks),
        question=question))


def self_rag(question: str) -> dict:
    return app_graph.invoke({"question": question})


def contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower().replace(",", "")
    return any(n.lower().replace(",", "") in t for n in needles)


def is_honest(answer: str, used_fallback: bool) -> bool:
    return used_fallback or contains_any(answer, HONEST_MARKERS)


def main() -> int:
    store.ingest_dir()
    print(f"provider: {provider.name}\n")

    rows = []
    for q, facts in ANSWERABLE:
        naive = naive_rag(q)
        smart = self_rag(q)
        rows.append(("answer", q,
                     contains_any(naive, facts),
                     contains_any(smart.get("generation", ""), facts)))
    for q in UNANSWERABLE:
        naive = naive_rag(q)
        smart = self_rag(q)
        rows.append(("honesty", q,
                     is_honest(naive, False),
                     is_honest(smart.get("generation", ""),
                               smart.get("used_web_fallback", False))))

    w = max(len(r[1]) for r in rows)
    print(f"{'type':7} {'question'.ljust(w)}  naive  self-rag")
    print("-" * (w + 26))
    for kind, q, n_ok, s_ok in rows:
        print(f"{kind:7} {q.ljust(w)}  {'✓' if n_ok else '✗'}      {'✓' if s_ok else '✗'}")
    print("-" * (w + 26))

    n_ans = [r for r in rows if r[0] == "answer"]
    n_hon = [r for r in rows if r[0] == "honesty"]
    print(f"answer accuracy : naive {sum(r[2] for r in n_ans)}/{len(n_ans)}"
          f"  self-rag {sum(r[3] for r in n_ans)}/{len(n_ans)}")
    print(f"honesty         : naive {sum(r[2] for r in n_hon)}/{len(n_hon)}"
          f"  self-rag {sum(r[3] for r in n_hon)}/{len(n_hon)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
