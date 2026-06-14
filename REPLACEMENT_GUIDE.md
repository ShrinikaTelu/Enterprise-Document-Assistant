# How to apply this update to Enterprise-Document-Assistant

## 1. DELETE these files from the repo
    PROJECT_SUMMARY.md
    QUICK_START.md
    VERIFICATION_CHECKLIST.md
    VERIFICATION_REPORT.md
    scripts/setup_index.py
    src/vector_store.py
    src/state.py

## 2. REPLACE / ADD everything in this zip (structure below)
    README.md            <- full rewrite (honest claims, real mermaid diagram)
    requirements.txt     <- 5 lean deps (was 10 incl. heavy langchain/vertex)
    .env.example         <- 1 key (was 5 GCP infra vars)
    .gitignore
    src/config.py        <- env-based settings
    src/providers.py     <- Gemini REST + FakeProvider behind one interface
    src/store.py         <- in-memory numpy vector store
    src/graph.py         <- Self-RAG graph: + real grounded web search,
                            + real check_grounding node, singletons fixed
    src/main.py          <- /ask now returns sources + grounded flag; /health
    docs/                <- 3 sample documents (works out of the box)
    static/index.html    <- visualizer UI served at / (pipeline trace,
                            grounding badges, sources) — no build step
    scripts/evaluate.py  <- REAL naive-vs-self-RAG harness (replaces dummy)
    tests/               <- 4 tests, run with: FAKE_LLM=1 python -m pytest tests/

## 3. After pushing — two required follow-ups
    a) Run the real eval and paste YOUR output into the README results section:
         GEMINI_API_KEY=... python scripts/evaluate.py
       The old README's RAGAS table (0.72/0.95/...) was not produced by the
       code in the repo — never restore it. Only publish numbers you ran.
    b) Update the repo description (Settings -> About):
       "Self-correcting agentic RAG — LangGraph state machine with retrieval
        grading, grounded web-search fallback, and a naive-vs-self-RAG eval
        harness. Runs with one free Gemini key."

## 4. Verify before pushing (30 seconds, no API key needed)
    pip install -r requirements.txt pytest
    FAKE_LLM=1 python -m pytest tests/     # 4 passed
    FAKE_LLM=1 python scripts/evaluate.py  # table prints

## UPDATE (visualizer added)
Also add/replace:
    static/index.html    <- single-page state-machine visualizer (served at /)
    src/main.py          <- now serves static/ at root, /ask returns `trace`
    src/graph.py         <- every node appends a trace entry (Annotated reducer)
The /ask response gained a `trace` field; the UI animates it. No new deps.
