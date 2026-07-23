# Architecture

```
                 ┌──────────────┐
  documents ───► │  chunking    │  overlapping windows, traceable ids
                 └──────┬───────┘
                        ▼
                 ┌───────────────────────────────┐
                 │  hybrid retrieval             │
     question ──►│  BM25 (lexical) ┐             │
                 │  TF-IDF (semantic) ┘─► RRF ─► re-rank ─► top-k
                 └──────┬────────────────────────┘
                        ▼  chunks + calibrated cosine scores
                 ┌──────────────┐   grounded?  no ─► refuse (+ HITL)
                 │  grounding   │──────────────────────────────────►
                 └──────┬───────┘
                        ▼ yes
                 ┌──────────────┐
                 │ prompt (vN)  │  versioned template: context + question
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │     LLM      │  OpenAI (if key) | deterministic offline model
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  guardrails  │  PII redaction + HITL flag (low-conf/high-stakes)
                 └──────┬───────┘
                        ▼
              answer + sources + flags
```

## Key modules
- `app/chunking.py` — recursive character splitter with overlap.
- `app/retrieval.py` — TF-IDF (semantic) vector store + cosine similarity.
- `app/bm25.py` — Okapi BM25 lexical retriever.
- `app/fusion.py` — Reciprocal Rank Fusion.
- `app/rerank.py` — second-stage re-rankers: `LexicalReRanker` (default) and
  `CrossEncoderReRanker` (neural, production), chosen via `make_reranker`.
- `app/hybrid.py` — hybrid retriever = TF-IDF + BM25 → RRF → re-rank.
- `app/backends/langchain_chroma.py` — optional LangChain + Chroma + real embeddings.
- `app/llm.py` — OpenAI or deterministic offline model, same interface.
- `app/guardrails.py` — PII redaction + grounding check.
- `app/rag.py` — orchestrates the pipeline; owns the HITL policy.
- `app/main.py` — FastAPI (`/health`, `/ask`).
- `eval/` — golden set, scorers, report generator.

## Why two backends?
The default backend has zero heavy dependencies so the project is reproducible
and testable anywhere. The production backend shows the same design on the
industry-standard stack (LangChain + Chroma + sentence-transformers/OpenAI).
Both satisfy the same `retrieve(query, k)` interface, so swapping them changes
nothing downstream — a clean seam and a demonstration of dependency inversion.

## Roadmap (natural next steps)
- ✅ Hybrid retrieval (BM25 + TF-IDF) with Reciprocal Rank Fusion — **done**.
- ✅ Second-stage re-ranking — **done** (lexical default + neural cross-encoder).
- Dense embeddings by default (sentence-transformers) via the Chroma backend.
- Streaming responses (SSE) and multi-turn memory.
- LLM-as-judge scorer in the eval harness (faithfulness/groundedness).
- Auth, per-tenant indexes, and observability (traces + metrics).
