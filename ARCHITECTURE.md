# Architecture

```
                 ┌──────────────┐
  documents ───► │  chunking    │  overlapping windows, traceable ids
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  retrieval   │  TF-IDF vectors + cosine  (or LangChain+Chroma)
                 └──────┬───────┘
     question ─────────►│  top-k chunks + scores
                        ▼
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
- `app/retrieval.py` — TF-IDF vector store + cosine similarity (default backend).
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
- Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion.
- Cross-encoder re-ranking of retrieved chunks.
- Streaming responses (SSE) and multi-turn memory.
- LLM-as-judge scorer in the eval harness (faithfulness/groundedness).
- Auth, per-tenant indexes, and observability (traces + metrics).
