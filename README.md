# RAG Eval Service

A production-style **Retrieval-Augmented Generation (RAG)** microservice with a
built-in **evaluation harness**, prompt versioning, safety guardrails, and a
human-in-the-loop (HITL) gate.

Built to demonstrate reliable LLM-application engineering: not just *generating*
answers, but *retrieving* the right context, *grounding* the model, *refusing*
when it shouldn't answer, and **measuring quality with a reproducible eval**.

> Runs out of the box with **no API key** and **no heavy ML dependencies** — the
> default backend uses a self-contained TF-IDF vector store and a deterministic
> offline model, so the whole pipeline (including the eval harness and tests) is
> reproducible anywhere. An optional [production backend](app/backends/langchain_chroma.py)
> swaps in **LangChain + Chroma + real embeddings** and **OpenAI** generation.

## What it does

Ask a question against a small knowledge base (a fictional "AcmePay" wallet). The
service:

1. **Retrieves** the most relevant chunks (TF-IDF cosine similarity / vector search).
2. **Grounds** the answer — refuses instead of hallucinating when nothing relevant is found.
3. **Generates** an answer from a **versioned prompt template**.
4. **Guardrails** the output — redacts PII, and flags high-stakes or low-confidence answers for **human review (HITL)**.
5. Returns the answer **with its sources**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Run the evaluation harness (writes eval/report.md)
python -m eval.run_eval

# 2) Run the tests
pytest -q

# 3) Serve the API
uvicorn app.main:app --reload
```

Then:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "How much does the Premium plan cost?"}'
```

Example response:

```json
{
  "answer": "The Premium plan costs 9.99 USD per month ...",
  "grounded": true,
  "prompt_version": "qa_v2",
  "sources": [{"id": "acmepay_fees.md#0", "source": "acmepay_fees.md", "score": 0.31, "preview": "..."}],
  "needs_human_review": false
}
```

## Evaluation harness (the important part)

`python -m eval.run_eval` runs the pipeline over a **golden set** ([eval/golden.json](eval/golden.json))
and scores three metrics, then writes [eval/report.md](eval/report.md):

| Metric | Question it answers |
|---|---|
| **Retrieval hit-rate** | Did we retrieve the doc that actually contains the answer? |
| **Answer accuracy** | Does the answer contain the required facts? |
| **Refusal accuracy** | Did we correctly refuse out-of-scope questions? |

Because the offline model is deterministic, this is a **regression test for AI
quality** — change a prompt or retrieval setting, re-run, and see the score move.
Try it: set `PROMPT_VERSION=qa_v1` in `.env` and compare.

## How it maps to real job requirements

| Requirement seen in JDs | Where it lives here |
|---|---|
| RAG pipeline: chunking, embeddings, vector store, retrieval | [`chunking.py`](app/chunking.py), [`retrieval.py`](app/retrieval.py), [`backends/langchain_chroma.py`](app/backends/langchain_chroma.py) |
| Prompt engineering + **versioning** | [`prompts/`](app/prompts), [`config.py`](app/config.py) |
| Guardrails, PII handling, responsible AI | [`guardrails.py`](app/guardrails.py) |
| Human-in-the-loop for high-stakes actions | HITL flag in [`rag.py`](app/rag.py) |
| **Evaluation pipelines** (quality/accuracy) | [`eval/`](eval) |
| FastAPI, Pydantic, REST, tested code | [`main.py`](app/main.py), [`schemas.py`](app/schemas.py), [`tests/`](tests) |
| LangChain / Chroma / vector DBs | [`backends/langchain_chroma.py`](app/backends/langchain_chroma.py) |

## Production backend (optional)

```bash
pip install -r requirements-prod.txt   # LangChain + Chroma + sentence-transformers
```

See [`app/backends/langchain_chroma.py`](app/backends/langchain_chroma.py) — same
`retrieve()` interface, real embeddings + a Chroma vector store.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) and the design spec in [SPEC.md](SPEC.md)
(this project was built spec-first).

## License

MIT — see [LICENSE](LICENSE).
