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

1. **Retrieves** the most relevant chunks via **hybrid search** — BM25 (lexical) +
   TF-IDF (semantic) fused with **Reciprocal Rank Fusion**, then **re-ranked**.
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

# 2) Compare retrieval strategies (semantic / lexical / hybrid / hybrid+rerank)
python -m eval.compare_retrieval

# 3) Run the tests
pytest -q

# 4) Serve the API
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
| **MRR** (mean reciprocal rank) | *How highly* did we rank the right chunk? (ordering-sensitive) |
| **Answer accuracy** | Does the answer contain the required facts? |
| **Refusal accuracy** | Did we correctly refuse out-of-scope questions? |

Because the offline model is deterministic, this is a **regression test for AI
quality** — change a prompt or retrieval setting, re-run, and see the score move.
Try it: set `PROMPT_VERSION=qa_v1` in `.env` and compare.

## Retrieval: hybrid search + re-ranking

Retrieval runs in two stages (configurable via `RETRIEVAL_MODE` / `USE_RERANK`):

1. **Hybrid first-stage** — a **BM25** lexical ranking and a **TF-IDF** semantic
   ranking are fused with **Reciprocal Rank Fusion (RRF)**. Fusing by *rank*
   (not raw score) combines two signals with incomparable scales and is robust
   when one retriever misses a phrasing the other catches.
2. **Re-ranking** — a second-stage re-ranker rescoring the fused shortlist for
   precision at the top. The default is a dependency-free lexical re-ranker; the
   production stack swaps in a cross-encoder (`requirements-prod.txt`).

Compare strategies yourself:

```
$ python -m eval.compare_retrieval
strategy               hit@k     MRR
--------------------------------------
semantic (tf-idf)      1.000   1.000
lexical (bm25)         1.000   1.000
hybrid (rrf)           1.000   1.000
hybrid + rerank        1.000   1.000
```

> On this compact knowledge base every strategy retrieves correctly, so the
> scores tie — the point is the **measurement harness** and the **MRR** metric
> (sensitive to ordering, unlike a binary hit@k). Hybrid + re-ranking is the
> robust default that pays off as a corpus grows larger and noisier; the harness
> is what lets you justify the choice with evidence.

## How it maps to real job requirements

| Requirement seen in JDs | Where it lives here |
|---|---|
| RAG pipeline: chunking, embeddings, vector store, retrieval | [`chunking.py`](app/chunking.py), [`retrieval.py`](app/retrieval.py), [`backends/langchain_chroma.py`](app/backends/langchain_chroma.py) |
| **Hybrid search + Reciprocal Rank Fusion** | [`bm25.py`](app/bm25.py), [`fusion.py`](app/fusion.py), [`hybrid.py`](app/hybrid.py) |
| **Re-ranking** (+ cross-encoder path) | [`rerank.py`](app/rerank.py) |
| Retrieval metrics (hit-rate, **MRR**) + A/B harness | [`eval/scorers.py`](eval/scorers.py), [`eval/compare_retrieval.py`](eval/compare_retrieval.py) |
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
