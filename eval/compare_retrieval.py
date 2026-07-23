"""Compare retrieval strategies on the golden set.

Runs each strategy — semantic-only, lexical-only, hybrid (RRF), and
hybrid + re-rank — and reports retrieval hit-rate and MRR (mean reciprocal
rank). This is how you justify a retrieval change with evidence instead of
vibes.

Usage:
    python -m eval.compare_retrieval
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import DATA_DIR, Settings
from app.rag import build_retriever, load_documents
from eval.scorers import reciprocal_rank, retrieval_hit

GOLDEN = Path(__file__).resolve().parent / "golden.json"

MODES = [
    ("semantic (tf-idf)", Settings(retrieval_mode="tfidf")),
    ("lexical (bm25)", Settings(retrieval_mode="bm25")),
    ("hybrid (rrf)", Settings(retrieval_mode="hybrid", use_rerank=False)),
    ("hybrid + rerank", Settings(retrieval_mode="hybrid", use_rerank=True)),
]


def evaluate(retriever, items, top_k: int) -> dict:
    hits, rrs = [], []
    for it in items:
        if it["expected_source"] is None:
            continue  # retrieval metrics only apply to answerable questions
        sources = [rc.chunk.source for rc in retriever.retrieve(it["question"], k=top_k)]
        hits.append(retrieval_hit(it["expected_source"], sources))
        rrs.append(reciprocal_rank(it["expected_source"], sources))
    return {
        "hit_rate": round(sum(1 for h in hits if h) / len(hits), 3),
        "mrr": round(sum(rrs) / len(rrs), 3),
    }


def main() -> None:
    items = json.loads(GOLDEN.read_text(encoding="utf-8"))["items"]
    chunks = load_documents(DATA_DIR)

    print(f"{'strategy':<22}{'hit@k':>8}{'MRR':>8}")
    print("-" * 38)
    for name, cfg in MODES:
        retriever = build_retriever(chunks, cfg)
        m = evaluate(retriever, items, cfg.top_k)
        print(f"{name:<22}{m['hit_rate']:>8.3f}{m['mrr']:>8.3f}")


if __name__ == "__main__":
    main()
