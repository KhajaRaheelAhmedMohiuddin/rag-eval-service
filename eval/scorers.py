"""Scoring functions for the evaluation harness.

Three complementary metrics, mirroring how production RAG systems are judged:
  * retrieval_hit  — did we retrieve the document that contains the answer?
  * answer_correct — does the answer contain the required facts (keywords)?
  * refusal_correct— did we correctly refuse out-of-scope questions?
"""
from __future__ import annotations


def retrieval_hit(expected_source: str | None, retrieved_sources: list[str]) -> bool | None:
    """True if the expected source doc is among the retrieved chunks.

    Returns None for items with no expected source (out-of-scope questions),
    so they're excluded from the retrieval metric rather than counted as misses.
    """
    if expected_source is None:
        return None
    return expected_source in retrieved_sources


def reciprocal_rank(expected_source: str | None, retrieved_sources: list[str]) -> float | None:
    """1/rank of the first chunk from the expected source (0 if never retrieved).

    More sensitive than a binary hit@k: it rewards ranking the right source
    higher, so it distinguishes retrieval strategies that all "hit" but order
    results differently. Averaged across items this is MRR.
    """
    if expected_source is None:
        return None
    for i, src in enumerate(retrieved_sources):
        if src == expected_source:
            return 1.0 / (i + 1)
    return 0.0


def answer_correct(answer: str, keywords: list[str]) -> bool:
    """True if every required keyword appears in the answer (case-insensitive)."""
    if not keywords:
        return True
    text = answer.lower()
    return all(kw.lower() in text for kw in keywords)


def refusal_correct(should_refuse: bool, grounded: bool) -> bool:
    """A correct refusal: refused when it should, answered when it shouldn't."""
    refused = not grounded
    return refused == should_refuse


def aggregate(rows: list[dict]) -> dict:
    def rate(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else 0.0

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "n": len(rows),
        "retrieval_hit_rate": rate("retrieval_hit"),
        "mrr": mean("reciprocal_rank"),
        "answer_accuracy": rate("answer_correct"),
        "refusal_accuracy": rate("refusal_correct"),
    }
