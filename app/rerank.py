"""Second-stage re-ranking.

First-stage retrieval (hybrid) is tuned for recall — get the right chunks into a
candidate shortlist cheaply. A re-ranker then applies a sharper, more expensive
relevance signal to that short list to improve precision at the top.

This dependency-free re-ranker scores candidates by query-term coverage plus a
phrase/bigram bonus. The production stack swaps in a cross-encoder (e.g.
sentence-transformers CrossEncoder), which jointly encodes (query, passage) for
much stronger precision — same interface (see app/backends/langchain_chroma.py
notes and requirements-prod.txt).
"""
from __future__ import annotations

from app.retrieval import RetrievedChunk, tokenize


class LexicalReRanker:
    def _score(self, query_tokens: list[str], text: str) -> float:
        q_set = set(query_tokens)
        q_bigrams = set(zip(query_tokens, query_tokens[1:]))
        toks = tokenize(text)
        coverage = len(q_set & set(toks))
        bigrams = set(zip(toks, toks[1:]))
        bigram_bonus = len(q_bigrams & bigrams)
        phrase_bonus = 1.0 if " ".join(query_tokens) in " ".join(toks) else 0.0
        return coverage + 0.5 * bigram_bonus + 1.0 * phrase_bonus

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Reorder candidates by rerank score; preserve their calibrated .score."""
        q = tokenize(query)
        # Stable sort keyed by rerank score; ties keep first-stage order.
        ranked = sorted(
            candidates,
            key=lambda rc: self._score(q, rc.chunk.text),
            reverse=True,
        )
        return ranked[:top_k]
