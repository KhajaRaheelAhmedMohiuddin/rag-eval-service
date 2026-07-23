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


class ReRanker:
    """Interface: reorder candidates and return the top_k."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:  # pragma: no cover
        raise NotImplementedError


class LexicalReRanker(ReRanker):
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


class CrossEncoderReRanker(ReRanker):
    """Neural cross-encoder re-ranker (production path).

    A cross-encoder jointly encodes each (query, passage) pair and outputs a
    single relevance score, so it captures interaction between the query and the
    passage far better than the lexical re-ranker or first-stage retrieval.
    It's more accurate and more expensive — which is exactly why it runs only on
    the small fused shortlist, not the whole corpus.

    Requires `sentence-transformers` (see requirements-prod.txt). The import is
    lazy so the default, dependency-free path never loads it.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # heavy; imported lazily

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, rc.chunk.text) for rc in candidates]
        scores = self._model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        # Preserve each candidate's calibrated .score; only the order changes.
        return [candidates[i] for i in order[:top_k]]


def make_reranker(kind: str = "lexical", model: str | None = None) -> ReRanker | None:
    """Factory: select a re-ranker by name.

    kind: "lexical" (default, dependency-free), "cross_encoder" (production),
    or "none"/None to disable re-ranking.
    """
    if kind in (None, "none", ""):
        return None
    if kind == "lexical":
        return LexicalReRanker()
    if kind == "cross_encoder":
        return CrossEncoderReRanker(model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    raise ValueError(f"Unknown reranker kind: {kind!r}")
