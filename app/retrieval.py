"""Vector retrieval.

Default backend: a dependency-free TF-IDF vector store with cosine similarity.
It computes real embeddings (TF-IDF vectors), indexes them, and ranks by cosine
distance — the same mechanics as a production vector database, with zero heavy
dependencies so the project runs anywhere.

The optional production backend (see app/backends/langchain_chroma.py) swaps this
for HuggingFace/OpenAI embeddings + a Chroma vector store behind the same
`Retriever` interface.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.chunking import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfRetriever:
    """A minimal but real TF-IDF + cosine-similarity vector store."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._docs_tokens = [tokenize(c.text) for c in chunks]
        self._idf = self._compute_idf(self._docs_tokens)
        self._vectors = [self._vectorize(toks) for toks in self._docs_tokens]

    @staticmethod
    def _compute_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
        n = len(docs_tokens) or 1
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        # Smoothed idf keeps common terms from dominating.
        return {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        vec = {t: (count / length) * self._idf.get(t, 0.0) for t, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        # Vectors are unit-normalized, so the dot product is the cosine.
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        qvec = self._vectorize(tokenize(query))
        scored = [
            RetrievedChunk(chunk=c, score=self._cosine(qvec, v))
            for c, v in zip(self.chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
