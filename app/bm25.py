"""BM25 lexical retriever (Okapi BM25), dependency-free.

BM25 is the standard sparse/keyword retrieval function. It complements the
TF-IDF semantic retriever: BM25 rewards exact term matches with term-frequency
saturation and document-length normalisation, catching relevant passages that
cosine similarity can rank lower. Fusing the two (see app/hybrid.py) is a
hybrid-retrieval strategy.
"""
from __future__ import annotations

import math
from collections import Counter

from app.chunking import Chunk
from app.retrieval import RetrievedChunk, tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(c.text) for c in chunks]
        self._doc_len = [len(d) for d in self._docs]
        self._avgdl = (sum(self._doc_len) / len(self._docs)) if self._docs else 0.0
        self._tf = [Counter(d) for d in self._docs]
        self._idf = self._compute_idf(self._docs)

    @staticmethod
    def _compute_idf(docs: list[list[str]]) -> dict[str, float]:
        n = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        # BM25 idf with the standard +0.5 smoothing.
        return {t: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()}

    def _score(self, q_tokens: list[str], i: int) -> float:
        tf = self._tf[i]
        dl = self._doc_len[i]
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = self._idf.get(t, 0.0)
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score += idf * num / den
        return score

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        q = tokenize(query)
        scored = [
            RetrievedChunk(chunk=c, score=self._score(q, i))
            for i, c in enumerate(self.chunks)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
