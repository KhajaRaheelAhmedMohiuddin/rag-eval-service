"""Hybrid retrieval: semantic (TF-IDF) + lexical (BM25), fused with RRF, then
optionally re-ranked.

Pipeline:
    query
      ├─► TF-IDF cosine  ──► ranking A
      └─► BM25           ──► ranking B
                 │
     Reciprocal Rank Fusion (A, B)  ──► candidate shortlist (candidate_k)
                 │
         optional re-ranker         ──► final top-k

The reported `.score` on each result is the calibrated **semantic cosine**
(0..1), not the fusion/rerank score — so grounding thresholds and confidence
displays stay meaningful regardless of how ranking was produced. Fusion and
re-ranking only decide the *order*.
"""
from __future__ import annotations

from app.chunking import Chunk
from app.retrieval import RetrievedChunk, TfidfRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        rrf_k: int = 60,
        candidate_k: int = 10,
        use_rerank: bool = True,
    ):
        from app.bm25 import BM25Retriever
        from app.fusion import reciprocal_rank_fusion
        from app.rerank import LexicalReRanker

        self.chunks = chunks
        self._by_id = {c.id: c for c in chunks}
        self.dense = TfidfRetriever(chunks)
        self.sparse = BM25Retriever(chunks)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._rrf = reciprocal_rank_fusion
        self.reranker = LexicalReRanker() if use_rerank else None

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        n = len(self.chunks)
        dense = self.dense.retrieve(query, k=n)
        sparse = self.sparse.retrieve(query, k=n)

        dense_cos = {rc.chunk.id: rc.score for rc in dense}
        dense_rank = [rc.chunk.id for rc in dense]
        sparse_rank = [rc.chunk.id for rc in sparse]

        fused = self._rrf([dense_rank, sparse_rank], k=self.rrf_k)
        candidate_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[: self.candidate_k]

        # Calibrated score = semantic cosine (used for grounding & display).
        candidates = [
            RetrievedChunk(chunk=self._by_id[cid], score=dense_cos.get(cid, 0.0))
            for cid in candidate_ids
        ]

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=k)
        return candidates[:k]
