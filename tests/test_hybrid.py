"""Tests for hybrid retrieval, RRF, and re-ranking."""
from app.bm25 import BM25Retriever
from app.chunking import chunk_document
from app.fusion import reciprocal_rank_fusion
from app.hybrid import HybridRetriever
from app.rerank import LexicalReRanker
from app.retrieval import RetrievedChunk


DOC = (
    "The Premium plan costs 9.99 USD per month.\n\n"
    "Domestic transfers between users are free and instant.\n\n"
    "Instant bank withdrawal has a 1.5 percent fee capped at 10 USD.\n\n"
    "Two-factor authentication is required on every account."
)


def _chunks():
    return chunk_document("kb.md", DOC, chunk_size=80, chunk_overlap=10)


def test_rrf_rewards_agreement_across_lists():
    # id "b" is ranked highly by both lists; "a" and "c" only by one.
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert fused["b"] == max(fused.values())


def test_bm25_ranks_keyword_match_first():
    r = BM25Retriever(_chunks())
    top = r.retrieve("instant bank withdrawal fee", k=1)[0]
    assert "1.5" in top.chunk.text


def test_reranker_promotes_phrase_match():
    chunks = _chunks()
    cands = [RetrievedChunk(chunk=c, score=0.0) for c in chunks]
    ranked = LexicalReRanker().rerank("two-factor authentication required", cands, top_k=1)
    assert "authentication" in ranked[0].chunk.text


def test_hybrid_returns_calibrated_cosine_score():
    r = HybridRetriever(_chunks(), use_rerank=True)
    results = r.retrieve("How much is the Premium plan?", k=2)
    assert results, "hybrid returned no results"
    # Score is the semantic cosine in [0, 1], not the RRF/rerank score.
    assert all(0.0 <= rc.score <= 1.0 for rc in results)
    assert "9.99" in results[0].chunk.text


def test_hybrid_retrieves_relevant_chunk_top1():
    r = HybridRetriever(_chunks(), use_rerank=True)
    top = r.retrieve("What is the instant withdrawal fee?", k=1)[0]
    assert "1.5" in top.chunk.text
