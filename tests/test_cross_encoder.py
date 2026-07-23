"""Cross-encoder re-ranker tests.

The neural test is skipped unless `sentence-transformers` is installed
(`pip install -r requirements-prod.txt`), so the default suite stays fast and
dependency-free. The factory test always runs.
"""
import pytest

from app.chunking import chunk_document
from app.rerank import LexicalReRanker, make_reranker
from app.retrieval import RetrievedChunk


def test_make_reranker_selects_correct_type():
    assert isinstance(make_reranker("lexical"), LexicalReRanker)
    assert make_reranker("none") is None
    assert make_reranker(None) is None
    with pytest.raises(ValueError):
        make_reranker("bogus")


def test_cross_encoder_reranks_when_available():
    pytest.importorskip("sentence_transformers")
    from app.rerank import CrossEncoderReRanker

    chunks = chunk_document(
        "kb.md",
        "The Premium plan costs 9.99 USD per month.\n\n"
        "Two-factor authentication is required on every account.",
        chunk_size=60,
        chunk_overlap=10,
    )
    cands = [RetrievedChunk(chunk=c, score=0.0) for c in chunks]
    out = CrossEncoderReRanker().rerank("How much is the premium plan?", cands, top_k=1)
    assert "9.99" in out[0].chunk.text
