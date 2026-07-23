"""Unit tests for the framework-light core (no API key, no heavy deps)."""
from app.chunking import chunk_text, chunk_document
from app.guardrails import redact_pii, is_grounded
from app.retrieval import TfidfRetriever
from app.llm import ExtractiveLLM


def test_chunking_overlap_and_size():
    text = "Sentence one. " * 100
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + overlap slack


def test_chunk_document_ids_are_traceable():
    chunks = chunk_document("doc.md", "a. b. c. " * 50, chunk_size=100, chunk_overlap=10)
    assert all(c.source == "doc.md" for c in chunks)
    assert chunks[0].id == "doc.md#0"


def test_redact_pii():
    dirty = "Email me at john.doe@example.com or call +1 415 555 0199."
    clean = redact_pii(dirty)
    assert "john.doe@example.com" not in clean
    assert "[REDACTED_EMAIL]" in clean
    assert "555" not in clean


def test_grounding_threshold():
    assert is_grounded(0.5, 0.05) is True
    assert is_grounded(0.01, 0.05) is False


def test_retriever_ranks_relevant_chunk_first():
    chunks = chunk_document(
        "fees.md",
        "The Premium plan costs 9.99 USD per month. Domestic transfers are free.",
        chunk_size=60,
        chunk_overlap=10,
    )
    r = TfidfRetriever(chunks)
    top = r.retrieve("How much is the Premium plan?", k=1)[0]
    assert "9.99" in top.chunk.text


def test_extractive_llm_is_grounded():
    prompt = "CONTEXT: The Premium plan costs 9.99 USD per month. QUESTION: How much is Premium? ANSWER:"
    out = ExtractiveLLM().generate(prompt)
    assert "9.99" in out
