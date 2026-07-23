"""The RAG pipeline: ingest -> retrieve -> prompt -> generate -> guardrails.

This wires the components together behind a single `RagService.answer()` call.
The same interface is used by the FastAPI app and the evaluation harness, so
what you evaluate is exactly what you serve.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.chunking import Chunk, chunk_document
from app.config import DATA_DIR, Settings, settings
from app.guardrails import REFUSAL_MESSAGE, is_grounded, redact_pii
from app.llm import LLM, get_llm
from app.retrieval import RetrievedChunk, TfidfRetriever

# Intents that should never be auto-executed without a human — the HITL gate.
_HIGH_STAKES_RE = re.compile(
    r"\b(refund|chargeback|transfer|delete|close account|cancel|reset password)\b",
    re.IGNORECASE,
)


@dataclass
class Answer:
    answer: str
    grounded: bool
    prompt_version: str
    sources: list[RetrievedChunk]
    needs_human_review: bool


def load_documents(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            chunks.extend(
                chunk_document(
                    path.name,
                    path.read_text(encoding="utf-8"),
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
            )
    return chunks


class RagService:
    def __init__(self, cfg: Settings = settings, llm: LLM | None = None, data_dir: Path | None = None):
        self.cfg = cfg
        self.llm = llm or get_llm(cfg.llm_model)
        chunks = load_documents(data_dir or DATA_DIR)
        if not chunks:
            raise RuntimeError("No documents found to index.")
        self.retriever = TfidfRetriever(chunks)
        self.prompt_template = cfg.load_prompt()

    def _format_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        context = "\n\n".join(f"[{r.chunk.id}] {r.chunk.text}" for r in retrieved)
        return self.prompt_template.format(context=context, question=question)

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        k = top_k or self.cfg.top_k
        retrieved = self.retriever.retrieve(question, k=k)
        top_score = retrieved[0].score if retrieved else 0.0
        grounded = is_grounded(top_score, self.cfg.grounding_threshold)

        if not grounded:
            # Refuse rather than hallucinate; route to a human.
            return Answer(
                answer=REFUSAL_MESSAGE,
                grounded=False,
                prompt_version=self.cfg.prompt_version,
                sources=[],
                needs_human_review=True,
            )

        prompt = self._format_prompt(question, retrieved)
        raw = self.llm.generate(prompt)
        safe = redact_pii(raw)

        # HITL: flag low-confidence answers and high-stakes intents for sign-off.
        low_confidence = top_score < (self.cfg.grounding_threshold * 1.5)
        high_stakes = bool(_HIGH_STAKES_RE.search(question))
        needs_review = low_confidence or high_stakes

        return Answer(
            answer=safe,
            grounded=True,
            prompt_version=self.cfg.prompt_version,
            sources=retrieved,
            needs_human_review=needs_review,
        )
