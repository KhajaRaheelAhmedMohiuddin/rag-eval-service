"""FastAPI service exposing the RAG pipeline.

Endpoints:
  GET  /health   -> liveness + index stats
  POST /ask      -> grounded, guardrailed answer with sources + HITL flag
"""
from __future__ import annotations

from fastapi import FastAPI

from app.rag import RagService
from app.schemas import AskRequest, AskResponse, Source

app = FastAPI(title="RAG Eval Service", version="1.0.0")

# Build the index once at startup (documents -> chunks -> vector store).
_service: RagService | None = None


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


@app.get("/health")
def health() -> dict:
    svc = get_service()
    return {
        "status": "ok",
        "chunks_indexed": len(svc.retriever.chunks),
        "prompt_version": svc.cfg.prompt_version,
        "llm": type(svc.llm).__name__,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    svc = get_service()
    result = svc.answer(req.question, top_k=req.top_k)
    return AskResponse(
        answer=result.answer,
        grounded=result.grounded,
        prompt_version=result.prompt_version,
        needs_human_review=result.needs_human_review,
        sources=[
            Source(
                id=r.chunk.id,
                source=r.chunk.source,
                score=round(r.score, 4),
                preview=r.chunk.text[:160],
            )
            for r in result.sources
        ],
    )
