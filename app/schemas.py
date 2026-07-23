"""Pydantic request/response schemas — the API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    id: str
    source: str
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    prompt_version: str
    sources: list[Source]
    # HITL: high-stakes / low-confidence answers are flagged for human sign-off.
    needs_human_review: bool
