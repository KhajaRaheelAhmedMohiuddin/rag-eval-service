"""Safety guardrails: PII redaction and grounding/refusal checks.

These enforce two production-critical behaviours:
  1. Never echo personally identifiable information back to a caller.
  2. Refuse to answer when retrieval found nothing relevant, instead of letting
     the model hallucinate an ungrounded answer.
"""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,12}\d\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")

REFUSAL_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that "
    "reliably."
)


def redact_pii(text: str) -> str:
    """Mask emails, phone numbers, and card-like digit strings."""
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = CARD_RE.sub("[REDACTED_CARD]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def is_grounded(top_score: float, threshold: float) -> bool:
    """True when the best retrieved chunk is relevant enough to answer from."""
    return top_score >= threshold
