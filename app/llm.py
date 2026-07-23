"""LLM abstraction.

get_llm() returns a callable `generate(prompt: str) -> str`.

* If OPENAI_API_KEY is set, it uses OpenAI (via langchain-openai if installed,
  else the openai SDK).
* Otherwise it falls back to a deterministic extractive model so the whole
  pipeline — including the evaluation harness and tests — runs offline with no
  API key. The fallback returns the context sentence(s) most relevant to the
  question, which keeps offline answers grounded and eval scores meaningful.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Protocol

from app.retrieval import tokenize


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class ExtractiveLLM:
    """Deterministic, offline stand-in for a chat model.

    It parses the CONTEXT block out of the prompt and returns the sentences most
    lexically relevant to the QUESTION. Not a real generator — but grounded and
    reproducible, which is exactly what you want for deterministic evals/tests.
    """

    _CONTEXT_RE = re.compile(r"CONTEXT:\s*(.*?)\s*QUESTION:", re.DOTALL)
    _QUESTION_RE = re.compile(r"QUESTION:\s*(.*?)\s*(?:ANSWER:|$)", re.DOTALL)

    def generate(self, prompt: str) -> str:
        ctx = self._CONTEXT_RE.search(prompt)
        q = self._QUESTION_RE.search(prompt)
        context = ctx.group(1) if ctx else prompt
        question = q.group(1) if q else ""
        qtokens = set(tokenize(question))
        sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
        scored = sorted(
            sentences,
            key=lambda s: len(qtokens & set(tokenize(s))),
            reverse=True,
        )
        best = [s.strip() for s in scored[:2] if s.strip() and (qtokens & set(tokenize(s)))]
        return " ".join(best) if best else "I don't have enough information to answer that."


class OpenAILLM:
    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        from openai import OpenAI  # imported lazily so it's an optional dep

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm(model: str | None = None) -> LLM:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAILLM(model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"), api_key=api_key)
    return ExtractiveLLM()
