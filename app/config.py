"""Runtime configuration, loaded from environment / .env.

Prompt versioning is first-class here: switching PROMPT_VERSION swaps the prompt
template the whole service uses, so prompt changes are auditable and reversible.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DATA_DIR = ROOT / "data" / "sample_docs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Retrieval
    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 80

    # Retrieval strategy: "tfidf" (semantic only), "bm25" (lexical only),
    # or "hybrid" (both, fused with Reciprocal Rank Fusion).
    retrieval_mode: str = "hybrid"
    use_rerank: bool = True   # second-stage re-ranker on the fused shortlist
    rrf_k: int = 60           # RRF smoothing constant
    candidate_k: int = 10     # shortlist size passed to the re-ranker

    # Re-ranker: "lexical" (default, dependency-free) or "cross_encoder"
    # (neural, needs requirements-prod.txt).
    reranker: str = "lexical"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Grounding: refuse to answer when the best chunk scores below this.
    # Tuned from the score distribution: in-scope questions score ~0.28-0.48,
    # out-of-scope ones ~0.15, so 0.20 separates them cleanly.
    grounding_threshold: float = 0.20

    # Prompt versioning (semantic): "qa_v1", "qa_v2", ...
    prompt_version: str = "qa_v2"

    # LLM (only used when OPENAI_API_KEY is present; otherwise offline fallback).
    llm_model: str = "gpt-4o-mini"

    def load_prompt(self) -> str:
        path = PROMPTS_DIR / f"{self.prompt_version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt version not found: {path}")
        return path.read_text(encoding="utf-8")


settings = Settings()
