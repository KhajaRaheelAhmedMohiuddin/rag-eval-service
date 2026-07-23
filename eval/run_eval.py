"""Run the evaluation harness over the golden set and write a Markdown report.

Usage:
    python -m eval.run_eval

Because the offline LLM is deterministic, this produces a reproducible score you
can track across prompt versions and retrieval changes — the core discipline of
building reliable LLM systems (regression testing for AI).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.rag import RagService
from eval.scorers import aggregate, answer_correct, refusal_correct, retrieval_hit

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden.json"
REPORT = EVAL_DIR / "report.md"


def run() -> dict:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    service = RagService()

    rows = []
    for item in golden["items"]:
        result = service.answer(item["question"])
        retrieved_sources = [s.chunk.source for s in result.sources]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "grounded": result.grounded,
                "needs_human_review": result.needs_human_review,
                "retrieval_hit": retrieval_hit(item["expected_source"], retrieved_sources),
                "answer_correct": answer_correct(result.answer, item["answer_keywords"]),
                "refusal_correct": refusal_correct(item["should_refuse"], result.grounded),
            }
        )

    summary = aggregate(rows)
    write_report(rows, summary, service.cfg.prompt_version)
    return summary


def write_report(rows: list[dict], summary: dict, prompt_version: str) -> None:
    lines = [
        "# Evaluation Report",
        "",
        f"- Prompt version: `{prompt_version}`",
        f"- Items: **{summary['n']}**",
        f"- Retrieval hit-rate: **{summary['retrieval_hit_rate']:.0%}**",
        f"- Answer accuracy: **{summary['answer_accuracy']:.0%}**",
        f"- Refusal accuracy: **{summary['refusal_accuracy']:.0%}**",
        "",
        "| ID | Question | Retrieved? | Answer OK? | Refusal OK? | HITL |",
        "|----|----------|:---------:|:----------:|:-----------:|:----:|",
    ]
    for r in rows:
        def mark(v):
            return "n/a" if v is None else ("✅" if v else "❌")

        lines.append(
            f"| {r['id']} | {r['question']} | {mark(r['retrieval_hit'])} | "
            f"{mark(r['answer_correct'])} | {mark(r['refusal_correct'])} | "
            f"{'🔶' if r['needs_human_review'] else ''} |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    s = run()
    print(json.dumps(s, indent=2))
    print(f"\nReport written to {REPORT}")
