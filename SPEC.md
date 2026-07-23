# SPEC — RAG Eval Service

Built spec-first: the specification below was written before the implementation,
then the code was built to satisfy it. (Spec-driven / agentic development.)

## Goal
A question-answering microservice over a document knowledge base that is
**grounded, safe, and measurable**.

## Functional requirements
1. **Ingest** `.md`/`.txt` documents, split into overlapping chunks, index them.
2. **/ask**: given a question, retrieve top-k chunks and answer from them only.
3. **Ground**: if the best chunk's similarity is below `GROUNDING_THRESHOLD`,
   refuse (`grounded=false`) instead of answering.
4. **Prompt versioning**: the active prompt is selected by `PROMPT_VERSION`;
   changing it must change behaviour without code edits.
5. **Guardrails**: redact emails/phones/card numbers from any answer.
6. **HITL**: flag `needs_human_review=true` for (a) low-confidence answers and
   (b) high-stakes intents (refund, transfer, delete, reset password, ...).
7. **/health**: report index size, active prompt version, and LLM in use.

## Non-functional requirements
- Runs with **no API key** and **no heavy dependencies** (deterministic offline mode).
- Deterministic evaluation → reproducible scores for regression testing.
- Same code path serves the API and the eval harness.

## Evaluation contract
Golden set items declare: `question`, `expected_source`, `answer_keywords`,
`should_refuse`. Metrics: retrieval hit-rate, answer accuracy, refusal accuracy.
Acceptance bar (offline): retrieval ≥ 0.85, answer ≥ 0.70, refusal ≥ 0.85.

## Out of scope (v1)
Multi-turn memory, auth/tenancy, streaming responses, re-ranking. These are
natural next steps (see ARCHITECTURE.md → Roadmap).
