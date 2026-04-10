# application-strategist

A Python CLI tool that evaluates a resume (and optional cover letter) against a job description from two perspectives: the hiring manager's view (`EmployerEvaluation`) and the candidate's view (`CandidateEvaluation`). After the initial evaluation it drops into an interactive REPL for follow-up questions. Run via `uv`.

## Commands

```bash
uv run app-strategist --resume <path> --job <path> [--cover-letter <path>] [--verbose]
uv run pytest
```

Sample documents for manual testing are in `example_docs/`.

## Architecture

All source code is under `src/app_strategist/`. All tests are under `src/tests/`.

| Module | Role |
|--------|------|
| `services/employer_scorer.py` | Employer-side LLM evaluation (single-call, fixed rubric) |
| `services/candidate_scorer.py` | Candidate-side LLM evaluation |
| `services/analysis.py` | Orchestrates parsing + both scorers → `AnalysisSession` |
| `models/evaluation.py` | `EmployerEvaluation`, `CandidateEvaluation`, `WordingSuggestion` |
| `models/scoring.py` | `FitScore`, `ScoreComponent` |
| `models/session.py` | `AnalysisSession` — full context for REPL |
| `llm/base.py` | `LLMProvider` protocol (`complete(system_prompt, messages) → str`) |
| `llm/anthropic_provider.py` | Claude (Sonnet 4.6) |
| `llm/openai_provider.py` | OpenAI |
| `config.py` | Reads `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` from `.env` |
| `utils.py` | `extract_json()` — strips markdown fences, parses LLM JSON responses |
| `rendering/console.py` | Rich-based console output |
| `parsers/` | Text/Markdown document parsers; registry pattern for extension |

## Conventions

**Fixed rubric weights** — `EMPLOYER_RUBRIC` and `CANDIDATE_RUBRIC` define scoring weights. These are enforced in Python after each LLM response; the LLM's returned weights are discarded. All score aggregation (weighted sums, totals) is computed in Python. Never delegate math or weight application to the LLM.

**No fabrication** — All prompts enforce that the LLM only uses evidence explicitly present in the documents. This constraint is non-negotiable; preserve it in any prompt changes.

**JSON extraction** — LLM responses are always parsed through `extract_json()` in `utils.py`, which handles markdown-wrapped JSON.

**Pydantic models for all outputs** — Evaluation outputs are Pydantic models. Add fields to `models/evaluation.py` when expanding outputs.

**LangGraph** — `langgraph` is a required dependency for graph-based evaluation workflows. When adding graph-based scorers, use `StateGraph` with a `TypedDict` state schema. Always use the existing `LLMProvider` protocol inside graph nodes — do not use LangChain LLM wrappers.

**Future UI support** — A UI is a planned future addition. Keep business logic cleanly separated from CLI rendering; avoid hardcoding stdout assumptions in service/model layers; prefer returning structured data objects over side-effectful output. Do not implement UI features now.

## Environment

`.env` should already be present. Default provider is Anthropic (`LLM_PROVIDER=anthropic`).

**Never modify `.env` without explicit user permission.** Always ask before touching it.
