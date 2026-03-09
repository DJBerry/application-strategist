# application-strategist

Job search assistance CLI — compare your resume and cover letter against job descriptions and get actionable feedback from two perspectives: hiring team fit and candidate fit.

## Features

- **Employer-side evaluation**: Strengths, gaps, suggested improvements, wording suggestions, fit score (0–100)
- **Candidate-side evaluation**: Positive alignments, red flags, questions to ask, worker fit score (0–100)
- **Follow-up REPL**: Ask follow-up questions about the evaluation with full context preserved
- **Extraction-validation loop** (optional): Run a multi-agent extraction → validation → retry loop before evaluation to ensure claims are grounded in documents; view audit trace with `--show-trace`
- **MCP server**: Expose resume validation tools for Cursor and other MCP clients via `resume-validator`

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Clone and install:

   ```bash
   uv sync
   ```

2. Create a `.env` file with your API key(s). Set `LLM_PROVIDER` to choose the backend:

   **Anthropic (default):**
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=your-api-key-here
   ```
   Get your key at [console.anthropic.com](https://console.anthropic.com/).

   **OpenAI:**
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your-api-key-here
   ```
   Get your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

## Usage

```bash
# Basic analysis (resume + job description)
uv run app-strategist --resume path/to/resume.md --job path/to/job.txt

# With cover letter
uv run app-strategist --resume resume.md --job job.txt --cover-letter cover.md

# Run extraction-validation loop before evaluation (ensures claims are document-grounded)
uv run app-strategist --resume resume.md --job job.txt --validate

# Show validation audit trace when using --validate
uv run app-strategist --resume resume.md --job job.txt --validate --show-trace

# Verbose logging
uv run app-strategist --resume resume.md --job job.txt -v
```

Supported file formats: `.txt` and `.md` for resume, cover letter, and job description.

After the evaluation is displayed, you can ask follow-up questions such as:

- "Why was the employer fit score low?"
- "What should I change in my resume summary?"
- "Give me a version of bullet 3 tailored to this role."

Type `quit` or `exit` to leave the REPL.

## MCP Server

The project includes an MCP server (`resume-validator`) that exposes tools for validating extracted claims against resume and job description documents. Use it from Cursor or other MCP clients:

```bash
uv run resume-validator
```

The server provides tools such as `check_claim_in_document`, `validate_extraction_batch`, and `log_validation_event` for semantic claim validation.

## Development

```bash
# Run tests
uv run pytest tests/ -v
```

## Project Structure

```
src/app_strategist/
├── llm/           # LLM provider abstraction (Anthropic, OpenAI)
├── parsers/       # File parsing (extensible for PDF, DOCX, URLs)
├── models/        # Pydantic evaluation and session models
├── services/      # Analysis orchestration, scoring, extractor, validator, orchestrator
├── mcp_server/    # MCP server exposing resume validation tools
└── rendering/     # Rich console output, audit trail
```
