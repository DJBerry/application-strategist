# application-strategist

Job search assistance CLI — compare your resume and cover letter against job descriptions and get actionable feedback from two perspectives: hiring team fit and candidate fit.

## Features

- **Employer-side evaluation**: Strengths, gaps, suggested improvements, wording suggestions, fit score (0–100)
- **Candidate-side evaluation**: Positive alignments, red flags, questions to ask, worker fit score (0–100)
- **Follow-up REPL**: Ask follow-up questions about the evaluation with full context preserved

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Clone and install:

   ```bash
   uv sync
   ```

2. Create a `.env` file with your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

   Get your key at [console.anthropic.com](https://console.anthropic.com/).

## Usage

```bash
# Basic analysis (resume + job description)
uv run app-strategist --resume path/to/resume.md --job path/to/job.txt

# With cover letter
uv run app-strategist --resume resume.md --job job.txt --cover-letter cover.md

# Verbose logging
uv run app-strategist --resume resume.md --job job.txt -v
```

Supported file formats: `.txt` and `.md` for resume, cover letter, and job description.

After the evaluation is displayed, you can ask follow-up questions such as:

- "Why was the employer fit score low?"
- "What should I change in my resume summary?"
- "Give me a version of bullet 3 tailored to this role."

Type `quit` or `exit` to leave the REPL.

## Development

```bash
# Run tests
uv run pytest tests/ -v
```

## Project Structure

```
src/app_strategist/
├── llm/           # LLM provider abstraction (Anthropic in v1)
├── parsers/       # File parsing (extensible for PDF, DOCX, URLs)
├── models/        # Pydantic evaluation models
├── services/      # Analysis orchestration, scoring
└── rendering/     # Rich console output
```
