# application-strategist

Job search assistance CLI — compare your resume and cover letter against job descriptions and get actionable feedback from two perspectives: hiring team fit and candidate fit.

## Features

- **Employer-side evaluation**: Strengths, gaps, suggested improvements, wording suggestions, fit score (0–100)
- **Candidate-side evaluation**: Positive alignments, red flags, questions to ask, worker fit score (0–100)
- **Follow-up REPL**: Ask follow-up questions about the evaluation with full context preserved
- **Job description extraction** _(LangGraph)_: Structured extraction of company and role metadata, with LLM-based validation and automatic retry on errors

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

### Full evaluation (`analyze`)

Runs both the employer-side and candidate-side evaluations, then opens a follow-up REPL.

```bash
# Resume + job description
uv run app-strategist analyze --resume path/to/resume.md --job path/to/job.txt

# With cover letter
uv run app-strategist analyze --resume resume.md --job job.txt --cover-letter cover.md

# Verbose logging
uv run app-strategist analyze --resume resume.md --job job.txt -v
```

Supported file formats: `.txt` and `.md` for all inputs.

After the evaluation is displayed, you can ask follow-up questions such as:

- "Why was the employer fit score low?"
- "What should I change in my resume summary?"
- "Give me a version of bullet 3 tailored to this role."

Type `quit` or `exit` to leave the REPL.

### Job description extraction (`extract`)

Extracts structured company and role metadata from a job description using a LangGraph pipeline. The pipeline runs an extraction step, validates the result with a second LLM call, and retries any incorrect fields automatically (up to 3 total attempts).

```bash
# Extract from a job description
uv run app-strategist extract --resume path/to/resume.md --job path/to/job.txt

# With cover letter (carried through for future pipeline steps)
uv run app-strategist extract --resume resume.md --job job.txt --cover-letter cover.md

# Verbose logging (shows graph node execution and attempt count)
uv run app-strategist extract --resume resume.md --job job.txt -v
```

Output is a JSON object with two sections:

```json
{
  "company_info": {
    "company_name": "Acme Corp",
    "company_description": "A widget manufacturer",
    "company_mission": "Build tools that delight developers"
  },
  "job_info": {
    "title": "Senior Software Engineer",
    "seniority_level": "Senior",
    "location": "Austin, TX",
    "work_environment": "hybrid"
  }
}
```

Fields that cannot be found in the job description are set to `"N/A"`. If validation still fails after 3 attempts, the best-effort data is shown along with a warning panel listing the unresolved concerns.

## Development

```bash
# Run all tests
uv run pytest tests/ -v

# Run only the graph tests
uv run pytest tests/test_graph/ -v
```

## Project Structure

```
src/app_strategist/
├── graph/         # LangGraph extraction pipeline (state, nodes, prompts, graph)
├── llm/           # LLM provider abstraction (Anthropic, OpenAI)
├── parsers/       # File parsing (extensible for PDF, DOCX, URLs)
├── models/        # Pydantic evaluation models
├── services/      # Analysis orchestration, scoring
└── rendering/     # Rich console output
```
