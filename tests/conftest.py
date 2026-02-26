"""Pytest fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_resume(tmp_path: Path) -> Path:
    """Create a sample resume file."""
    content = """# John Doe
Software Engineer with 5 years of Python experience.

## Experience
- Built REST APIs at Acme Corp
- Led team of 3 developers
"""
    f = tmp_path / "resume.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_job(tmp_path: Path) -> Path:
    """Create a sample job description file."""
    content = """Senior Python Developer

Requirements:
- 5+ years Python
- REST API design
- Team leadership

Nice to have: AWS, Kubernetes
"""
    f = tmp_path / "job.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def sample_cover_letter(tmp_path: Path) -> Path:
    """Create a sample cover letter file."""
    content = """Dear Hiring Manager,

I am excited to apply for the Senior Python Developer role...
"""
    f = tmp_path / "cover.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def mock_llm_response():
    """Sample LLM JSON response for employer evaluation."""
    return """
```json
{
  "strengths": ["5 years Python experience", "REST API experience"],
  "gaps": ["AWS - missing"],
  "suggested_improvements": ["Add metrics to achievements"],
  "wording_suggestions": [
    {
      "current": "Built REST APIs",
      "suggested": "Designed and built REST APIs serving 10K+ requests/day",
      "rationale": "Add quantification",
      "status": "present_underemphasized"
    }
  ],
  "fit_score": {
    "value": 75,
    "components": [
      {"name": "Skills/experience alignment", "weight": 0.35, "score": 80, "explanation": "Strong match"},
      {"name": "Keyword/match density", "weight": 0.25, "score": 70, "explanation": "Good"},
      {"name": "Clarity and structure", "weight": 0.20, "score": 75, "explanation": "Clear"},
      {"name": "Quantified achievements", "weight": 0.15, "score": 60, "explanation": "Could add more"},
      {"name": "Tailoring to role", "weight": 0.05, "score": 80, "explanation": "Relevant"}
    ]
  },
  "score_rationale": "Strong candidate with minor gaps."
}
```
"""
