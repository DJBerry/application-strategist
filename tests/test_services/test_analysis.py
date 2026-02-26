"""Tests for AnalysisService."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app_strategist.models.evaluation import CandidateEvaluation, EmployerEvaluation
from app_strategist.models.scoring import FitScore, ScoreComponent
from app_strategist.services.analysis import AnalysisService


class MockLLMProvider:
    """Mock LLM that returns predefined JSON for employer and candidate evals."""

    def __init__(self, employer_json: str, candidate_json: str) -> None:
        self._responses = [employer_json, candidate_json]
        self._call_count = 0

    def complete(self, system_prompt: str, messages: list) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


EMPLOYER_JSON = """
{
  "strengths": ["Python experience", "REST APIs"],
  "gaps": ["AWS - missing"],
  "suggested_improvements": ["Add metrics"],
  "wording_suggestions": [
    {
      "current": "Built APIs",
      "suggested": "Built APIs serving 10K req/day",
      "rationale": "Quantify",
      "status": "present_underemphasized"
    }
  ],
  "fit_score": {
    "value": 75,
    "components": [
      {"name": "Skills/experience alignment", "weight": 0.35, "score": 80, "explanation": "Strong"},
      {"name": "Keyword/match density", "weight": 0.25, "score": 70, "explanation": "Good"},
      {"name": "Clarity and structure", "weight": 0.20, "score": 75, "explanation": "Clear"},
      {"name": "Quantified achievements", "weight": 0.15, "score": 60, "explanation": "Add more"},
      {"name": "Tailoring to role", "weight": 0.05, "score": 80, "explanation": "Relevant"}
    ]
  },
  "score_rationale": "Strong candidate."
}
"""

CANDIDATE_JSON = """
{
  "positive_alignments": ["Growth opportunity"],
  "concerns": ["Vague compensation"],
  "questions_to_ask": ["What is the salary range?"],
  "worker_fit_score": {
    "value": 70,
    "components": [
      {"name": "Role clarity", "weight": 0.25, "score": 80, "explanation": "Clear"},
      {"name": "Compensation/benefits transparency", "weight": 0.20, "score": 50, "explanation": "Vague"},
      {"name": "Work-life balance signals", "weight": 0.15, "score": 70, "explanation": "OK"},
      {"name": "Growth/career path", "weight": 0.20, "score": 75, "explanation": "Good"},
      {"name": "Red-flag absence", "weight": 0.20, "score": 75, "explanation": "Few"}
    ]
  },
  "score_rationale": "Reasonable fit."
}
"""


def test_analysis_service_produces_session(
    sample_resume: Path, sample_job: Path, sample_cover_letter: Path
) -> None:
    mock_llm = MockLLMProvider(EMPLOYER_JSON, CANDIDATE_JSON)
    service = AnalysisService(llm=mock_llm)

    session = service.analyze(
        resume_path=sample_resume,
        job_path=sample_job,
        cover_letter_path=sample_cover_letter,
    )

    assert session.resume_content
    assert session.job_description
    assert session.cover_letter_content
    assert session.employer_eval.fit_score.value >= 0
    assert session.employer_eval.fit_score.value <= 100
    assert session.candidate_eval.worker_fit_score.value >= 0
    assert session.candidate_eval.worker_fit_score.value <= 100
    assert len(session.employer_eval.strengths) > 0
    assert len(session.candidate_eval.questions_to_ask) > 0


def test_analysis_service_without_cover_letter(sample_resume: Path, sample_job: Path) -> None:
    mock_llm = MockLLMProvider(EMPLOYER_JSON, CANDIDATE_JSON)
    service = AnalysisService(llm=mock_llm)

    session = service.analyze(
        resume_path=sample_resume,
        job_path=sample_job,
        cover_letter_path=None,
    )

    assert session.cover_letter_content is None
    assert session.employer_eval is not None
