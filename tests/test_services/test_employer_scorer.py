"""Tests for EmployerScorer - isolated unit tests with mock LLM."""

from unittest.mock import MagicMock

import pytest

from app_strategist.services.employer_scorer import EmployerScorer
from app_strategist.utils import JSONExtractionError


EMPLOYER_JSON_FULL = """
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
    },
    {
      "current": "Missing Kubernetes",
      "suggested": "Add if you have k8s experience",
      "rationale": "Nice to have",
      "status": "missing_not_evidenced"
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


class MockLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, messages: list) -> str:
        return self._response


def test_employer_scorer_produces_evaluation() -> None:
    mock_llm = MockLLM(EMPLOYER_JSON_FULL)
    scorer = EmployerScorer(llm=mock_llm)

    eval_ = scorer.evaluate(
        resume="5 years Python, REST APIs",
        job_description="Senior Python Developer",
        cover_letter="Excited to apply...",
    )

    assert eval_.fit_score.value == 73.5
    assert len(eval_.fit_score.components) == 5
    assert eval_.strengths == ["Python experience", "REST APIs"]
    assert eval_.gaps == ["AWS - missing"]
    assert len(eval_.wording_suggestions) == 2
    assert eval_.wording_suggestions[0].status == "present_underemphasized"
    assert eval_.wording_suggestions[1].status == "missing_not_evidenced"
    assert eval_.score_rationale == "Strong candidate."


def test_employer_scorer_without_cover_letter() -> None:
    mock_llm = MockLLM(EMPLOYER_JSON_FULL)
    scorer = EmployerScorer(llm=mock_llm)

    eval_ = scorer.evaluate(
        resume="Resume content",
        job_description="Job desc",
        cover_letter=None,
    )

    assert eval_.fit_score.value >= 0
    assert eval_.fit_score.value <= 100


def test_employer_scorer_enforces_rubric_weights() -> None:
    """Our rubric weights override LLM-returned weights."""
    mock_llm = MockLLM(EMPLOYER_JSON_FULL)
    scorer = EmployerScorer(llm=mock_llm)
    eval_ = scorer.evaluate("r", "j", None)

    for comp in eval_.fit_score.components:
        assert comp.weight in (0.35, 0.25, 0.20, 0.15, 0.05)


def test_employer_scorer_fallback_for_missing_component() -> None:
    """When LLM omits a rubric component, we use score 50 and 'Component not provided by model'."""
    json_missing_tailoring = """
    {
      "strengths": [],
      "gaps": [],
      "suggested_improvements": [],
      "wording_suggestions": [],
      "fit_score": {
        "value": 70,
        "components": [
          {"name": "Skills/experience alignment", "weight": 0.35, "score": 80, "explanation": "x"},
          {"name": "Keyword/match density", "weight": 0.25, "score": 70, "explanation": "x"},
          {"name": "Clarity and structure", "weight": 0.20, "score": 75, "explanation": "x"},
          {"name": "Quantified achievements", "weight": 0.15, "score": 60, "explanation": "x"}
        ]
      },
      "score_rationale": "OK"
    }
    """
    mock_llm = MockLLM(json_missing_tailoring)
    scorer = EmployerScorer(llm=mock_llm)
    eval_ = scorer.evaluate("r", "j", None)

    tailoring = next(c for c in eval_.fit_score.components if c.name == "Tailoring to role")
    assert tailoring.score == 50.0
    assert tailoring.explanation == "Component not provided by model"


def test_employer_scorer_raises_on_invalid_json() -> None:
    mock_llm = MockLLM("This is not JSON at all")
    scorer = EmployerScorer(llm=mock_llm)

    with pytest.raises(JSONExtractionError):
        scorer.evaluate("r", "j", None)
