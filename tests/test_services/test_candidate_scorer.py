"""Tests for CandidateScorer - isolated unit tests with mock LLM."""

from unittest.mock import MagicMock

import pytest

from app_strategist.services.candidate_scorer import CandidateScorer
from app_strategist.utils import JSONExtractionError


CANDIDATE_JSON_FULL = """
{
  "positive_alignments": ["Growth opportunity", "Remote work"],
  "concerns": ["Vague compensation"],
  "questions_to_ask": ["What is the salary range?", "Benefits?"],
  "worker_fit_score": {
    "value": 70,
    "components": [
      {"name": "Likely interest alignment", "weight": 0.20, "score": 85, "explanation": "Good match"},
      {"name": "Role clarity", "weight": 0.15, "score": 80, "explanation": "Clear"},
      {"name": "Compensation/benefits transparency", "weight": 0.20, "score": 50, "explanation": "Vague"},
      {"name": "Work-life balance signals", "weight": 0.15, "score": 70, "explanation": "OK"},
      {"name": "Growth/career path", "weight": 0.15, "score": 75, "explanation": "Good"},
      {"name": "Red-flag absence", "weight": 0.15, "score": 75, "explanation": "Few"}
    ]
  },
  "score_rationale": "Reasonable fit with pay concerns."
}
"""


class MockLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, messages: list) -> str:
        return self._response


def test_candidate_scorer_produces_evaluation() -> None:
    mock_llm = MockLLM(CANDIDATE_JSON_FULL)
    scorer = CandidateScorer(llm=mock_llm)

    eval_ = scorer.evaluate(
        resume="Senior dev background",
        job_description="Senior role with remote",
        cover_letter="I value growth...",
    )

    assert eval_.worker_fit_score.value >= 0
    assert eval_.worker_fit_score.value <= 100
    assert len(eval_.worker_fit_score.components) == 6
    assert eval_.positive_alignments == ["Growth opportunity", "Remote work"]
    assert eval_.concerns == ["Vague compensation"]
    assert eval_.questions_to_ask == ["What is the salary range?", "Benefits?"]
    assert eval_.score_rationale == "Reasonable fit with pay concerns."


def test_candidate_scorer_without_cover_letter() -> None:
    mock_llm = MockLLM(CANDIDATE_JSON_FULL)
    scorer = CandidateScorer(llm=mock_llm)

    eval_ = scorer.evaluate(
        resume="Resume",
        job_description="Job",
        cover_letter=None,
    )

    assert eval_.worker_fit_score is not None
    assert len(eval_.positive_alignments) > 0


def test_candidate_scorer_enforces_rubric_weights() -> None:
    mock_llm = MockLLM(CANDIDATE_JSON_FULL)
    scorer = CandidateScorer(llm=mock_llm)
    eval_ = scorer.evaluate("r", "j", None)

    expected_weights = {0.20, 0.15}
    for comp in eval_.worker_fit_score.components:
        assert comp.weight in expected_weights


def test_candidate_scorer_fallback_for_missing_component() -> None:
    """When LLM omits a rubric component, we use score 50 and fallback explanation."""
    json_missing_growth = """
    {
      "positive_alignments": [],
      "concerns": [],
      "questions_to_ask": [],
      "worker_fit_score": {
        "value": 65,
        "components": [
          {"name": "Likely interest alignment", "weight": 0.20, "score": 80, "explanation": "x"},
          {"name": "Role clarity", "weight": 0.15, "score": 80, "explanation": "x"},
          {"name": "Compensation/benefits transparency", "weight": 0.20, "score": 50, "explanation": "x"},
          {"name": "Work-life balance signals", "weight": 0.15, "score": 70, "explanation": "x"},
          {"name": "Red-flag absence", "weight": 0.15, "score": 75, "explanation": "x"}
        ]
      },
      "score_rationale": "OK"
    }
    """
    mock_llm = MockLLM(json_missing_growth)
    scorer = CandidateScorer(llm=mock_llm)
    eval_ = scorer.evaluate("r", "j", None)

    growth = next(c for c in eval_.worker_fit_score.components if c.name == "Growth/career path")
    assert growth.score == 50.0
    assert growth.explanation == "Component not provided by model"


def test_candidate_scorer_raises_on_invalid_json() -> None:
    mock_llm = MockLLM("Not valid JSON")
    scorer = CandidateScorer(llm=mock_llm)

    with pytest.raises(JSONExtractionError):
        scorer.evaluate("r", "j", None)
