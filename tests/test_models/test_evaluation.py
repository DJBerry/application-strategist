"""Tests for evaluation models - WordingSuggestion, EmployerEvaluation, CandidateEvaluation."""

import pytest

from app_strategist.models.evaluation import (
    CandidateEvaluation,
    EmployerEvaluation,
    WordingSuggestion,
)
from app_strategist.models.scoring import FitScore, ScoreComponent


class TestWordingSuggestion:
    def test_valid_present_underemphasized(self) -> None:
        ws = WordingSuggestion(
            current="Built APIs",
            suggested="Built APIs serving 10K req/day",
            rationale="Quantify",
            status="present_underemphasized",
        )
        assert ws.current == "Built APIs"
        assert ws.suggested == "Built APIs serving 10K req/day"
        assert ws.status == "present_underemphasized"

    def test_valid_missing_not_evidenced(self) -> None:
        ws = WordingSuggestion(
            current="Missing k8s",
            suggested="Add if you have k8s",
            rationale="Nice to have",
            status="missing_not_evidenced",
        )
        assert ws.status == "missing_not_evidenced"

    def test_from_dict(self) -> None:
        d = {
            "current": "x",
            "suggested": "y",
            "rationale": "z",
            "status": "present_underemphasized",
        }
        ws = WordingSuggestion.model_validate(d)
        assert ws.current == "x" and ws.suggested == "y"


class TestEmployerEvaluation:
    def test_from_valid_dict(self) -> None:
        fit_components = [
            ScoreComponent(name="A", weight=0.35, score=80, explanation="x"),
            ScoreComponent(name="B", weight=0.65, score=70, explanation="x"),
        ]
        eval_ = EmployerEvaluation(
            strengths=["Python"],
            gaps=["AWS"],
            suggested_improvements=["Add metrics"],
            wording_suggestions=[],
            fit_score=FitScore(value=73.5, components=fit_components),
            score_rationale="Strong",
        )
        assert len(eval_.strengths) == 1
        assert len(eval_.gaps) == 1
        assert eval_.fit_score.value == 73.5

    def test_defaults_empty_lists(self) -> None:
        fit = FitScore(value=70.0, components=[])
        eval_ = EmployerEvaluation(
            fit_score=fit,
            score_rationale="OK",
        )
        assert eval_.strengths == []
        assert eval_.gaps == []
        assert eval_.wording_suggestions == []


class TestCandidateEvaluation:
    def test_from_valid_dict(self) -> None:
        components = [
            ScoreComponent(name="X", weight=0.5, score=75, explanation="x"),
            ScoreComponent(name="Y", weight=0.5, score=65, explanation="x"),
        ]
        eval_ = CandidateEvaluation(
            positive_alignments=["Growth"],
            concerns=["Vague comp"],
            questions_to_ask=["Salary?"],
            worker_fit_score=FitScore(value=70.0, components=components),
            score_rationale="Reasonable",
        )
        assert len(eval_.positive_alignments) == 1
        assert len(eval_.questions_to_ask) == 1
        assert eval_.worker_fit_score.value == 70.0

    def test_defaults_empty_lists(self) -> None:
        fit = FitScore(value=70.0, components=[])
        eval_ = CandidateEvaluation(
            worker_fit_score=fit,
            score_rationale="OK",
        )
        assert eval_.positive_alignments == []
        assert eval_.concerns == []
        assert eval_.questions_to_ask == []
