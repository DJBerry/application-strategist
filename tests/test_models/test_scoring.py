"""Tests for scoring models."""

import pytest

from app_strategist.models.scoring import FitScore, ScoreComponent


def test_score_component_validation() -> None:
    c = ScoreComponent(name="Test", weight=0.35, score=80, explanation="Good")
    assert c.name == "Test"
    assert c.weight == 0.35
    assert c.score == 80


def test_score_component_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError):
        ScoreComponent(name="Test", weight=1.5, score=80, explanation="x")


def test_score_component_rejects_invalid_score() -> None:
    with pytest.raises(ValueError):
        ScoreComponent(name="Test", weight=0.5, score=150, explanation="x")


def test_fit_score_aggregate_from_components() -> None:
    components = [
        ScoreComponent(name="A", weight=0.5, score=80, explanation="x"),
        ScoreComponent(name="B", weight=0.5, score=60, explanation="x"),
    ]
    fs = FitScore(value=70, components=components)
    expected = 0.5 * 80 + 0.5 * 60
    assert fs.aggregate_from_components() == expected


def test_fit_score_value_bounds() -> None:
    with pytest.raises(ValueError):
        FitScore(value=150, components=[])
