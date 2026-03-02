"""Tests for AnalysisSession model."""

import pytest

from app_strategist.models.evaluation import CandidateEvaluation, EmployerEvaluation
from app_strategist.models.scoring import FitScore, ScoreComponent
from app_strategist.models.session import AnalysisSession


def _make_employer_eval() -> EmployerEvaluation:
    components = [
        ScoreComponent(name="A", weight=0.5, score=80, explanation="x"),
        ScoreComponent(name="B", weight=0.5, score=60, explanation="x"),
    ]
    return EmployerEvaluation(
        strengths=["Python"],
        gaps=["AWS"],
        suggested_improvements=[],
        wording_suggestions=[],
        fit_score=FitScore(value=70.0, components=components),
        score_rationale="Strong match",
    )


def _make_candidate_eval() -> CandidateEvaluation:
    components = [
        ScoreComponent(name="X", weight=0.5, score=75, explanation="x"),
        ScoreComponent(name="Y", weight=0.5, score=65, explanation="x"),
    ]
    return CandidateEvaluation(
        positive_alignments=["Growth"],
        concerns=["Vague comp"],
        questions_to_ask=["Salary range?"],
        worker_fit_score=FitScore(value=70.0, components=components),
        score_rationale="Reasonable fit",
    )


def test_to_context_string_includes_resume_and_job() -> None:
    session = AnalysisSession(
        resume_content="My resume text",
        job_description="Job description text",
        employer_eval=_make_employer_eval(),
        candidate_eval=_make_candidate_eval(),
    )
    ctx = session.to_context_string()
    assert "=== RESUME ===" in ctx
    assert "My resume text" in ctx
    assert "=== JOB DESCRIPTION ===" in ctx
    assert "Job description text" in ctx


def test_to_context_string_includes_cover_letter_when_present() -> None:
    session = AnalysisSession(
        resume_content="Resume",
        cover_letter_content="Dear Hiring Manager...",
        job_description="Job",
        employer_eval=_make_employer_eval(),
        candidate_eval=_make_candidate_eval(),
    )
    ctx = session.to_context_string()
    assert "=== COVER LETTER ===" in ctx
    assert "Dear Hiring Manager..." in ctx


def test_to_context_string_omits_cover_letter_when_absent() -> None:
    session = AnalysisSession(
        resume_content="Resume",
        cover_letter_content=None,
        job_description="Job",
        employer_eval=_make_employer_eval(),
        candidate_eval=_make_candidate_eval(),
    )
    ctx = session.to_context_string()
    assert "=== COVER LETTER ===" not in ctx


def test_to_context_string_includes_employer_eval() -> None:
    session = AnalysisSession(
        resume_content="R",
        job_description="J",
        employer_eval=_make_employer_eval(),
        candidate_eval=_make_candidate_eval(),
    )
    ctx = session.to_context_string()
    assert "=== EMPLOYER EVALUATION ===" in ctx
    assert "Fit Score: 70.0/100" in ctx
    assert "Strong match" in ctx
    assert "Python" in ctx
    assert "AWS" in ctx


def test_to_context_string_includes_candidate_eval() -> None:
    session = AnalysisSession(
        resume_content="R",
        job_description="J",
        employer_eval=_make_employer_eval(),
        candidate_eval=_make_candidate_eval(),
    )
    ctx = session.to_context_string()
    assert "=== CANDIDATE EVALUATION ===" in ctx
    assert "Worker Fit Score: 70.0/100" in ctx
    assert "Reasonable fit" in ctx
    assert "Growth" in ctx
    assert "Vague comp" in ctx
