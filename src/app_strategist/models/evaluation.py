"""Evaluation models - employer-side and candidate-side outputs."""

from pydantic import BaseModel, Field

from app_strategist.models.scoring import FitScore


class WordingSuggestion(BaseModel):
    """A specific wording improvement suggestion (never invents experience)."""

    current: str = Field(..., description="Current or missing phrasing")
    suggested: str = Field(..., description="Suggested improvement")
    rationale: str = Field(..., description="Why this helps")
    status: str = Field(
        ...,
        description="'present_underemphasized' or 'missing_not_evidenced'",
    )


class EmployerEvaluation(BaseModel):
    """Hiring manager / recruiter perspective evaluation."""

    strengths: list[str] = Field(
        default_factory=list,
        description="Where candidate experience aligns with role",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Missing requirements or underemphasized areas",
    )
    suggested_improvements: list[str] = Field(
        default_factory=list,
        description="General improvements to resume/cover letter",
    )
    wording_suggestions: list[WordingSuggestion] = Field(
        default_factory=list,
        description="Specific wording improvements (no fabrication)",
    )
    fit_score: FitScore = Field(..., description="Employer-side fit score 0-100")
    score_rationale: str = Field(
        ...,
        description="Overall rationale for the fit score",
    )


class CandidateEvaluation(BaseModel):
    """Worker perspective evaluation."""

    positive_alignments: list[str] = Field(
        default_factory=list,
        description="Alignments with candidate interests/background",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Red flags: vague comp, unrealistic scope, suspicious language, etc.",
    )
    questions_to_ask: list[str] = Field(
        default_factory=list,
        description="Questions for recruiter/hiring manager",
    )
    worker_fit_score: FitScore = Field(
        ...,
        description="Candidate-side fit score 0-100",
    )
    score_rationale: str = Field(
        ...,
        description="Overall rationale for the worker fit score",
    )
