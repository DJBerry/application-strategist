"""Scoring models - explainable, weighted components."""

from pydantic import BaseModel, Field

from app_strategist.models.requirements import RequirementPriority


class ScoreAttempt(BaseModel):
    """A single scoring attempt for a qualification."""

    score: int = Field(..., ge=0, le=4)
    rejection_reason: str | None = None


class QualificationScore(BaseModel):
    """Evidence-based score for one qualification against the candidate's documents."""

    label: str
    description: str
    priority: RequirementPriority
    is_implicit: bool
    evidence: list[str]
    attempts: list[ScoreAttempt]
    final_score: int = Field(..., ge=0, le=4)
    unresolved: bool = False


class QualificationScoringResult(BaseModel):
    """Scored qualification pool with per-priority totals."""

    qualifications: list[QualificationScore]
    totals: dict[str, float]
    warnings: list[str]


class ScoreComponent(BaseModel):
    """A single weighted component of a fit score."""

    name: str = Field(..., description="Component name (e.g., 'Skills alignment')")
    weight: float = Field(..., ge=0, le=1, description="Weight as fraction (e.g., 0.35 for 35%)")
    score: float = Field(..., ge=0, le=100, description="Score for this component (0-100)")
    explanation: str = Field(..., description="Brief explanation of the score")


class FitScore(BaseModel):
    """Overall fit score with weighted components."""

    value: float = Field(..., ge=0, le=100, description="Overall score (0-100)")
    components: list[ScoreComponent] = Field(
        default_factory=list,
        description="Weighted component breakdown",
    )

    def aggregate_from_components(self) -> float:
        """Compute weighted average from components. Used to validate LLM output."""
        if not self.components:
            return self.value
        total_weight = sum(c.weight for c in self.components)
        if total_weight <= 0:
            return self.value
        return sum(c.score * c.weight for c in self.components) / total_weight
