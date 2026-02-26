"""Scoring models - explainable, weighted components."""

from pydantic import BaseModel, Field


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
