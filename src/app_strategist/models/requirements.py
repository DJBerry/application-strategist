"""Pydantic model for a single extracted job requirement."""

from typing import Literal

from pydantic import BaseModel, Field

RequirementPriority = Literal[
    "minimum_requirement",
    "preferred_requirement",
    "nice_to_have",
    "ambiguous",
]


class JobRequirement(BaseModel):
    """A single requirement extracted from a job description."""

    label: str = Field(..., description="Short human-readable label (e.g., 'Python proficiency')")
    description: str = Field(
        ...,
        description=(
            "Full description with enough context for a downstream LLM to evaluate "
            "whether a candidate's resume fulfils it. Captures all JD nuance and "
            "conditions faithfully; does not add information absent from the source."
        ),
    )
    priority: RequirementPriority = Field(
        ...,
        description=(
            "Priority derived from JD language cues. "
            "'minimum_requirement' for must/required/minimum language; "
            "'preferred_requirement' for preferred/desired/ideal; "
            "'nice_to_have' for bonus/plus/nice-to-have; "
            "'ambiguous' when the JD language is unclear or contradictory."
        ),
    )
