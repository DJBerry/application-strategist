"""Extract key requirements from job description (for context, not used in scoring)."""

import json
import logging
from pydantic import BaseModel, Field

from app_strategist.llm.base import LLMProvider
from app_strategist.utils import extract_json

logger = logging.getLogger(__name__)


class ExtractedRequirements(BaseModel):
    """Structured extraction of job requirements."""

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_years: str | None = None
    key_responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)


class RequirementExtractor:
    """Extract structured requirements from job description text."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def extract(self, job_description: str) -> ExtractedRequirements:
        """Extract key requirements from job description."""
        system = """You are an expert at analyzing job descriptions. Extract key requirements as structured JSON.
Output ONLY valid JSON matching this schema:
{
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1"],
  "experience_years": "3+ years" or null,
  "key_responsibilities": ["responsibility1"],
  "qualifications": ["qual1", "qual2"]
}"""

        user = f"Extract requirements from this job description:\n\n{job_description}"
        response = self._llm.complete(system, [{"role": "user", "content": user}])
        text = extract_json(response)
        data = json.loads(text)
        return ExtractedRequirements.model_validate(data)
