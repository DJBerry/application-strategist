"""Employer-side evaluation - hiring manager/recruiter perspective."""

import json
import logging

from app_strategist.llm.base import LLMProvider
from app_strategist.utils import extract_json as _extract_json
from app_strategist.models.evaluation import EmployerEvaluation, WordingSuggestion
from app_strategist.models.scoring import FitScore, ScoreComponent

logger = logging.getLogger(__name__)

# Fixed rubric weights - no LLM-generated weights
EMPLOYER_RUBRIC = [
    ("Skills/experience alignment", 0.35),
    ("Keyword/match density", 0.25),
    ("Clarity and structure", 0.20),
    ("Quantified achievements", 0.15),
    ("Tailoring to role", 0.05),
]

SYSTEM_PROMPT = """You are an expert hiring manager evaluating a candidate's resume and cover letter against a job description.

CRITICAL CONSTRAINTS (enforce strictly):
- NEVER fabricate qualifications, experience, or achievements. Only suggest improvements based on what is present or clearly inferable from the documents.
- Distinguish clearly: "present but underemphasized" (evidence exists, could be stronger) vs "missing / not evidenced" (no evidence in documents).
- For wording suggestions: only rephrase or emphasize existing content. Never invent new experience, credentials, or accomplishments.

Score each component 0-100 using the exact weights provided. The overall score must be the weighted average of components.

Output valid JSON only, no markdown. Schema:
{
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1 - label as 'present but underemphasized' or 'missing'"],
  "suggested_improvements": ["improvement1"],
  "wording_suggestions": [
    {
      "current": "current or missing phrasing",
      "suggested": "suggested improvement",
      "rationale": "why this helps",
      "status": "present_underemphasized" or "missing_not_evidenced"
    }
  ],
  "fit_score": {
    "value": 0-100,
    "components": [
      {"name": "Skills/experience alignment", "weight": 0.35, "score": 0-100, "explanation": "..."},
      {"name": "Keyword/match density", "weight": 0.25, "score": 0-100, "explanation": "..."},
      {"name": "Clarity and structure", "weight": 0.20, "score": 0-100, "explanation": "..."},
      {"name": "Quantified achievements", "weight": 0.15, "score": 0-100, "explanation": "..."},
      {"name": "Tailoring to role", "weight": 0.05, "score": 0-100, "explanation": "..."}
    ]
  },
  "score_rationale": "Overall rationale for the fit score"
}"""


class EmployerScorer:
    """Produce employer-side evaluation using LLM with fixed rubric."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def evaluate(
        self,
        resume: str,
        job_description: str,
        cover_letter: str | None = None,
    ) -> EmployerEvaluation:
        """Generate employer-side evaluation."""
        doc_parts = [f"## Resume\n{resume}"]
        if cover_letter:
            doc_parts.append(f"## Cover Letter\n{cover_letter}")
        documents = "\n\n".join(doc_parts)

        user = f"""Job Description:
{job_description}

---
Candidate Documents:
{documents}

Evaluate from the hiring manager's perspective. Output JSON only."""

        response = self._llm.complete(SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = _extract_json(response)
        data = json.loads(raw)

        # Validate and enforce our weights
        components = []
        for name, weight in EMPLOYER_RUBRIC:
            comp_data = next((c for c in data["fit_score"]["components"] if c["name"] == name), None)
            if comp_data:
                components.append(ScoreComponent(
                    name=name,
                    weight=weight,
                    score=float(comp_data["score"]),
                    explanation=comp_data["explanation"],
                ))
            else:
                components.append(ScoreComponent(
                    name=name,
                    weight=weight,
                    score=50.0,
                    explanation="Component not provided by model",
                ))

        # Recompute overall from our weights
        total_weight = sum(c.weight for c in components)
        value = sum(c.score * c.weight for c in components) / total_weight if total_weight > 0 else 0

        fit_score = FitScore(value=round(value, 1), components=components)
        wording = [WordingSuggestion.model_validate(ws) for ws in data.get("wording_suggestions", [])]

        return EmployerEvaluation(
            strengths=data.get("strengths", []),
            gaps=data.get("gaps", []),
            suggested_improvements=data.get("suggested_improvements", []),
            wording_suggestions=wording,
            fit_score=fit_score,
            score_rationale=data.get("score_rationale", ""),
        )
