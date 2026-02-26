"""Candidate-side evaluation - worker perspective."""

import json
import logging

from app_strategist.llm.base import LLMProvider
from app_strategist.utils import extract_json as _extract_json
from app_strategist.models.evaluation import CandidateEvaluation
from app_strategist.models.scoring import FitScore, ScoreComponent

logger = logging.getLogger(__name__)

# Fixed rubric weights
CANDIDATE_RUBRIC = [
    ("Role clarity", 0.25),
    ("Compensation/benefits transparency", 0.20),
    ("Work-life balance signals", 0.15),
    ("Growth/career path", 0.20),
    ("Red-flag absence", 0.20),
]

SYSTEM_PROMPT = """You are an expert career advisor evaluating a job description from the candidate's perspective.

Identify:
- Positive alignments with the candidate's likely interests/background (based on their resume)
- Potential concerns: vague compensation, unrealistic scope, suspicious language, poor role definition, red flags
- Questions the candidate should ask the recruiter/hiring manager

Score each component 0-100. The overall worker fit score must be the weighted average of components.

Output valid JSON only, no markdown. Schema:
{
  "positive_alignments": ["alignment1", "alignment2"],
  "concerns": ["concern1", "concern2"],
  "questions_to_ask": ["question1", "question2"],
  "worker_fit_score": {
    "value": 0-100,
    "components": [
      {"name": "Role clarity", "weight": 0.25, "score": 0-100, "explanation": "..."},
      {"name": "Compensation/benefits transparency", "weight": 0.20, "score": 0-100, "explanation": "..."},
      {"name": "Work-life balance signals", "weight": 0.15, "score": 0-100, "explanation": "..."},
      {"name": "Growth/career path", "weight": 0.20, "score": 0-100, "explanation": "..."},
      {"name": "Red-flag absence", "weight": 0.20, "score": 0-100, "explanation": "..."}
    ]
  },
  "score_rationale": "Overall rationale for the worker fit score"
}"""


class CandidateScorer:
    """Produce candidate-side evaluation using LLM with fixed rubric."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def evaluate(
        self,
        resume: str,
        job_description: str,
        cover_letter: str | None = None,
    ) -> CandidateEvaluation:
        """Generate candidate-side evaluation."""
        doc_parts = [f"## Resume\n{resume}"]
        if cover_letter:
            doc_parts.append(f"## Cover Letter\n{cover_letter}")
        documents = "\n\n".join(doc_parts)

        user = f"""Job Description:
{job_description}

---
Candidate Documents (for context on candidate background):
{documents}

Evaluate from the worker's perspective. Output JSON only."""

        response = self._llm.complete(SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = _extract_json(response)
        data = json.loads(raw)

        components = []
        for name, weight in CANDIDATE_RUBRIC:
            comp_data = next(
                (c for c in data["worker_fit_score"]["components"] if c["name"] == name),
                None,
            )
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

        total_weight = sum(c.weight for c in components)
        value = sum(c.score * c.weight for c in components) / total_weight if total_weight > 0 else 0

        worker_fit_score = FitScore(value=round(value, 1), components=components)

        return CandidateEvaluation(
            positive_alignments=data.get("positive_alignments", []),
            concerns=data.get("concerns", []),
            questions_to_ask=data.get("questions_to_ask", []),
            worker_fit_score=worker_fit_score,
            score_rationale=data.get("score_rationale", ""),
        )
