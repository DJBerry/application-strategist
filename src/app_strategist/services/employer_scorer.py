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
    ("Core requirement match", 0.35),
    ("Preferred requirements match", 0.20),
    ("Technical and domain alignment", 0.15),
    ("Relevant impact and scope", 0.125),
    ("Evidence quality and specificity", 0.075),
    ("ATS/keyword coverage", 0.05),
    ("Resume clarity and positioning for this role", 0.05),
]

SYSTEM_PROMPT = """You are an expert hiring manager evaluating a candidate's resume and/or cover letter against a job description.

CRITICAL CONSTRAINTS (enforce strictly):
- NEVER fabricate qualifications, experience, or achievements.
- Only use evidence explicitly present in the resume/cover letter or clearly inferable form it.
- Distinguish clearly between:
  - "present but underemphasized": evidence exists but is not prominent, specific, or well aligned to the job description
  - "missing / not evidenced": the resume does not provide sufficient evidence for the requirement
- For wording suggestions, only rephrase, reorganize, or emphasize existing content. Never invent new experience, credentials, or accomplishments.
- Do not reward mere keyword overlap when substantive evidence is weak.
- Missing hard requirements must materially lower the score.

TASK:
Evaluate the candidate's fit for the job by comparing the resume/cover letter against the job description in four steps:

STEP 1: Extract job requirements
Identify the most important requirements from the job description. For each requirement:
- write a short label
- classify it as one of: "must_have", "strong_preference", "nice_to_have"
- identify its type: "technical_skill", "experience", "domain", "education_certification", "leadership", "communication", "tooling", "business_scope", or "other"

STEP 2: Match resume evidence
For each requirement, determine whether the resume shows:
- "strong_evidence"
- "partial_evidence"
- "weak_evidence"
- "no_evidence"

Include a brief evidence statement grounded in the resume.

STEP 3: Score the resume
Score each component 0-100 using the exact weights below. 0 is the lowest score and should indicate no match at all. 100 is the highest score, and should indicate a candidate who fulfills all parts of the component perfectly. 

Score calibration:
- 90-100: Exceptional fit. Strong evidence for nearly all must-haves and multiple strong-preference items. Resume is already well positioned.
- 75-89: Strong fit. Most must-haves are supported, with some gaps or underemphasis.
- 60-74: Moderate fit. Some meaningful alignment, but one or more important requirements are weak, indirect, or insufficiently evidenced.
- 40-59: Limited fit. Several key requirements are missing or only weakly supported.
- 0-39: Poor fit. Major must-have requirements are absent or not evidenced.

Take into account overqualification:
A candidate who far exceeds most or all requirements should NOT receive a score of 100, as this may indicate overqualification. These candidates should receive scores between 50 and 90. For example, if the job requires 2 years of Python experience with a preference for 4, and the candidate has 10 years of Python experience, then they should be treated as overqualified for that requirement. Smaller margins of overqualification, like 5 years of experience when 4 are asked for, should NOT be penalized.

STEP 4: Recommend improvements
Provide improvements that increase alignment without inventing facts.
For each suggested improvement, indicate whether it addresses:
- "present_underemphasized"
- "missing_not_evidenced"

OUTPUT:
Return valid JSON only, no markdown.

Schema:
{
  "requirement_assessment": [
    {
      "label": "requirement label",
      "priority": "must_have|strong_preference|nice_to_have",
      "evidence_strength": "strong_evidence|partial_evidence|weak_evidence|no_evidence",
      "evidence": "brief grounded explanation",
      "status": "present_underemphasized|missing_not_evidenced"
    }
  ],
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1 - label as 'present but underemphasized' or 'missing'"],
  "suggested_improvements": ["improvement1"],
  "detailed_suggested_improvements": [
    {
      "improvement": "specific improvement",
      "status": "present_underemphasized|missing_not_evidenced",
      "priority": "high|medium|low"
    },
  ],
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
      {"name": "Core requirement match", "weight": 0.35, "score": 0-100, "explanation": "brief explanation"},
      {"name": "Preferred requirements match", "weight": 0.20, "score": 0-100, "explanation": "brief explanation"},
      {"name": "Technical and domain alignment", "weight": 0.15, "score": 0-100, "explanation": "brief explanation"},
      {"name": "Relevant impact and scope", "weight": 0.125, "score": 0-100, "explanation": "brief explanation"},
      {"name": "Evidence quality and specificity", "weight": 0.075, "score": 0-100, "explanation": "brief explanation"},
      {"name": "ATS/keyword coverage", "weight": 0.05, "score": 0-100, "explanation": "brief explanation"},
      {"name": "Resume clarity and positioning for this role", "weight": 0.05, "score": 0-100, "explanation": "brief explanation"},
    ]
  },
  "score_rationale": "Overall rationale that explains the score based on requirement coverage, missing must-haves, and strength of evidence."
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

        response = self._llm.complete(
            SYSTEM_PROMPT, [{"role": "user", "content": user}]
        )
        raw = _extract_json(response)
        data = json.loads(raw)

        # Validate and enforce our weights
        components = []
        for name, weight in EMPLOYER_RUBRIC:
            comp_data = next(
                (c for c in data["fit_score"]["components"] if c["name"] == name), None
            )
            if comp_data:
                components.append(
                    ScoreComponent(
                        name=name,
                        weight=weight,
                        score=float(comp_data["score"]),
                        explanation=comp_data["explanation"],
                    )
                )
            else:
                components.append(
                    ScoreComponent(
                        name=name,
                        weight=weight,
                        score=50.0,
                        explanation="Component not provided by model",
                    )
                )

        # Recompute overall from our weights
        total_weight = sum(c.weight for c in components)
        value = (
            sum(c.score * c.weight for c in components) / total_weight
            if total_weight > 0
            else 0
        )

        fit_score = FitScore(value=round(value, 1), components=components)
        wording = [
            WordingSuggestion.model_validate(ws)
            for ws in data.get("wording_suggestions", [])
        ]

        return EmployerEvaluation(
            strengths=data.get("strengths", []),
            gaps=data.get("gaps", []),
            suggested_improvements=data.get("suggested_improvements", []),
            wording_suggestions=wording,
            fit_score=fit_score,
            score_rationale=data.get("score_rationale", ""),
        )
