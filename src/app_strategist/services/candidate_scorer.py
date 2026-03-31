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
    ("Interest and mission alignment", 0.20),
    ("Scope and seniority fit", 0.20),
    ("Role clarity and credibility", 0.20),
    ("Compensation and employment terms transparency", 0.15),
    ("Work environment and sustainability", 0.15),
    ("Career value and growth potential", 0.10),
    ("Apply-worthiness / competitive payoff", 0.10),
]

SYSTEM_PROMPT = """You are an expert career advisor evaluating a job opportunity from the candidate's perspective.

Your goal is to help answer two questions:
1) Should this candidate apply for this job?
2) If yes, what aspects of the candidate's background seem most important to emphasize or clarify before applying?

CRITICAL CONSTRAINTS (enforce strictly):
- NEVER fabricate candidate interests, preferences, constraints, deal-breakers, qualifications, or career goals.
- Only use evidence explicitly present in the resume and cover letter, or clearly inferable from them.
- Only use evidence explicitly present in the job description when assessing the opportunity, risks, and employer signals.
- Treat the resume as the primary source of evidence about the candidate. Use the cover letter only if provided.
- Distinguish clearly between:
  - "supported positive": a meaningful positive supported by both the candidate materials and the job description
  - "supported concern": a risk, downside, or red flag supported by the job description
  - "uncertain due to missing information": an important unknown caused by lack of detail in the job description
- Do not treat generic employer branding, prestige language, culture slogans, or buzzwords as meaningful positives unless supported by concrete role details.
- Do not invent hidden benefits, team quality, manager quality, growth opportunities, or work-life balance when these are not evidenced.
- Do not over-penalize missing information as if it were explicit negative evidence, but do reduce relevant component scores when important information is absent.
- Strong interest alignment must NOT fully offset serious red flags, major ambiguity, poor transparency, or evidence that the role is likely a poor use of the candidate's time.
- Avoid generic advice. Keep all judgments concrete, specific, and decision-useful.

TASK:
Evaluate the opportunity from the candidate's perspective using the following steps.

STEP 1: Characterize the opportunity
Based only on the job description, determine what kind of role this appears to be.
Identify:
- the apparent function of the role
- the apparent seniority or level
- the likely scope and type of work
- the domain, product area, or mission if stated
- practical terms explicitly stated, such as remote/hybrid/on-site, location, travel, on-call, employment type, compensation, or benefits
- important missing information separately from explicit facts

STEP 2: Infer candidate-relevant interests and priorities from evidence
Using only the resume and optional cover letter, identify the candidate's likely interests, strengths, preferred type of work, demonstrated domains, apparent seniority, and likely career direction.
- Do not invent preferences or constraints that are not supported by evidence.
- Treat missing evidence as uncertainty, not as a mismatch.
- Focus on what the candidate appears to value or be best positioned for based on their documented background.

STEP 3: Evaluate the role across the seven candidate-side components
Score each component from 0 to 100 using the exact weights below.
Base each score on explicit evidence and important omissions.
Avoid double counting the same issue across multiple components unless it genuinely affects more than one component.

STEP 4: Identify the most important positives, concerns, and unresolved unknowns
Determine:
- the strongest supported positive alignments between the role and candidate
- the most decision-relevant concerns, including both supported concerns and important unknowns
- the highest-value questions the candidate should ask to clarify whether the role is worth pursuing

STEP 5: Assess whether the role is worth pursuing
Form an overall judgment about whether this role appears worth pursuing for this candidate, taking into account:
- how attractive the work appears
- whether the scope and level fit the candidate
- how credible and well-defined the role appears
- practical terms and sustainability
- whether applying seems like a worthwhile use of time and effort

STEP 6: Identify application strategy implications
If the role appears worth pursuing, determine which aspects of the candidate's background seem most worth emphasizing or clarifying in a tailored application.
Use this to inform the positive_alignments, questions_to_ask, and score_rationale.

SCORING:
Score each component from 0 to 100 using the exact weights below.

Score calibration:
- 90-100: Exceptionally strong signal for this component; clear, concrete, and compelling evidence with minimal risk or ambiguity
- 75-89: Strong signal; meaningful positives with only minor gaps or moderate unknowns
- 60-74: Mixed signal; some meaningful positives, but notable ambiguity, tradeoffs, or concerns
- 40-59: Weak signal; substantial concerns, poor fit, or lack of important information
- 0-39: Very weak signal; major problems, severe ambiguity, or strong evidence that this dimension is unattractive for the candidate

Component guidance:
1) Interest and mission alignment
- Score how well the actual work, domain, and mission appear to align with the candidate's demonstrated interests, strengths, and likely motivations.
- Prioritize substantive alignment over superficial keyword overlap.
- Do not assume mission alignment unless supported by evidence.

2) Scope and seniority fit
- Score how well the role's apparent level, ownership, breadth, autonomy, and expected impact fit the candidate's background.
- Materially lower this score if the role appears significantly too junior, too senior, too narrow, too managerial, too execution-only, or unrealistically broad for the candidate.

3) Role clarity and credibility
- Score how clearly and credibly the job description defines responsibilities, expectations, priorities, success metrics, and organizational context.
- Materially lower this score if the role lacks clear scope, mixes conflicting expectations, resembles a generic catch-all requisition, or leaves major basics undefined.

4) Compensation and employment terms transparency
- Score the clarity and attractiveness of practical terms as described in the listing.
- Consider salary range, bonus/equity, benefits, employment type, location, work arrangement, travel, on-call, relocation, visa constraints, and other material terms if stated.
- Missing key practical terms should materially lower this score, but should not automatically collapse the overall score.

5) Work environment and sustainability
- Score whether the role appears sustainable, well-supported, and reasonably bounded.
- Consider workload signals, pace, breadth, urgency language, staffing assumptions, travel burden, on-call burden, and after-hours expectations when stated or strongly implied.
- Materially lower this score for explicit burnout signals, contradictory expectations, or recurring signs of chronic overload.

6) Career value and growth potential
- Score whether this role appears likely to build meaningful skills, increase scope, improve career trajectory, or create strong future opportunities for the candidate.
- Only score highly when growth, ownership expansion, mentorship, advancement, or strong career value is supported by concrete evidence rather than generic promises.

7) Apply-worthiness / competitive payoff
- Score whether applying appears likely to be a worthwhile use of the candidate's time.
- Consider whether the candidate appears plausibly competitive, whether the upside seems meaningful, whether the gaps seem bridgeable through positioning, and whether the opportunity appears credible enough to justify the effort.
- Lower this score when the role seems like a poor return on application effort, even if some aspects are attractive.

SCORING RULES:
- The overall worker_fit_score.value must be the weighted average of the component scores.
- Strong positives in one component should not erase serious weaknesses in another.
- Missing evidence should lower confidence and relevant component scores, but should not be described as explicit negative evidence unless the job description clearly supports that conclusion.

OUTPUT REQUIREMENTS:
- positive_alignments should contain the strongest concrete reasons this role may appeal to the candidate or align with their background.
- concerns should contain the most important risks, downsides, red flags, and unresolved unknowns from the candidate's perspective.
- questions_to_ask should contain the highest-value direct questions the candidate should ask the recruiter or hiring manager.
- Keep list items specific, concise, non-redundant, and grounded in the inputs.
- Component explanations must reference the strongest supporting evidence and/or the most important missing information.
- score_rationale must synthesize the main positives, the main concerns, and whether the role seems worth pursuing for this candidate.
- Use concerns and score_rationale to reflect both supported concerns and uncertainty due to missing information.
- Do not output chain-of-thought or the step-by-step reasoning process.

Return JSON matching exactly this structure:
{
  "positive_alignments": ["alignment1", "alignment2"],
  "concerns": ["concern1", "concern2"],
  "questions_to_ask": ["question1", "question2"],
  "worker_fit_score": {
    "value": 0-100,
    "components": [
      {"name": "Interest and mission alignment", "weight": 0.18, "score": 0-100, "explanation": "..."},
      {"name": "Scope and seniority fit", "weight": 0.17, "score": 0-100, "explanation": "..."},
      {"name": "Role clarity and credibility", "weight": 0.16, "score": 0-100, "explanation": "..."},
      {"name": "Compensation and employment terms transparency", "weight": 0.13, "score": 0-100, "explanation": "..."},
      {"name": "Work environment and sustainability", "weight": 0.13, "score": 0-100, "explanation": "..."},
      {"name": "Career value and growth potential", "weight": 0.13, "score": 0-100, "explanation": "..."},
      {"name": "Apply-worthiness / competitive payoff", "weight": 0.10, "score": 0-100, "explanation": "..."}
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

        response = self._llm.complete(
            SYSTEM_PROMPT, [{"role": "user", "content": user}]
        )
        raw = _extract_json(response)
        data = json.loads(raw)

        components = []
        for name, weight in CANDIDATE_RUBRIC:
            comp_data = next(
                (
                    c
                    for c in data["worker_fit_score"]["components"]
                    if c["name"] == name
                ),
                None,
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

        total_weight = sum(c.weight for c in components)
        value = (
            sum(c.score * c.weight for c in components) / total_weight
            if total_weight > 0
            else 0
        )

        worker_fit_score = FitScore(value=round(value, 1), components=components)

        return CandidateEvaluation(
            positive_alignments=data.get("positive_alignments", []),
            concerns=data.get("concerns", []),
            questions_to_ask=data.get("questions_to_ask", []),
            worker_fit_score=worker_fit_score,
            score_rationale=data.get("score_rationale", ""),
        )
