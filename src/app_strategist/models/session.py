"""Session model - full context for REPL follow-up."""

from typing import Any

from pydantic import BaseModel, Field

from app_strategist.models.evaluation import CandidateEvaluation, EmployerEvaluation


class AnalysisSession(BaseModel):
    """Full analysis context for REPL - documents and evaluations."""

    resume_content: str = Field(..., description="Parsed resume text")
    cover_letter_content: str | None = Field(
        default=None,
        description="Parsed cover letter text (optional)",
    )
    job_description: str = Field(..., description="Parsed job description text")
    employer_eval: EmployerEvaluation = Field(
        ...,
        description="Employer-side evaluation",
    )
    candidate_eval: CandidateEvaluation = Field(
        ...,
        description="Candidate-side evaluation",
    )
    extraction_result: dict[str, Any] | None = Field(
        default=None,
        description="Optional validated extraction and audit trail from --validate",
    )

    def to_context_string(self) -> str:
        """Serialize session for LLM context in follow-up messages."""
        parts = [
            "=== RESUME ===",
            self.resume_content,
            "",
            "=== JOB DESCRIPTION ===",
            self.job_description,
        ]
        if self.cover_letter_content:
            parts.extend(["", "=== COVER LETTER ===", self.cover_letter_content])
        parts.extend([
            "",
            "=== EMPLOYER EVALUATION ===",
            f"Fit Score: {self.employer_eval.fit_score.value}/100",
            f"Rationale: {self.employer_eval.score_rationale}",
            "Strengths: " + "; ".join(self.employer_eval.strengths),
            "Gaps: " + "; ".join(self.employer_eval.gaps),
            "",
            "=== CANDIDATE EVALUATION ===",
            f"Worker Fit Score: {self.candidate_eval.worker_fit_score.value}/100",
            f"Rationale: {self.candidate_eval.score_rationale}",
            "Positive alignments: " + "; ".join(self.candidate_eval.positive_alignments),
            "Concerns: " + "; ".join(self.candidate_eval.concerns),
        ])
        return "\n".join(parts)
