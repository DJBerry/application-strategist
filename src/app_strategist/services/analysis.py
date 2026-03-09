"""Analysis service - orchestrates parsing, scoring, and evaluation."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app_strategist.config import get_llm_provider
from app_strategist.models.session import AnalysisSession
from app_strategist.parsers import DocumentParserRegistry, JobDescriptionParserRegistry
from app_strategist.services.candidate_scorer import CandidateScorer
from app_strategist.services.employer_scorer import EmployerScorer

if TYPE_CHECKING:
    from app_strategist.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AnalysisService:
    """Orchestrates the full analysis pipeline."""

    def __init__(
        self,
        *,
        doc_registry: DocumentParserRegistry | None = None,
        job_registry: JobDescriptionParserRegistry | None = None,
        llm: "LLMProvider | None" = None,
    ) -> None:
        self._doc_registry = doc_registry or DocumentParserRegistry()
        self._job_registry = job_registry or JobDescriptionParserRegistry()
        self._llm = llm or get_llm_provider()
        self._employer_scorer = EmployerScorer(self._llm)
        self._candidate_scorer = CandidateScorer(self._llm)

    def analyze(
        self,
        resume_path: Path,
        job_path: Path,
        cover_letter_path: Path | None = None,
        *,
        validate: bool = False,
    ) -> AnalysisSession:
        """Parse inputs and produce full analysis (employer + candidate evaluations)."""
        logger.info("Parsing resume: %s", resume_path)
        resume_content = self._doc_registry.parse(resume_path)

        logger.info("Parsing job description: %s", job_path)
        job_description = self._job_registry.parse(job_path)

        cover_letter_content: str | None = None
        if cover_letter_path:
            logger.info("Parsing cover letter: %s", cover_letter_path)
            cover_letter_content = self._doc_registry.parse(cover_letter_path)

        extraction_result = None
        if validate:
            from app_strategist.services.orchestrator import (
                AgentStatus,
                run_extraction_validation,
            )

            logger.info("Running extraction-validation for resume")
            resume_result = run_extraction_validation(
                document_text=resume_content,
                document_type="resume",
                task_description="Extract claims from this resume. Identify candidate experience, achievements, and responsibilities explicitly stated.",
            )

            logger.info("Running extraction-validation for job description")
            jd_result = run_extraction_validation(
                document_text=job_description,
                document_type="job_description",
                task_description="Extract claims from this job description. Identify required qualifications, preferred qualifications, and responsibilities explicitly stated.",
            )

            # Combine extraction outputs and audit trails
            combined_output = {
                "resume": resume_result.final_output,
                "job_description": jd_result.final_output,
            }
            combined_trail = resume_result.audit_trail + jd_result.audit_trail
            combined_unresolvable = (
                resume_result.unresolvable_claims + jd_result.unresolvable_claims
            )
            status = (
                AgentStatus.PASS
                if resume_result.status == AgentStatus.PASS
                and jd_result.status == AgentStatus.PASS
                else AgentStatus.EXHAUSTED
            )

            extraction_result = {
                "final_output": combined_output,
                "status": status,
                "attempts": resume_result.attempts + jd_result.attempts,
                "audit_trail": combined_trail,
                "unresolvable_claims": combined_unresolvable,
            }

        logger.info("Running employer-side evaluation")
        employer_eval = self._employer_scorer.evaluate(
            resume=resume_content,
            job_description=job_description,
            cover_letter=cover_letter_content,
        )

        logger.info("Running candidate-side evaluation")
        candidate_eval = self._candidate_scorer.evaluate(
            resume=resume_content,
            job_description=job_description,
            cover_letter=cover_letter_content,
        )

        return AnalysisSession(
            resume_content=resume_content,
            cover_letter_content=cover_letter_content,
            job_description=job_description,
            employer_eval=employer_eval,
            candidate_eval=candidate_eval,
            extraction_result=extraction_result,
        )
