"""Orchestrator - extraction → validation → retry loop."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app_strategist.mcp_server.resume_validator import (
    get_audit_trail,
    log_validation_event,
    reset_audit_trail,
)
from app_strategist.services.extractor import extract
from app_strategist.services.validator import validate

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class AgentStatus(str, Enum):
    """Status of the extraction-validation run."""

    PASS = "pass"
    FAIL = "fail"
    EXHAUSTED = "exhausted"


@dataclass
class AgentRunResult:
    """Result of the extraction-validation orchestration."""

    final_output: dict[str, list[str]]
    status: AgentStatus
    attempts: int
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    unresolvable_claims: list[dict[str, str]] = field(default_factory=list)


def _strip_unvalidated(
    extraction: dict[str, list[str]],
    failed_claims: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Remove claims that failed validation from the extraction."""
    failed_set = {f.get("claim", "") for f in failed_claims}
    result: dict[str, list[str]] = {}
    for key, items in extraction.items():
        result[key] = [c for c in items if c not in failed_set]
    return result


def run_extraction_validation(
    document_text: str,
    document_type: str,
    task_description: str,
) -> AgentRunResult:
    """Run extraction → validation → retry loop for one document.

    Args:
        document_text: Raw text of resume or job description.
        document_type: "resume" or "job_description".
        task_description: Instruction for extraction (e.g. "Extract claims from this resume").

    Returns:
        AgentRunResult with final_output, status, attempts, audit_trail, unresolvable_claims.
    """
    reset_audit_trail()
    previous_failures: list[dict[str, str]] = []
    extraction: dict[str, list[str]] = {}

    for attempt in range(1, MAX_RETRIES + 1):
        extraction = extract(
            document_text=document_text,
            task_description=task_description,
            previous_failures=previous_failures if previous_failures else None,
        )

        log_validation_event(
            event_type="extraction_produced",
            attempt_number=attempt,
            details={"document_type": document_type},
        )

        validation = validate(
            extraction=extraction,
            document_text=document_text,
            document_type=document_type,
            attempt_number=attempt,
        )

        if validation.all_passed:
            trail = get_audit_trail()
            return AgentRunResult(
                final_output=extraction,
                status=AgentStatus.PASS,
                attempts=attempt,
                audit_trail=trail,
                unresolvable_claims=[],
            )

        previous_failures.extend(validation.failed_claims)
        log_validation_event(
            event_type="recall_triggered",
            attempt_number=attempt,
            details={
                "corrections": [f.get("correction_instruction", "") for f in validation.failed_claims],
                "document_type": document_type,
            },
        )

    log_validation_event(
        event_type="retry_exhausted",
        attempt_number=MAX_RETRIES,
        details={
            "unresolvable_claims": previous_failures,
            "document_type": document_type,
        },
    )

    stripped = _strip_unvalidated(extraction, previous_failures)
    trail = get_audit_trail()

    return AgentRunResult(
        final_output=stripped,
        status=AgentStatus.EXHAUSTED,
        attempts=MAX_RETRIES,
        audit_trail=trail,
        unresolvable_claims=previous_failures,
    )
