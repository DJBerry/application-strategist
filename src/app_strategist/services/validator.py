"""Validator agent - verifies extracted claims against source documents via MCP tools."""

import logging
from dataclasses import dataclass
from typing import Any

from app_strategist.mcp_server.resume_validator import (
    log_validation_event,
    validate_extraction_batch,
)

logger = logging.getLogger(__name__)

# Map extraction keys to claim_type enum
_KEY_TO_CLAIM_TYPE = {
    "required_qualifications": "required_qualification",
    "preferred_qualifications": "preferred_qualification",
    "responsibilities": "responsibility",
    "candidate_experience": "candidate_experience",
    "candidate_achievements": "candidate_achievement",
}


def _flatten_to_claims(
    extraction: dict[str, list[str]],
    document_type: str,
) -> list[dict[str, str]]:
    """Map extraction schema to list of {id, claim, claim_type} objects."""
    prefix = "resume" if document_type == "resume" else "jd"
    claims: list[dict[str, str]] = []
    idx = 0
    for key, claim_type in _KEY_TO_CLAIM_TYPE.items():
        items = extraction.get(key, [])
        for i, claim in enumerate(items):
            if claim and isinstance(claim, str):
                claims.append({
                    "id": f"{prefix}_{key}_{i}",
                    "claim": claim,
                    "claim_type": claim_type,
                })
                idx += 1
    return claims


@dataclass
class ValidationResult:
    """Result of validating extraction against a document."""

    all_passed: bool
    failed_claims: list[dict[str, str]]


def validate(
    extraction: dict[str, list[str]],
    document_text: str,
    document_type: str,
    attempt_number: int = 1,
) -> ValidationResult:
    """Validate extracted claims against the source document using MCP tools.

    Args:
        extraction: Structured extraction from extractor.extract().
        document_text: Raw document text (resume or JD).
        document_type: "resume" or "job_description".
        attempt_number: Current attempt number (for audit logging).

    Returns:
        ValidationResult with all_passed and failed_claims (each has claim, correction_instruction).
    """
    claims = _flatten_to_claims(extraction, document_type)
    if not claims:
        log_validation_event(
            event_type="validation_passed",
            attempt_number=attempt_number,
            details={"reason": "no claims to validate"},
        )
        return ValidationResult(all_passed=True, failed_claims=[])

    report = validate_extraction_batch(
        claims=claims,
        document_type=document_type,
        document_text=document_text,
    )

    failed_claims: list[dict[str, str]] = []
    for r in report.get("results", []):
        if not r.get("passed", False):
            failed_claims.append({
                "claim": r.get("claim", ""),
                "correction_instruction": r.get("mismatch_reason", "Claim not supported by document."),
            })

    all_passed = report.get("all_passed", True) and len(failed_claims) == 0

    if all_passed:
        log_validation_event(
            event_type="validation_passed",
            attempt_number=attempt_number,
            details={"claims_validated": len(claims)},
        )
    else:
        log_validation_event(
            event_type="validation_failed",
            attempt_number=attempt_number,
            details={"failed_claims": failed_claims, "total": len(claims)},
        )

    return ValidationResult(all_passed=all_passed, failed_claims=failed_claims)
