"""MCP server exposing resume validation tools."""

import copy
import json
import logging
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from app_strategist.llm import AnthropicProvider
from app_strategist.services.extractor import EXTRACTION_MODEL

logger = logging.getLogger(__name__)

# In-memory audit trail - reset at start of each validation run
_audit_trail: list[dict[str, Any]] = []


def reset_audit_trail() -> None:
    """Clear the audit trail. Call at start of each validation run."""
    _audit_trail.clear()


def get_audit_trail() -> list[dict[str, Any]]:
    """Return a deep copy of the audit trail."""
    return copy.deepcopy(_audit_trail)


mcp = FastMCP(
    "resume-validator",
    instructions="Tools for validating extracted claims against resume and job description documents.",
)


def _check_claim_via_llm(claim: str, document_text: str) -> dict[str, Any]:
    """Use LLM for semantic matching of claim against document."""
    provider = AnthropicProvider(model=EXTRACTION_MODEL)
    system = """You are a strict fact-checker. Given a CLAIM and a DOCUMENT, determine if the claim is explicitly supported by the document.

Rules:
- A claim is SUPPORTED only if the document explicitly states it or contains clear evidence for it.
- Semantic equivalence counts: "5 years Python" supports "Python experience" or "5+ years Python".
- Do NOT support claims that are inferred, assumed, or only partially suggested.
- If the claim is not in the document, return supported=false with a brief mismatch_reason."""

    user = f"""CLAIM: {claim}

DOCUMENT:
{document_text}

Respond with valid JSON only, no markdown:
{{
  "supported": true or false,
  "confidence": "high" or "medium" or "low",
  "supporting_text": "exact quote from document if supported, else null",
  "mismatch_reason": "brief explanation if not supported, else null"
}}"""

    response = provider.complete(system, [{"role": "user", "content": user}])
    # Extract JSON from response
    start = response.find("{")
    if start < 0:
        return {"supported": False, "confidence": "low", "supporting_text": None, "mismatch_reason": "Could not parse LLM response"}
    try:
        decoder = json.JSONDecoder()
        _, end = decoder.raw_decode(response, start)
        data = json.loads(response[start:end])
        return {
            "supported": bool(data.get("supported", False)),
            "confidence": data.get("confidence", "low") if data.get("confidence") in ("high", "medium", "low") else "low",
            "supporting_text": data.get("supporting_text") or None,
            "mismatch_reason": data.get("mismatch_reason") or None,
        }
    except (json.JSONDecodeError, KeyError):
        return {"supported": False, "confidence": "low", "supporting_text": None, "mismatch_reason": "Could not parse LLM response"}


@mcp.tool()
def check_claim_in_document(
    claim: str,
    claim_type: Literal[
        "required_qualification",
        "preferred_qualification",
        "responsibility",
        "candidate_experience",
        "candidate_achievement",
    ],
    document_type: Literal["job_description", "resume"],
    document_text: str,
) -> dict[str, Any]:
    """Check if a single claim is explicitly supported by the document.

    Uses semantic matching (not just string search) via LLM.
    """
    result = _check_claim_via_llm(claim, document_text)
    return result


@mcp.tool()
def validate_extraction_batch(
    claims: list[dict[str, str]],
    document_type: str,
    document_text: str,
) -> dict[str, Any]:
    """Validate a batch of claims against the document.

    Calls check_claim_in_document for each claim and returns a structured report.
    claims: list of {id, claim, claim_type} objects.
    """
    report = {"results": [], "all_passed": True}
    for c in claims:
        claim_id = c.get("id", "")
        claim = c.get("claim", "")
        claim_type = c.get("claim_type", "candidate_experience")
        result = check_claim_in_document(
            claim=claim,
            claim_type=claim_type,
            document_type=document_type,
            document_text=document_text,
        )
        passed = result.get("supported", False)
        report["results"].append({
            "id": claim_id,
            "claim": claim,
            "claim_type": claim_type,
            "passed": passed,
            "confidence": result.get("confidence", "low"),
            "supporting_text": result.get("supporting_text"),
            "mismatch_reason": result.get("mismatch_reason"),
        })
        if not passed:
            report["all_passed"] = False
    return report


@mcp.tool()
def log_validation_event(
    event_type: Literal[
        "extraction_produced",
        "validation_passed",
        "validation_failed",
        "recall_triggered",
        "retry_exhausted",
        "final_output_accepted",
    ],
    attempt_number: int,
    details: dict[str, Any],
) -> dict[str, str]:
    """Log a validation event to the audit trail."""
    _audit_trail.append({
        "event_type": event_type,
        "attempt_number": attempt_number,
        "details": details,
    })
    return {"status": "logged"}


def run_server() -> None:
    """Run the MCP server (for standalone usage)."""
    mcp.run(transport="stdio")
