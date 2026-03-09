"""Structured claim extraction - extract only what is explicitly stated."""

import json
import logging
from typing import Any

from app_strategist.llm import AnthropicProvider
from app_strategist.utils import extract_json as _extract_json

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

EXTRACTION_SCHEMA = """
{
  "required_qualifications": ["claim1", "claim2"],
  "preferred_qualifications": ["claim1", "claim2"],
  "candidate_experience": ["claim1", "claim2"],
  "candidate_achievements": ["claim1", "claim2"],
  "responsibilities": ["claim1", "claim2"]
}

- Use empty lists [] for categories not applicable to the document (e.g. resume has no required_qualifications).
- Each string must be a single, atomic claim explicitly stated in the document.
- NEVER infer. If something is implied but not stated, do not include it.
"""

BASE_SYSTEM_PROMPT = f"""You are a strict extraction agent. Your job is to extract structured claims from documents.

CRITICAL: Extract ONLY what is explicitly stated in the document. NEVER infer, extrapolate, or assume.
If the document does not mention something, do not include it. Prefer empty lists over fabricated claims.

Output valid JSON only, no markdown. Schema:
{EXTRACTION_SCHEMA}"""

CORRECTIONS_BLOCK = """
CORRECTIONS FROM PREVIOUS ATTEMPT (do not repeat these errors):
{corrections}
"""


def _build_corrections_text(failures: list[dict[str, str]]) -> str:
    """Format previous failures as correction instructions."""
    lines = []
    for i, f in enumerate(failures, 1):
        claim = f.get("claim", "")
        instruction = f.get("correction_instruction", "")
        claim_preview = claim[:100] + "..." if len(claim) > 100 else claim
        lines.append(f"{i}. Claim: \"{claim_preview}\"")
        lines.append(f"   Correction: {instruction}")
    return "\n".join(lines)


def extract(
    document_text: str,
    task_description: str,
    previous_failures: list[dict[str, str]] | None = None,
    *,
    llm: AnthropicProvider | None = None,
) -> dict[str, list[str]]:
    """Extract structured claims from a document.

    Args:
        document_text: Raw text of the resume or job description.
        task_description: Describes what to extract (e.g. "Extract claims from this job description").
        previous_failures: Optional list of {{claim, correction_instruction}} from failed validations.

    Returns:
        Structured extraction with keys: required_qualifications, preferred_qualifications,
        candidate_experience, candidate_achievements, responsibilities.
    """
    system = BASE_SYSTEM_PROMPT
    if previous_failures:
        corrections = _build_corrections_text(previous_failures)
        system = CORRECTIONS_BLOCK.format(corrections=corrections) + "\n" + system

    user = f"""{task_description}

Document:
{document_text}

Output valid JSON only."""

    provider = llm or AnthropicProvider(model=EXTRACTION_MODEL)
    response = provider.complete(system, [{"role": "user", "content": user}])
    raw = _extract_json(response)
    data: dict[str, Any] = json.loads(raw)

    # Normalize to expected schema
    schema_keys = [
        "required_qualifications",
        "preferred_qualifications",
        "candidate_experience",
        "candidate_achievements",
        "responsibilities",
    ]
    result: dict[str, list[str]] = {}
    for key in schema_keys:
        val = data.get(key)
        if isinstance(val, list):
            result[key] = [str(x) for x in val]
        else:
            result[key] = []

    return result
