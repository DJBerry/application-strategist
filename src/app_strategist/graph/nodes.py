"""LangGraph node functions for the job description extraction graph.

Each node that calls the LLM is built with a factory function so the LLMProvider
can be injected at graph-construction time (clean for testing, no global state).

Node signatures all follow the LangGraph convention:
    node_fn(state: GraphState) -> dict   # dict is merged into state

Routing:
    route_after_check(state: GraphState) -> str   # returns edge key
"""

import json
import logging
from collections.abc import Callable

from app_strategist.llm.base import LLMProvider
from app_strategist.utils import extract_json
from app_strategist.graph.state import GraphState
from app_strategist.models.requirements import JobRequirement
from app_strategist.graph.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_USER_TEMPLATE,
    CHECK_SYSTEM_PROMPT,
    CHECK_USER_TEMPLATE,
    RETRY_SYSTEM_PROMPT_TEMPLATE,
    RETRY_USER_TEMPLATE,
    EXTRACT_REQUIREMENTS_SYSTEM_PROMPT,
    EXTRACT_REQUIREMENTS_USER_TEMPLATE,
    VALIDATE_REQUIREMENTS_SYSTEM_PROMPT,
    VALIDATE_REQUIREMENTS_USER_TEMPLATE,
    CORRECT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE,
    CORRECT_REQUIREMENTS_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_REQUIREMENTS_ATTEMPTS = 3  # 1 initial extraction + 2 correction retries


# ---------------------------------------------------------------------------
# Factory: extract node
# ---------------------------------------------------------------------------

def make_extract_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return an extract_node function that uses *llm* to extract job data."""

    def extract_node(state: GraphState) -> dict:
        user = EXTRACT_USER_TEMPLATE.format(job_description=state["job_description"])
        logger.debug("extract_node: calling LLM (attempt %d)", state["attempt_count"] + 1)
        response = llm.complete(EXTRACT_SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = extract_json(response)
        extracted_data = json.loads(raw)
        return {
            "extracted_data": extracted_data,
            "attempt_count": state["attempt_count"] + 1,
        }

    return extract_node


# ---------------------------------------------------------------------------
# Factory: check node
# ---------------------------------------------------------------------------

def make_check_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a check_node function that validates extracted data against the source."""

    def check_node(state: GraphState) -> dict:
        user = CHECK_USER_TEMPLATE.format(
            job_description=state["job_description"],
            extracted_data=json.dumps(state["extracted_data"], indent=2),
        )
        logger.debug("check_node: validating extraction (attempt %d)", state["attempt_count"])
        response = llm.complete(CHECK_SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = extract_json(response)
        validation_result = json.loads(raw)

        ambiguous_fields: list[dict] = validation_result.get("ambiguous_fields", [])
        merged_caveats = state.get("field_caveats", []) + ambiguous_fields

        # Validation passes when there are no incorrect fields; ambiguous fields
        # do not block the "ok" routing path.
        passed = not bool(validation_result.get("incorrect_fields"))
        if passed:
            logger.debug("check_node: validation passed")
        else:
            bad = [f["field"] for f in validation_result.get("incorrect_fields", [])]
            logger.debug("check_node: validation failed — incorrect fields: %s", bad)
        if ambiguous_fields:
            ambig = [f["field"] for f in ambiguous_fields]
            logger.debug("check_node: ambiguous fields (stored as caveats): %s", ambig)

        return {
            "validation_passed": passed,
            "validation_result": validation_result,
            "field_caveats": merged_caveats,
        }

    return check_node


# ---------------------------------------------------------------------------
# Factory: retry node
# ---------------------------------------------------------------------------

def make_retry_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a retry_node that re-extracts only the fields that failed validation."""

    def retry_node(state: GraphState) -> dict:
        incorrect_fields: list[dict] = (
            state["validation_result"].get("incorrect_fields", [])
            if state["validation_result"]
            else []
        )

        # Build a human-readable mistake list to embed in the system prompt.
        # We use plain text (not raw JSON) to avoid brace-escaping conflicts with
        # str.format() in RETRY_SYSTEM_PROMPT_TEMPLATE.
        mistakes_lines = []
        for item in incorrect_fields:
            field = item.get("field", "unknown")
            extracted = item.get("extracted_value", "")
            explanation = item.get("explanation", "")
            mistakes_lines.append(
                f"- {field}: previously extracted as {extracted!r} — {explanation}"
            )
        mistakes_str = "\n".join(mistakes_lines) if mistakes_lines else "(no details provided)"

        field_names = ", ".join(item.get("field", "") for item in incorrect_fields)

        system = RETRY_SYSTEM_PROMPT_TEMPLATE.format(mistakes=mistakes_str)
        user = RETRY_USER_TEMPLATE.format(
            job_description=state["job_description"],
            incorrect_fields=field_names,
        )

        logger.debug(
            "retry_node: re-extracting fields [%s] (attempt %d → %d)",
            field_names,
            state["attempt_count"],
            state["attempt_count"] + 1,
        )
        response = llm.complete(system, [{"role": "user", "content": user}])
        raw = extract_json(response)
        partial_result = json.loads(raw)

        # Merge only the corrected fields; preserve everything else.
        merged = _deep_merge(state["extracted_data"] or {}, partial_result)
        return {
            "extracted_data": merged,
            "attempt_count": state["attempt_count"] + 1,
        }

    return retry_node


# ---------------------------------------------------------------------------
# Plain node: finalize (no LLM)
# ---------------------------------------------------------------------------

def finalize_node(state: GraphState) -> dict:
    """Convert unresolved validation errors into human-readable concerns.

    Called only when max retries are exhausted and validation is still failing.
    The graph proceeds with the best-effort extracted_data and attaches the
    outstanding issues as metadata.
    """
    concerns: list[str] = []
    if state.get("validation_result") and not state.get("validation_passed"):
        for item in state["validation_result"].get("incorrect_fields", []):
            field = item.get("field", "unknown")
            extracted = item.get("extracted_value", "")
            explanation = item.get("explanation", "")
            concerns.append(
                f"Field '{field}': extracted {extracted!r} — {explanation}"
            )
    logger.debug("finalize_node: %d unresolved concern(s)", len(concerns))
    return {"unresolved_concerns": concerns}


# ---------------------------------------------------------------------------
# Conditional edge: route after check
# ---------------------------------------------------------------------------

def route_after_check(state: GraphState) -> str:
    """Decide the next step after the check node.

    Returns:
        "ok"       — validation passed, proceed to END
        "retry"    — validation failed, attempts remain, loop back to retry
        "give_up"  — validation failed, max attempts reached, proceed to finalize
    """
    if state["validation_passed"]:
        return "ok"
    if state["attempt_count"] >= MAX_ATTEMPTS:
        return "give_up"
    return "retry"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Factory: extract_requirements node
# ---------------------------------------------------------------------------

def make_extract_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return an extract_requirements_node that extracts every JD requirement."""

    def extract_requirements_node(state: GraphState) -> dict:
        extracted = state.get("extracted_data") or {}
        company_info = json.dumps(extracted.get("company_info", {}), indent=2)
        job_info = json.dumps(extracted.get("job_info", {}), indent=2)
        user = EXTRACT_REQUIREMENTS_USER_TEMPLATE.format(
            job_description=state["job_description"],
            company_info=company_info,
            job_info=job_info,
        )
        logger.debug(
            "extract_requirements_node: calling LLM (attempt %d)",
            state["requirements_attempt_count"] + 1,
        )
        response = llm.complete(EXTRACT_REQUIREMENTS_SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = extract_json(response)
        data = json.loads(raw)
        requirements = [
            JobRequirement.model_validate(r).model_dump()
            for r in data.get("requirements", [])
        ]
        return {
            "job_requirements": requirements,
            "requirements_attempt_count": state["requirements_attempt_count"] + 1,
        }

    return extract_requirements_node


# ---------------------------------------------------------------------------
# Factory: validate_requirements node
# ---------------------------------------------------------------------------

def make_validate_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a validate_requirements_node that checks extracted requirements."""

    def validate_requirements_node(state: GraphState) -> dict:
        user = VALIDATE_REQUIREMENTS_USER_TEMPLATE.format(
            job_description=state["job_description"],
            requirements=json.dumps(state["job_requirements"], indent=2),
        )
        logger.debug(
            "validate_requirements_node: validating %d requirement(s) (attempt %d)",
            len(state["job_requirements"] or []),
            state["requirements_attempt_count"],
        )
        response = llm.complete(VALIDATE_REQUIREMENTS_SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = extract_json(response)
        validation_result = json.loads(raw)

        issues: list[dict] = validation_result.get("issues", [])
        passed = not bool(issues)
        if passed:
            logger.debug("validate_requirements_node: validation passed")
        else:
            types = [i.get("type") for i in issues]
            logger.debug("validate_requirements_node: validation failed — issue types: %s", types)

        return {
            "requirements_validation_passed": passed,
            "requirements_validation_result": validation_result,
        }

    return validate_requirements_node


# ---------------------------------------------------------------------------
# Factory: correct_requirements node
# ---------------------------------------------------------------------------

def make_correct_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a correct_requirements_node that applies validator corrections."""

    def correct_requirements_node(state: GraphState) -> dict:
        issues: list[dict] = (
            state["requirements_validation_result"].get("issues", [])
            if state["requirements_validation_result"]
            else []
        )

        # Build a human-readable issues list to embed in the system prompt.
        # We use plain text (not raw JSON) to avoid brace-escaping conflicts
        # with str.format() in CORRECT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE.
        issue_lines = []
        for item in issues:
            issue_type = item.get("type", "unknown")
            label = item.get("label") or "(new requirement)"
            problem = item.get("problem", "")
            correction = item.get("correction", {})
            correction_label = correction.get("label", "")
            correction_desc = correction.get("description", "")
            correction_priority = correction.get("priority", "")
            if issue_type == "duplicate":
                duplicate_of = item.get("duplicate_of") or ""
                issue_lines.append(
                    f"- [{issue_type}] '{label}' and '{duplicate_of}': {problem}\n"
                    f"  Merged correction: label={correction_label!r}, "
                    f"description={correction_desc!r}, priority={correction_priority!r}"
                )
            else:
                issue_lines.append(
                    f"- [{issue_type}] '{label}': {problem}\n"
                    f"  Correction: label={correction_label!r}, "
                    f"description={correction_desc!r}, priority={correction_priority!r}"
                )
        issues_str = "\n".join(issue_lines) if issue_lines else "(no issues provided)"

        system = CORRECT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE.format(issues=issues_str)
        user = CORRECT_REQUIREMENTS_USER_TEMPLATE.format(
            job_description=state["job_description"],
            requirements=json.dumps(state["job_requirements"], indent=2),
        )

        logger.debug(
            "correct_requirements_node: applying %d correction(s) (attempt %d → %d)",
            len(issues),
            state["requirements_attempt_count"],
            state["requirements_attempt_count"] + 1,
        )
        response = llm.complete(system, [{"role": "user", "content": user}])
        raw = extract_json(response)
        data = json.loads(raw)
        requirements = [
            JobRequirement.model_validate(r).model_dump()
            for r in data.get("requirements", [])
        ]
        return {
            "job_requirements": requirements,
            "requirements_attempt_count": state["requirements_attempt_count"] + 1,
        }

    return correct_requirements_node


# ---------------------------------------------------------------------------
# Plain node: finalize_requirements (no LLM)
# ---------------------------------------------------------------------------

def finalize_requirements_node(state: GraphState) -> dict:
    """Add a warning when requirements validation exhausts max retries.

    Called only when max retries are exhausted and requirements validation is
    still failing.  The graph proceeds with the best-effort job_requirements
    and attaches the outstanding issues as a warning.
    """
    warnings: list[str] = []
    validation_result = state.get("requirements_validation_result")
    if validation_result and not state.get("requirements_validation_passed"):
        issues = validation_result.get("issues", [])
        if issues:
            issue_summary = "; ".join(
                f"{i.get('type', 'unknown')} — {i.get('label') or 'new'}: {i.get('problem', '')}"
                for i in issues
            )
            n = state.get("requirements_attempt_count", 0)
            warnings.append(
                f"Requirements validation did not fully pass after {n} attempt(s). "
                f"Outstanding issues: {issue_summary}"
            )
    logger.debug("finalize_requirements_node: %d warning(s)", len(warnings))
    return {"requirements_warnings": warnings}


# ---------------------------------------------------------------------------
# Conditional edge: route after requirements validation
# ---------------------------------------------------------------------------

def route_after_requirements_validation(state: GraphState) -> str:
    """Decide the next step after the validate_requirements node.

    Returns:
        "ok"       — validation passed, proceed to END
        "retry"    — validation failed, attempts remain, loop back to correct
        "give_up"  — validation failed, max attempts reached, proceed to finalize
    """
    if state["requirements_validation_passed"]:
        return "ok"
    if state["requirements_attempt_count"] >= MAX_REQUIREMENTS_ATTEMPTS:
        return "give_up"
    return "retry"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge *updates* into *base*, returning a new dict.

    For any key present in both dicts where both values are dicts, the merge
    recurses.  Otherwise *updates* value wins (overwrites).  *base* is not
    mutated.

    Used by retry_node to overlay partial corrections without clobbering
    already-correct sibling keys in nested sub-dicts like company_info.
    """
    result = dict(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
