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
from app_strategist.graph.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_USER_TEMPLATE,
    CHECK_SYSTEM_PROMPT,
    CHECK_USER_TEMPLATE,
    RETRY_SYSTEM_PROMPT_TEMPLATE,
    RETRY_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


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
