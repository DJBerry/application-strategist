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

from pydantic import ValidationError

from app_strategist.llm.base import LLMProvider
from app_strategist.utils import extract_json
from app_strategist.graph.state import GraphState
from app_strategist.models.requirements import JobRequirement
from app_strategist.models.scoring import QualificationScore, QualificationScoringResult, ScoreAttempt
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
    EXTRACT_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT,
    EXTRACT_IMPLICIT_REQUIREMENTS_USER_TEMPLATE,
    VALIDATE_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT,
    VALIDATE_IMPLICIT_REQUIREMENTS_USER_TEMPLATE,
    CORRECT_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE,
    CORRECT_IMPLICIT_REQUIREMENTS_USER_TEMPLATE,
    DEDUPLICATE_REQUIREMENTS_SYSTEM_PROMPT,
    DEDUPLICATE_REQUIREMENTS_USER_TEMPLATE,
    SCORE_QUALIFICATIONS_SYSTEM_PROMPT,
    SCORE_QUALIFICATIONS_USER_TEMPLATE,
    VALIDATE_QUALIFICATION_SCORES_SYSTEM_PROMPT,
    VALIDATE_QUALIFICATION_SCORES_USER_TEMPLATE,
    RECHECK_QUALIFICATION_SCORES_SYSTEM_PROMPT_TEMPLATE,
    RECHECK_QUALIFICATION_SCORES_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_VALID_PRIORITIES = frozenset(
    {"minimum_requirement", "preferred_requirement", "nice_to_have", "ambiguous"}
)


def _validate_requirement(r: dict) -> dict | None:
    """Validate a raw requirement dict through JobRequirement.

    If the priority field holds an unrecognized value, coerces it to 'ambiguous'
    and retries validation so that one bad LLM token cannot drop the entire item.
    Returns None (with a warning) only when validation fails even after coercion.
    """
    try:
        return JobRequirement.model_validate(r).model_dump()
    except ValidationError:
        raw_priority = r.get("priority", "")
        if raw_priority not in _VALID_PRIORITIES:
            logger.warning(
                "requirement %r has unrecognized priority %r; coercing to 'ambiguous'",
                r.get("label"),
                raw_priority,
            )
            try:
                return JobRequirement.model_validate(
                    {**r, "priority": "ambiguous"}
                ).model_dump()
            except ValidationError:
                pass
        logger.warning("requirement %r is invalid after coercion; skipping", r.get("label"))
        return None


MAX_ATTEMPTS = 3
MAX_REQUIREMENTS_ATTEMPTS = 3  # 1 initial extraction + 2 correction retries
MAX_IMPLICIT_REQUIREMENTS_ATTEMPTS = 3  # 1 initial extraction + 2 correction retries
MAX_QUALIFICATION_SCORE_ATTEMPTS = 3   # 1 initial scoring + 2 rechecks

_FULLY_MET_SCORE = 4


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
            v for r in data.get("requirements", [])
            if (v := _validate_requirement(r)) is not None
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
            v for r in data.get("requirements", [])
            if (v := _validate_requirement(r)) is not None
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

# ---------------------------------------------------------------------------
# Factory: extract_implicit_requirements node
# ---------------------------------------------------------------------------

def make_extract_implicit_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that infers implicit requirements from the job description."""

    def extract_implicit_requirements_node(state: GraphState) -> dict:
        extracted = state.get("extracted_data") or {}
        company_info = json.dumps(extracted.get("company_info", {}), indent=2)
        job_info = json.dumps(extracted.get("job_info", {}), indent=2)
        explicit_requirements = json.dumps(state.get("job_requirements") or [], indent=2)
        user = EXTRACT_IMPLICIT_REQUIREMENTS_USER_TEMPLATE.format(
            job_description=state["job_description"],
            company_info=company_info,
            job_info=job_info,
            explicit_requirements=explicit_requirements,
        )
        logger.debug(
            "extract_implicit_requirements_node: calling LLM (attempt %d)",
            state["implicit_requirements_attempt_count"] + 1,
        )
        response = llm.complete(
            EXTRACT_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT,
            [{"role": "user", "content": user}],
        )
        raw = extract_json(response)
        data = json.loads(raw)
        requirements = []
        for r in data.get("requirements", []):
            validated = _validate_requirement(r)
            if validated is None:
                continue
            if not validated.get("is_implicit"):
                logger.warning(
                    "extract_implicit_requirements_node: skipping item %r — "
                    "is_implicit is False (LLM did not set it)",
                    validated.get("label"),
                )
                continue
            requirements.append(validated)
        return {
            "implicit_requirements": requirements,
            "implicit_requirements_attempt_count": state["implicit_requirements_attempt_count"] + 1,
        }

    return extract_implicit_requirements_node


# ---------------------------------------------------------------------------
# Factory: validate_implicit_requirements node
# ---------------------------------------------------------------------------

def make_validate_implicit_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that validates the inferred implicit requirements."""

    def validate_implicit_requirements_node(state: GraphState) -> dict:
        user = VALIDATE_IMPLICIT_REQUIREMENTS_USER_TEMPLATE.format(
            job_description=state["job_description"],
            requirements=json.dumps(state["implicit_requirements"], indent=2),
        )
        logger.debug(
            "validate_implicit_requirements_node: validating %d requirement(s) (attempt %d)",
            len(state["implicit_requirements"] or []),
            state["implicit_requirements_attempt_count"],
        )
        response = llm.complete(
            VALIDATE_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT,
            [{"role": "user", "content": user}],
        )
        raw = extract_json(response)
        validation_result = json.loads(raw)

        issues: list[dict] = validation_result.get("issues", [])
        passed = not bool(issues)
        if passed:
            logger.debug("validate_implicit_requirements_node: validation passed")
        else:
            types = [i.get("type") for i in issues]
            logger.debug(
                "validate_implicit_requirements_node: validation failed — issue types: %s", types
            )

        return {
            "implicit_requirements_validation_passed": passed,
            "implicit_requirements_validation_result": validation_result,
        }

    return validate_implicit_requirements_node


# ---------------------------------------------------------------------------
# Factory: correct_implicit_requirements node
# ---------------------------------------------------------------------------

def make_correct_implicit_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that applies validator corrections to implicit requirements."""

    def correct_implicit_requirements_node(state: GraphState) -> dict:
        issues: list[dict] = (
            state["implicit_requirements_validation_result"].get("issues", [])
            if state["implicit_requirements_validation_result"]
            else []
        )

        issue_lines = []
        for item in issues:
            issue_type = item.get("type", "unknown")
            label = item.get("label") or "(new requirement)"
            problem = item.get("problem", "")
            correction = item.get("correction")  # may be None for "ungrounded"

            if issue_type == "ungrounded":
                issue_lines.append(
                    f"- [ungrounded] '{label}': {problem}\n"
                    f"  Action: REMOVE this requirement entirely (no replacement)"
                )
            elif issue_type == "duplicate":
                duplicate_of = item.get("duplicate_of") or ""
                correction_label = correction.get("label", "") if correction else ""
                correction_desc = correction.get("description", "") if correction else ""
                correction_priority = correction.get("priority", "") if correction else ""
                issue_lines.append(
                    f"- [duplicate] '{label}' and '{duplicate_of}': {problem}\n"
                    f"  Merged correction: label={correction_label!r}, "
                    f"description={correction_desc!r}, priority={correction_priority!r}"
                )
            else:
                correction_label = correction.get("label", "") if correction else ""
                correction_desc = correction.get("description", "") if correction else ""
                correction_priority = correction.get("priority", "") if correction else ""
                issue_lines.append(
                    f"- [{issue_type}] '{label}': {problem}\n"
                    f"  Correction: label={correction_label!r}, "
                    f"description={correction_desc!r}, priority={correction_priority!r}"
                )
        issues_str = "\n".join(issue_lines) if issue_lines else "(no issues provided)"

        system = CORRECT_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE.format(issues=issues_str)
        user = CORRECT_IMPLICIT_REQUIREMENTS_USER_TEMPLATE.format(
            job_description=state["job_description"],
            requirements=json.dumps(state["implicit_requirements"], indent=2),
        )

        logger.debug(
            "correct_implicit_requirements_node: applying %d correction(s) (attempt %d → %d)",
            len(issues),
            state["implicit_requirements_attempt_count"],
            state["implicit_requirements_attempt_count"] + 1,
        )
        response = llm.complete(system, [{"role": "user", "content": user}])
        raw = extract_json(response)
        data = json.loads(raw)
        requirements = []
        for r in data.get("requirements", []):
            validated = _validate_requirement(r)
            if validated is None:
                continue
            if not validated.get("is_implicit"):
                logger.warning(
                    "correct_implicit_requirements_node: skipping item %r — is_implicit is False",
                    validated.get("label"),
                )
                continue
            requirements.append(validated)
        return {
            "implicit_requirements": requirements,
            "implicit_requirements_attempt_count": state["implicit_requirements_attempt_count"] + 1,
        }

    return correct_implicit_requirements_node


# ---------------------------------------------------------------------------
# Plain node: finalize_implicit_requirements (no LLM)
# ---------------------------------------------------------------------------

def finalize_implicit_requirements_node(state: GraphState) -> dict:
    """Add a warning when implicit requirements validation exhausts max retries."""
    warnings: list[str] = []
    validation_result = state.get("implicit_requirements_validation_result")
    if validation_result and not state.get("implicit_requirements_validation_passed"):
        issues = validation_result.get("issues", [])
        if issues:
            issue_summary = "; ".join(
                f"{i.get('type', 'unknown')} — {i.get('label') or 'new'}: {i.get('problem', '')}"
                for i in issues
            )
            n = state.get("implicit_requirements_attempt_count", 0)
            warnings.append(
                f"Implicit requirements validation did not fully pass after {n} attempt(s). "
                f"Outstanding issues: {issue_summary}"
            )
    logger.debug("finalize_implicit_requirements_node: %d warning(s)", len(warnings))
    return {"implicit_requirements_warnings": warnings}


# ---------------------------------------------------------------------------
# Conditional edge: route after implicit requirements validation
# ---------------------------------------------------------------------------

def route_implicit_requirements(state: GraphState) -> str:
    """Decide the next step after the validate_implicit_requirements node.

    Returns:
        "ok"       — validation passed, proceed to END
        "retry"    — validation failed, attempts remain, loop back to correct
        "give_up"  — validation failed, max attempts reached, proceed to finalize
    """
    if state["implicit_requirements_validation_passed"]:
        return "ok"
    if state["implicit_requirements_attempt_count"] >= MAX_IMPLICIT_REQUIREMENTS_ATTEMPTS:
        return "give_up"
    return "retry"


# ---------------------------------------------------------------------------
# Factory: deduplicate_requirements node
# ---------------------------------------------------------------------------

def make_deduplicate_requirements_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that drops implicit requirements that substantially overlap with explicit ones."""

    def deduplicate_requirements_node(state: GraphState) -> dict:
        explicit = state.get("job_requirements") or []
        implicit = state.get("implicit_requirements") or []

        if not explicit or not implicit:
            return {}

        explicit_text = "\n".join(
            f"- {r['label']}: {r['description']}" for r in explicit
        )
        implicit_text = "\n".join(
            f"- {r['label']}: {r['description']}" for r in implicit
        )
        user = DEDUPLICATE_REQUIREMENTS_USER_TEMPLATE.format(
            explicit_requirements=explicit_text,
            implicit_requirements=implicit_text,
        )
        logger.debug(
            "deduplicate_requirements_node: comparing %d explicit vs %d implicit requirement(s)",
            len(explicit),
            len(implicit),
        )
        response = llm.complete(
            DEDUPLICATE_REQUIREMENTS_SYSTEM_PROMPT,
            [{"role": "user", "content": user}],
        )
        try:
            data = json.loads(extract_json(response))
            remove_labels = set(data.get("remove", []))
        except Exception:
            logger.warning(
                "deduplicate_requirements_node: failed to parse LLM response; "
                "keeping all implicit requirements"
            )
            return {}

        if remove_labels:
            logger.debug(
                "deduplicate_requirements_node: removing %d implicit requirement(s): %s",
                len(remove_labels),
                remove_labels,
            )
        filtered = [r for r in implicit if r["label"] not in remove_labels]
        return {"implicit_requirements": filtered}

    return deduplicate_requirements_node


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _validate_qualification_score(q: dict) -> dict | None:
    """Validate a raw qualification score dict through QualificationScore.

    Returns None (with a warning) only when validation fails so that one bad
    item cannot silently drop from the final result.
    """
    try:
        return QualificationScore.model_validate(q).model_dump()
    except Exception:
        logger.warning("qualification score %r is invalid; skipping", q.get("label"))
        return None


# ---------------------------------------------------------------------------
# Factory: score_qualifications node
# ---------------------------------------------------------------------------

def make_score_qualifications_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that scores every qualification against the candidate documents."""

    def score_qualifications_node(state: GraphState) -> dict:
        # Build the unified pool once; idempotent on re-entry.
        if state.get("qualifications_to_score") is None:
            explicit = [
                {
                    "label": r["label"],
                    "description": r["description"],
                    "priority": r["priority"],
                    "is_implicit": r.get("is_implicit", False),
                }
                for r in (state.get("job_requirements") or [])
            ]
            implicit = [
                {
                    "label": r["label"],
                    "description": r["description"],
                    "priority": r["priority"],
                    "is_implicit": r.get("is_implicit", True),
                }
                for r in (state.get("implicit_requirements") or [])
            ]
            qualifications_to_score = explicit + implicit
        else:
            qualifications_to_score = state["qualifications_to_score"]

        resume = state.get("resume") or "(not provided)"
        cover_letter = state.get("cover_letter") or "(not provided)"
        user = SCORE_QUALIFICATIONS_USER_TEMPLATE.format(
            qualifications=json.dumps(
                [{"label": q["label"], "description": q["description"]} for q in qualifications_to_score],
                indent=2,
            ),
            resume=resume,
            cover_letter=cover_letter,
        )
        logger.debug(
            "score_qualifications_node: scoring %d qualification(s) (attempt %d)",
            len(qualifications_to_score),
            state.get("qualification_scores_attempt_count", 0) + 1,
        )
        response = llm.complete(SCORE_QUALIFICATIONS_SYSTEM_PROMPT, [{"role": "user", "content": user}])
        raw = extract_json(response)
        data = json.loads(raw)
        scored_map = {item["label"]: item for item in data.get("scores", [])}

        qualification_scores = []
        for q in qualifications_to_score:
            label = q["label"]
            if label in scored_map:
                s = scored_map[label]
                score = int(s.get("score", 0))
                evidence = s.get("evidence", [])
                attempt = {"score": score, "rejection_reason": None}
            else:
                logger.warning(
                    "score_qualifications_node: LLM did not score %r; defaulting to 0", label
                )
                score = 0
                evidence = []
                attempt = {"score": 0, "rejection_reason": "LLM did not return a score"}

            qualification_scores.append({
                "label": label,
                "description": q["description"],
                "priority": q["priority"],
                "is_implicit": q["is_implicit"],
                "evidence": evidence,
                "attempts": [attempt],
                "final_score": score,
                "unresolved": False,
            })

        return {
            "qualifications_to_score": qualifications_to_score,
            "qualification_scores": qualification_scores,
            "qualification_scores_attempt_count": state.get("qualification_scores_attempt_count", 0) + 1,
        }

    return score_qualifications_node


# ---------------------------------------------------------------------------
# Factory: validate_qualification_scores node
# ---------------------------------------------------------------------------

def make_validate_qualification_scores_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that validates qualification scores against the candidate documents."""

    def validate_qualification_scores_node(state: GraphState) -> dict:
        qualification_scores = state.get("qualification_scores") or []
        resume = state.get("resume") or "(not provided)"
        cover_letter = state.get("cover_letter") or "(not provided)"
        user = VALIDATE_QUALIFICATION_SCORES_USER_TEMPLATE.format(
            qualification_scores=json.dumps(qualification_scores, indent=2),
            resume=resume,
            cover_letter=cover_letter,
        )
        logger.debug(
            "validate_qualification_scores_node: validating %d score(s) (attempt %d)",
            len(qualification_scores),
            state.get("qualification_scores_attempt_count", 0),
        )
        response = llm.complete(
            VALIDATE_QUALIFICATION_SCORES_SYSTEM_PROMPT, [{"role": "user", "content": user}]
        )
        raw = extract_json(response)
        validation_result = json.loads(raw)

        issues: list[dict] = validation_result.get("issues", [])
        passed = not bool(issues)
        if passed:
            logger.debug("validate_qualification_scores_node: validation passed")
        else:
            flagged = [i.get("label") for i in issues]
            logger.debug(
                "validate_qualification_scores_node: validation failed — flagged labels: %s", flagged
            )

        return {
            "qualification_scores_validation_passed": passed,
            "qualification_scores_validation_result": validation_result,
        }

    return validate_qualification_scores_node


# ---------------------------------------------------------------------------
# Factory: recheck_qualification_scores node
# ---------------------------------------------------------------------------

def make_recheck_qualification_scores_node(llm: LLMProvider) -> Callable[[GraphState], dict]:
    """Return a node that re-scores only the qualifications flagged by the validator."""

    def recheck_qualification_scores_node(state: GraphState) -> dict:
        qualification_scores = list(state.get("qualification_scores") or [])
        issues: list[dict] = (
            state.get("qualification_scores_validation_result", {}).get("issues", [])
            if state.get("qualification_scores_validation_result")
            else []
        )

        label_to_idx = {q["label"]: i for i, q in enumerate(qualification_scores)}
        flagged_labels = {issue["label"] for issue in issues if issue.get("label")}

        # Copy the list so we can mutate safely.
        updated_scores = [dict(q) for q in qualification_scores]

        # Attach the validator's rejection_reason onto the last attempt of each flagged item.
        for issue in issues:
            label = issue.get("label")
            if label and label in label_to_idx:
                idx = label_to_idx[label]
                q = dict(updated_scores[idx])
                attempts = list(q.get("attempts", []))
                if attempts:
                    last = dict(attempts[-1])
                    last["rejection_reason"] = issue.get("problem", "Flagged by validator")
                    attempts = attempts[:-1] + [last]
                q["attempts"] = attempts
                updated_scores[idx] = q

        # Build the LLM payload: flagged items only, with prior score, evidence, and problem.
        flagged_items = []
        for q in updated_scores:
            if q["label"] not in flagged_labels:
                continue
            problem = next(
                (i.get("problem", "") for i in issues if i.get("label") == q["label"]), ""
            )
            flagged_items.append({
                "label": q["label"],
                "description": q["description"],
                "prior_score": q["final_score"],
                "prior_evidence": q.get("evidence", []),
                "validator_problem": problem,
            })

        # Build human-readable issues string for the system prompt template.
        issue_lines = []
        for issue in issues:
            label = issue.get("label", "unknown")
            problem = issue.get("problem", "")
            suggested = issue.get("suggested_score")
            line = f"- '{label}': {problem}"
            if suggested is not None:
                line += f" (suggested score: {suggested})"
            issue_lines.append(line)
        issues_str = "\n".join(issue_lines) if issue_lines else "(no issues provided)"

        system = RECHECK_QUALIFICATION_SCORES_SYSTEM_PROMPT_TEMPLATE.format(issues=issues_str)
        resume = state.get("resume") or "(not provided)"
        cover_letter = state.get("cover_letter") or "(not provided)"
        user = RECHECK_QUALIFICATION_SCORES_USER_TEMPLATE.format(
            flagged_qualifications=json.dumps(flagged_items, indent=2),
            resume=resume,
            cover_letter=cover_letter,
        )

        logger.debug(
            "recheck_qualification_scores_node: re-scoring %d flagged item(s) (attempt %d → %d)",
            len(flagged_items),
            state.get("qualification_scores_attempt_count", 0),
            state.get("qualification_scores_attempt_count", 0) + 1,
        )
        response = llm.complete(system, [{"role": "user", "content": user}])
        raw = extract_json(response)
        data = json.loads(raw)
        rescored_map = {item["label"]: item for item in data.get("scores", [])}

        for i, q in enumerate(updated_scores):
            if q["label"] not in flagged_labels:
                continue
            q = dict(updated_scores[i])
            if q["label"] in rescored_map:
                rescored = rescored_map[q["label"]]
                new_score = int(rescored.get("score", q["final_score"]))
                new_evidence = rescored.get("evidence", q.get("evidence", []))
                q["attempts"] = list(q.get("attempts", [])) + [
                    {"score": new_score, "rejection_reason": None}
                ]
                q["evidence"] = new_evidence
                q["final_score"] = new_score
            # else: LLM didn't return this item — rejection_reason already set above; keep prior score.
            updated_scores[i] = q

        return {
            "qualification_scores": updated_scores,
            "qualification_scores_attempt_count": state.get("qualification_scores_attempt_count", 0) + 1,
        }

    return recheck_qualification_scores_node


# ---------------------------------------------------------------------------
# Plain node: finalize_qualification_scores (no LLM)
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = (
    "minimum_requirement",
    "preferred_requirement",
    "nice_to_have",
    "ambiguous",
)


def finalize_qualification_scores_node(state: GraphState) -> dict:
    """Compute per-priority totals in Python and write the final scoring result.

    Runs on both the ok and give_up branches. On give_up, marks still-flagged
    items as unresolved and appends warnings.
    """
    qualification_scores = [dict(q) for q in (state.get("qualification_scores") or [])]
    warnings: list[str] = list(state.get("qualification_scores_warnings") or [])

    give_up = (
        not state.get("qualification_scores_validation_passed", False)
        and state.get("qualification_scores_attempt_count", 0) >= MAX_QUALIFICATION_SCORE_ATTEMPTS
    )

    if give_up:
        validation_result = state.get("qualification_scores_validation_result") or {}
        issues = validation_result.get("issues", [])
        flagged_labels = {i.get("label") for i in issues if i.get("label")}
        for i, q in enumerate(qualification_scores):
            if q["label"] in flagged_labels:
                q = dict(q)
                q["unresolved"] = True
                n = state.get("qualification_scores_attempt_count", 0)
                warnings.append(
                    f"Qualification score for '{q['label']}' could not be verified after "
                    f"{n} attempt(s)."
                )
                qualification_scores[i] = q

    totals: dict[str, float] = {}
    for priority in _PRIORITY_ORDER:
        items = [q for q in qualification_scores if q.get("priority") == priority]
        totals[priority] = (
            sum(q["final_score"] for q in items) / (len(items) * _FULLY_MET_SCORE)
            if items else 0.0
        )

    validated_qualifications = [
        v for q in qualification_scores
        if (v := _validate_qualification_score(q)) is not None
    ]
    try:
        result_model = QualificationScoringResult.model_validate({
            "qualifications": validated_qualifications,
            "totals": totals,
            "warnings": warnings,
        })
        qualification_scoring_result = result_model.model_dump()
    except Exception as exc:
        logger.warning("finalize_qualification_scores_node: model validation failed — %s", exc)
        qualification_scoring_result = {
            "qualifications": validated_qualifications,
            "totals": totals,
            "warnings": warnings,
        }

    logger.debug(
        "finalize_qualification_scores_node: %d qualification(s), totals=%s, %d warning(s)",
        len(qualification_scores),
        {k: round(v, 3) for k, v in totals.items()},
        len(warnings),
    )
    return {
        "qualification_scores": qualification_scores,
        "qualification_scoring_result": qualification_scoring_result,
        "qualification_scores_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Conditional edge: route after qualification scores validation
# ---------------------------------------------------------------------------

def route_after_qualification_scores_validation(state: GraphState) -> str:
    """Decide the next step after the validate_qualification_scores node.

    Returns:
        "ok"       — validation passed, proceed to finalize
        "retry"    — validation failed, attempts remain, loop back to recheck
        "give_up"  — validation failed, max attempts reached, proceed to finalize
    """
    if state.get("qualification_scores_validation_passed"):
        return "ok"
    if state.get("qualification_scores_attempt_count", 0) >= MAX_QUALIFICATION_SCORE_ATTEMPTS:
        return "give_up"
    return "retry"


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
