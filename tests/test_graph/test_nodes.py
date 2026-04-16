"""Unit tests for graph node functions.

Each node is tested in isolation with a MockLLM that returns preset responses.
No actual LangGraph graph execution is needed here — nodes are plain functions.
"""

import json

import pytest

from app_strategist.graph.nodes import (
    _deep_merge,
    finalize_node,
    finalize_requirements_node,
    make_check_node,
    make_correct_requirements_node,
    make_extract_node,
    make_extract_requirements_node,
    make_retry_node,
    make_validate_requirements_node,
    route_after_check,
    route_after_requirements_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_EXTRACTED = {
    "company_info": {
        "company_name": "Acme Corp",
        "company_description": "A widget manufacturer",
        "company_mission": "N/A",
    },
    "job_info": {
        "title": "Senior Engineer",
        "seniority_level": "Senior",
        "location": "Austin, TX",
        "work_environment": "hybrid",
    },
}

SAMPLE_JD = "Acme Corp is hiring a Senior Engineer in Austin, TX. Hybrid work."


class MockLLM:
    """Simple mock that returns preset responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_count = 0

    def complete(self, system_prompt: str, messages: list) -> str:
        if self._call_count >= len(self._responses):
            raise AssertionError(
                f"Unexpected LLM call #{self._call_count + 1}; "
                f"only {len(self._responses)} response(s) configured"
            )
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


def _base_state(**overrides) -> dict:
    """Return a minimal GraphState-like dict for testing."""
    state = {
        "job_description": SAMPLE_JD,
        "resume": "Some resume text",
        "cover_letter": None,
        "extracted_data": None,
        "validation_passed": False,
        "validation_result": None,
        "attempt_count": 0,
        "unresolved_concerns": [],
        "field_caveats": [],
        "job_requirements": None,
        "requirements_validation_passed": False,
        "requirements_validation_result": None,
        "requirements_attempt_count": 0,
        "requirements_warnings": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# extract_node tests
# ---------------------------------------------------------------------------

def test_extract_node_success():
    llm = MockLLM([json.dumps(SAMPLE_EXTRACTED)])
    node = make_extract_node(llm)

    result = node(_base_state())

    assert result["extracted_data"] == SAMPLE_EXTRACTED
    assert result["attempt_count"] == 1


def test_extract_node_increments_attempt_count():
    llm = MockLLM([json.dumps(SAMPLE_EXTRACTED)])
    node = make_extract_node(llm)

    result = node(_base_state(attempt_count=1))

    assert result["attempt_count"] == 2


def test_extract_node_with_markdown_fences():
    """extract_json() must strip markdown before JSON parsing."""
    wrapped = f"```json\n{json.dumps(SAMPLE_EXTRACTED)}\n```"
    llm = MockLLM([wrapped])
    node = make_extract_node(llm)

    result = node(_base_state())

    assert result["extracted_data"]["company_info"]["company_name"] == "Acme Corp"


# ---------------------------------------------------------------------------
# check_node tests
# ---------------------------------------------------------------------------

def _check_pass_response() -> str:
    return json.dumps({"all_correct": True, "incorrect_fields": [], "ambiguous_fields": []})


def _check_fail_response(incorrect_fields: list[dict]) -> str:
    return json.dumps({"all_correct": False, "incorrect_fields": incorrect_fields, "ambiguous_fields": []})


def test_check_node_passes():
    llm = MockLLM([_check_pass_response()])
    node = make_check_node(llm)

    state = _base_state(extracted_data=SAMPLE_EXTRACTED, attempt_count=1)
    result = node(state)

    assert result["validation_passed"] is True
    assert result["validation_result"]["all_correct"] is True
    assert result["validation_result"]["incorrect_fields"] == []


def test_check_node_fails():
    incorrect = [
        {
            "field": "company_info.company_name",
            "extracted_value": "Acme Corp",
            "explanation": "The job description says 'Acme Corporation', not 'Acme Corp'",
        }
    ]
    llm = MockLLM([_check_fail_response(incorrect)])
    node = make_check_node(llm)

    state = _base_state(extracted_data=SAMPLE_EXTRACTED, attempt_count=1)
    result = node(state)

    assert result["validation_passed"] is False
    assert len(result["validation_result"]["incorrect_fields"]) == 1
    assert result["validation_result"]["incorrect_fields"][0]["field"] == "company_info.company_name"


def test_check_node_with_markdown_fences():
    wrapped = f"```json\n{_check_pass_response()}\n```"
    llm = MockLLM([wrapped])
    node = make_check_node(llm)

    result = node(_base_state(extracted_data=SAMPLE_EXTRACTED, attempt_count=1))

    assert result["validation_passed"] is True


def test_check_node_returns_caveats():
    """ambiguous_fields → validation_passed=True, field_caveats populated."""
    ambiguous = [
        {
            "field": "job_info.work_environment",
            "extracted_value": "remote",
            "explanation": "Correct for most staff, but JD notes hybrid near HQ",
        }
    ]
    response = json.dumps({"all_correct": True, "incorrect_fields": [], "ambiguous_fields": ambiguous})
    llm = MockLLM([response])
    node = make_check_node(llm)

    result = node(_base_state(extracted_data=SAMPLE_EXTRACTED, attempt_count=1))

    assert result["validation_passed"] is True
    assert len(result["field_caveats"]) == 1
    assert result["field_caveats"][0]["field"] == "job_info.work_environment"
    assert "hybrid near HQ" in result["field_caveats"][0]["explanation"]


def test_check_node_accumulates_caveats_across_iterations():
    """New ambiguous_fields are appended to existing field_caveats."""
    existing_caveat = {
        "field": "company_info.company_mission",
        "extracted_value": "N/A",
        "explanation": "Mission was inferred",
    }
    new_ambiguous = [
        {
            "field": "job_info.work_environment",
            "extracted_value": "remote",
            "explanation": "Hybrid caveat applies near HQ",
        }
    ]
    response = json.dumps({"all_correct": True, "incorrect_fields": [], "ambiguous_fields": new_ambiguous})
    llm = MockLLM([response])
    node = make_check_node(llm)

    state = _base_state(
        extracted_data=SAMPLE_EXTRACTED,
        attempt_count=2,
        field_caveats=[existing_caveat],
    )
    result = node(state)

    assert len(result["field_caveats"]) == 2
    fields = [c["field"] for c in result["field_caveats"]]
    assert "company_info.company_mission" in fields
    assert "job_info.work_environment" in fields


def test_check_node_ambiguous_does_not_block_ok_route():
    """route_after_check returns 'ok' when validation_passed=True even with caveats."""
    caveat = {
        "field": "job_info.work_environment",
        "extracted_value": "remote",
        "explanation": "Hybrid near HQ",
    }
    state = _base_state(
        validation_passed=True,
        attempt_count=1,
        field_caveats=[caveat],
    )
    assert route_after_check(state) == "ok"


def test_check_node_mixed_incorrect_and_ambiguous():
    """When both lists are present, validation_passed=False and field_caveats is populated."""
    incorrect = [
        {
            "field": "company_info.company_name",
            "extracted_value": "Acme Corp",
            "explanation": "Should be 'Acme Corporation'",
        }
    ]
    ambiguous = [
        {
            "field": "job_info.work_environment",
            "extracted_value": "remote",
            "explanation": "Hybrid caveat near HQ",
        }
    ]
    response = json.dumps({"all_correct": False, "incorrect_fields": incorrect, "ambiguous_fields": ambiguous})
    llm = MockLLM([response])
    node = make_check_node(llm)

    result = node(_base_state(extracted_data=SAMPLE_EXTRACTED, attempt_count=1))

    assert result["validation_passed"] is False
    assert len(result["field_caveats"]) == 1
    assert result["field_caveats"][0]["field"] == "job_info.work_environment"


# ---------------------------------------------------------------------------
# retry_node tests
# ---------------------------------------------------------------------------

def test_retry_node_merges_partial_correction():
    """Only company_name is wrong; job_info must be preserved."""
    corrected_partial = {"company_info": {"company_name": "Acme Corporation"}}
    llm = MockLLM([json.dumps(corrected_partial)])
    node = make_retry_node(llm)

    incorrect = [
        {
            "field": "company_info.company_name",
            "extracted_value": "Acme Corp",
            "explanation": "Should be 'Acme Corporation'",
        }
    ]
    state = _base_state(
        extracted_data=SAMPLE_EXTRACTED,
        attempt_count=1,
        validation_result={"all_correct": False, "incorrect_fields": incorrect},
    )

    result = node(state)

    assert result["extracted_data"]["company_info"]["company_name"] == "Acme Corporation"
    # Other company_info fields untouched
    assert result["extracted_data"]["company_info"]["company_description"] == "A widget manufacturer"
    # job_info entirely untouched
    assert result["extracted_data"]["job_info"] == SAMPLE_EXTRACTED["job_info"]
    assert result["attempt_count"] == 2


def test_retry_node_increments_attempt_count():
    llm = MockLLM([json.dumps({"company_info": {"company_name": "Fixed"}})])
    node = make_retry_node(llm)

    incorrect = [{"field": "company_info.company_name", "extracted_value": "X", "explanation": "wrong"}]
    state = _base_state(
        extracted_data=SAMPLE_EXTRACTED,
        attempt_count=2,
        validation_result={"all_correct": False, "incorrect_fields": incorrect},
    )

    result = node(state)
    assert result["attempt_count"] == 3


def test_retry_node_handles_empty_incorrect_fields():
    """Should not crash if incorrect_fields is somehow empty."""
    llm = MockLLM([json.dumps({})])
    node = make_retry_node(llm)

    state = _base_state(
        extracted_data=SAMPLE_EXTRACTED,
        attempt_count=1,
        validation_result={"all_correct": False, "incorrect_fields": []},
    )

    result = node(state)
    # Empty partial merge — extracted_data unchanged
    assert result["extracted_data"] == SAMPLE_EXTRACTED


# ---------------------------------------------------------------------------
# finalize_node tests
# ---------------------------------------------------------------------------

def test_finalize_node_sets_concerns():
    incorrect = [
        {
            "field": "job_info.title",
            "extracted_value": "Engineer",
            "explanation": "Should be 'Senior Engineer'",
        }
    ]
    state = _base_state(
        validation_passed=False,
        validation_result={"all_correct": False, "incorrect_fields": incorrect},
    )

    result = finalize_node(state)

    assert len(result["unresolved_concerns"]) == 1
    assert "job_info.title" in result["unresolved_concerns"][0]
    assert "Engineer" in result["unresolved_concerns"][0]


def test_finalize_node_empty_when_no_incorrect_fields():
    state = _base_state(
        validation_passed=False,
        validation_result={"all_correct": False, "incorrect_fields": []},
    )

    result = finalize_node(state)

    assert result["unresolved_concerns"] == []


def test_finalize_node_empty_when_no_validation_result():
    state = _base_state(validation_passed=False, validation_result=None)
    result = finalize_node(state)
    assert result["unresolved_concerns"] == []


# ---------------------------------------------------------------------------
# route_after_check tests
# ---------------------------------------------------------------------------

def test_route_ok_when_passed():
    state = _base_state(validation_passed=True, attempt_count=1)
    assert route_after_check(state) == "ok"


def test_route_retry_when_failed_and_attempts_remain():
    state = _base_state(validation_passed=False, attempt_count=1)
    assert route_after_check(state) == "retry"


def test_route_retry_at_attempt_two():
    state = _base_state(validation_passed=False, attempt_count=2)
    assert route_after_check(state) == "retry"


def test_route_give_up_at_max_attempts():
    state = _base_state(validation_passed=False, attempt_count=3)
    assert route_after_check(state) == "give_up"


def test_route_give_up_overrides_passed_false():
    """Sanity: give_up takes priority over retry when count is at max."""
    state = _base_state(validation_passed=False, attempt_count=4)
    assert route_after_check(state) == "give_up"


# ---------------------------------------------------------------------------
# _deep_merge tests
# ---------------------------------------------------------------------------

def test_deep_merge_flat_disjoint_keys():
    base = {"a": 1, "b": 2}
    updates = {"c": 3}
    result = _deep_merge(base, updates)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_deep_merge_nested_partial():
    base = {"info": {"name": "old", "desc": "keep me"}, "other": "x"}
    updates = {"info": {"name": "new"}}
    result = _deep_merge(base, updates)
    assert result["info"]["name"] == "new"
    assert result["info"]["desc"] == "keep me"
    assert result["other"] == "x"


def test_deep_merge_overwrites_leaf():
    base = {"a": 1}
    updates = {"a": 99}
    result = _deep_merge(base, updates)
    assert result["a"] == 99


def test_deep_merge_does_not_mutate_base():
    base = {"info": {"name": "original"}}
    _deep_merge(base, {"info": {"name": "changed"}})
    assert base["info"]["name"] == "original"


def test_deep_merge_empty_updates():
    base = {"a": 1, "b": {"c": 2}}
    result = _deep_merge(base, {})
    assert result == base
    assert result is not base  # new dict


def test_deep_merge_scalar_wins_over_dict():
    """If updates has a scalar where base has a dict, scalar wins."""
    base = {"info": {"name": "x"}}
    updates = {"info": "flat string"}
    result = _deep_merge(base, updates)
    assert result["info"] == "flat string"


# ---------------------------------------------------------------------------
# Helpers for requirements tests
# ---------------------------------------------------------------------------

SAMPLE_REQUIREMENTS = [
    {"label": "Python proficiency", "description": "3+ years of Python in production", "priority": "minimum_requirement"},
    {"label": "Team leadership", "description": "Experience leading small engineering teams", "priority": "preferred_requirement"},
]

SAMPLE_REQUIREMENTS_JSON = json.dumps({"requirements": SAMPLE_REQUIREMENTS})

REQ_VALIDATE_PASS = json.dumps({"all_correct": True, "issues": []})


def _req_validate_fail(issues: list[dict]) -> str:
    return json.dumps({"all_correct": False, "issues": issues})


# ---------------------------------------------------------------------------
# make_extract_requirements_node tests
# ---------------------------------------------------------------------------

def test_extract_requirements_node_success():
    llm = MockLLM([SAMPLE_REQUIREMENTS_JSON])
    node = make_extract_requirements_node(llm)

    result = node(_base_state(extracted_data=SAMPLE_EXTRACTED))

    assert len(result["job_requirements"]) == 2
    assert result["job_requirements"][0]["label"] == "Python proficiency"
    assert result["job_requirements"][0]["priority"] == "minimum_requirement"
    assert result["requirements_attempt_count"] == 1


def test_extract_requirements_node_increments_attempt_count():
    llm = MockLLM([SAMPLE_REQUIREMENTS_JSON])
    node = make_extract_requirements_node(llm)

    result = node(_base_state(extracted_data=SAMPLE_EXTRACTED, requirements_attempt_count=1))

    assert result["requirements_attempt_count"] == 2


def test_extract_requirements_node_with_markdown_fences():
    wrapped = f"```json\n{SAMPLE_REQUIREMENTS_JSON}\n```"
    llm = MockLLM([wrapped])
    node = make_extract_requirements_node(llm)

    result = node(_base_state(extracted_data=SAMPLE_EXTRACTED))

    assert result["job_requirements"][0]["label"] == "Python proficiency"


def test_extract_requirements_node_handles_missing_extracted_data():
    """extracted_data=None falls back to empty context — node should not crash."""
    llm = MockLLM([SAMPLE_REQUIREMENTS_JSON])
    node = make_extract_requirements_node(llm)

    result = node(_base_state(extracted_data=None))

    assert isinstance(result["job_requirements"], list)


# ---------------------------------------------------------------------------
# make_validate_requirements_node tests
# ---------------------------------------------------------------------------

def test_validate_requirements_node_passes():
    llm = MockLLM([REQ_VALIDATE_PASS])
    node = make_validate_requirements_node(llm)

    state = _base_state(
        job_requirements=SAMPLE_REQUIREMENTS,
        requirements_attempt_count=1,
    )
    result = node(state)

    assert result["requirements_validation_passed"] is True
    assert result["requirements_validation_result"]["all_correct"] is True
    assert result["requirements_validation_result"]["issues"] == []


def test_validate_requirements_node_fails():
    issues = [
        {
            "type": "inaccurate",
            "label": "Python proficiency",
            "duplicate_of": None,
            "problem": "Description adds a 5-year threshold not present in JD",
            "correction": {
                "label": "Python proficiency",
                "description": "Experience with Python in a production environment",
                "priority": "minimum_requirement",
            },
        }
    ]
    llm = MockLLM([_req_validate_fail(issues)])
    node = make_validate_requirements_node(llm)

    state = _base_state(
        job_requirements=SAMPLE_REQUIREMENTS,
        requirements_attempt_count=1,
    )
    result = node(state)

    assert result["requirements_validation_passed"] is False
    assert len(result["requirements_validation_result"]["issues"]) == 1
    assert result["requirements_validation_result"]["issues"][0]["type"] == "inaccurate"


def test_validate_requirements_node_with_markdown_fences():
    wrapped = f"```json\n{REQ_VALIDATE_PASS}\n```"
    llm = MockLLM([wrapped])
    node = make_validate_requirements_node(llm)

    result = node(_base_state(job_requirements=SAMPLE_REQUIREMENTS, requirements_attempt_count=1))

    assert result["requirements_validation_passed"] is True


# ---------------------------------------------------------------------------
# make_correct_requirements_node tests
# ---------------------------------------------------------------------------

CORRECTED_REQUIREMENTS = [
    {"label": "Python proficiency", "description": "Experience with Python in a production environment", "priority": "minimum_requirement"},
    {"label": "Team leadership", "description": "Experience leading small engineering teams", "priority": "preferred_requirement"},
]


def test_correct_requirements_node_returns_updated_list():
    corrected_json = json.dumps({"requirements": CORRECTED_REQUIREMENTS})
    llm = MockLLM([corrected_json])
    node = make_correct_requirements_node(llm)

    issues = [
        {
            "type": "inaccurate",
            "label": "Python proficiency",
            "duplicate_of": None,
            "problem": "Adds 5-year threshold",
            "correction": {
                "label": "Python proficiency",
                "description": "Experience with Python in a production environment",
                "priority": "minimum_requirement",
            },
        }
    ]
    state = _base_state(
        job_requirements=SAMPLE_REQUIREMENTS,
        requirements_attempt_count=1,
        requirements_validation_result={"all_correct": False, "issues": issues},
    )

    result = node(state)

    assert result["job_requirements"][0]["description"] == "Experience with Python in a production environment"
    assert result["requirements_attempt_count"] == 2


def test_correct_requirements_node_increments_attempt_count():
    corrected_json = json.dumps({"requirements": CORRECTED_REQUIREMENTS})
    llm = MockLLM([corrected_json])
    node = make_correct_requirements_node(llm)

    state = _base_state(
        job_requirements=SAMPLE_REQUIREMENTS,
        requirements_attempt_count=2,
        requirements_validation_result={"all_correct": False, "issues": [
            {"type": "wrong_priority", "label": "Team leadership", "duplicate_of": None,
             "problem": "Should be minimum", "correction": {"label": "Team leadership",
             "description": "...", "priority": "minimum_requirement"}}
        ]},
    )

    result = node(state)
    assert result["requirements_attempt_count"] == 3


def test_correct_requirements_node_handles_empty_issues():
    """Should not crash if issues list is empty."""
    corrected_json = json.dumps({"requirements": SAMPLE_REQUIREMENTS})
    llm = MockLLM([corrected_json])
    node = make_correct_requirements_node(llm)

    state = _base_state(
        job_requirements=SAMPLE_REQUIREMENTS,
        requirements_attempt_count=1,
        requirements_validation_result={"all_correct": False, "issues": []},
    )

    result = node(state)
    assert isinstance(result["job_requirements"], list)


# ---------------------------------------------------------------------------
# finalize_requirements_node tests
# ---------------------------------------------------------------------------

def test_finalize_requirements_node_sets_warning():
    issues = [
        {"type": "missing", "label": None, "problem": "Security clearance requirement missing",
         "correction": {"label": "Security clearance", "description": "...", "priority": "minimum_requirement"}}
    ]
    state = _base_state(
        requirements_validation_passed=False,
        requirements_attempt_count=3,
        requirements_validation_result={"all_correct": False, "issues": issues},
    )

    result = finalize_requirements_node(state)

    assert len(result["requirements_warnings"]) == 1
    assert "did not fully pass" in result["requirements_warnings"][0]
    assert "missing" in result["requirements_warnings"][0]


def test_finalize_requirements_node_empty_when_no_issues():
    state = _base_state(
        requirements_validation_passed=False,
        requirements_validation_result={"all_correct": False, "issues": []},
    )

    result = finalize_requirements_node(state)

    assert result["requirements_warnings"] == []


def test_finalize_requirements_node_empty_when_no_validation_result():
    state = _base_state(requirements_validation_passed=False, requirements_validation_result=None)
    result = finalize_requirements_node(state)
    assert result["requirements_warnings"] == []


# ---------------------------------------------------------------------------
# route_after_requirements_validation tests
# ---------------------------------------------------------------------------

def test_req_route_ok_when_passed():
    state = _base_state(requirements_validation_passed=True, requirements_attempt_count=1)
    assert route_after_requirements_validation(state) == "ok"


def test_req_route_retry_when_failed_and_attempts_remain():
    state = _base_state(requirements_validation_passed=False, requirements_attempt_count=1)
    assert route_after_requirements_validation(state) == "retry"


def test_req_route_retry_at_attempt_two():
    state = _base_state(requirements_validation_passed=False, requirements_attempt_count=2)
    assert route_after_requirements_validation(state) == "retry"


def test_req_route_give_up_at_max_attempts():
    state = _base_state(requirements_validation_passed=False, requirements_attempt_count=3)
    assert route_after_requirements_validation(state) == "give_up"


def test_req_route_ok_takes_priority_over_attempts():
    """validation_passed=True always wins regardless of attempt count."""
    state = _base_state(requirements_validation_passed=True, requirements_attempt_count=4)
    assert route_after_requirements_validation(state) == "ok"
