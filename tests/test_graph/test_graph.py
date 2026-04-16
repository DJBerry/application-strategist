"""Integration tests for the LangGraph extraction graph.

Tests run the full compiled graph with a MockLLMSequence that returns preset
responses in order.  This exercises graph routing, state merging across nodes,
and the retry/give_up logic end-to-end.
"""

import json

import pytest

from app_strategist.graph.graph import build_extraction_graph, run_extraction
from app_strategist.graph.state import GraphState

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_JD = """\
Acme Corporation is hiring a Senior Software Engineer based in Austin, TX.
This is a hybrid role. Our mission: build tools that delight developers.
"""

SAMPLE_EXTRACTED = {
    "company_info": {
        "company_name": "Acme Corporation",
        "company_description": "A company that builds tools",
        "company_mission": "build tools that delight developers",
    },
    "job_info": {
        "title": "Senior Software Engineer",
        "seniority_level": "Senior",
        "location": "Austin, TX",
        "work_environment": "hybrid",
    },
}

SAMPLE_REQUIREMENTS = [
    {
        "label": "Python proficiency",
        "description": "3+ years of Python in a production environment",
        "priority": "minimum_requirement",
    },
    {
        "label": "Team leadership",
        "description": "Experience leading small engineering teams",
        "priority": "preferred_requirement",
    },
]

CHECK_PASS = json.dumps({"all_correct": True, "incorrect_fields": [], "ambiguous_fields": []})
CHECK_FAIL_NAME = json.dumps({
    "all_correct": False,
    "incorrect_fields": [
        {
            "field": "company_info.company_name",
            "extracted_value": "Acme Corp",
            "explanation": "The job description says 'Acme Corporation'",
        }
    ],
    "ambiguous_fields": [],
})
CORRECTED_NAME = json.dumps({"company_info": {"company_name": "Acme Corporation"}})

REQ_EXTRACT_PASS = json.dumps({"requirements": SAMPLE_REQUIREMENTS})
REQ_VALIDATE_PASS = json.dumps({"all_correct": True, "issues": []})


class MockLLMSequence:
    """Returns responses in order; raises if more calls are made than configured."""

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

    @property
    def call_count(self) -> int:
        return self._call_count


def _full_initial_state() -> dict:
    """Return a complete initial GraphState dict for graph.invoke()."""
    return {
        "job_description": SAMPLE_JD,
        "resume": "resume text",
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


def _extracted_with_wrong_name() -> dict:
    data = {k: dict(v) for k, v in SAMPLE_EXTRACTED.items()}
    data["company_info"] = dict(SAMPLE_EXTRACTED["company_info"])
    data["company_info"]["company_name"] = "Acme Corp"  # wrong
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_success_on_first_try():
    """Graph completes: extract → check(pass) → extract_requirements → validate(pass) → END."""
    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),  # extract
        CHECK_PASS,                    # check → ok
        REQ_EXTRACT_PASS,              # extract_requirements
        REQ_VALIDATE_PASS,             # validate_requirements → ok
    ])
    graph = build_extraction_graph(llm=llm)

    result = graph.invoke(_full_initial_state())

    assert result["validation_passed"] is True
    assert result["extracted_data"] == SAMPLE_EXTRACTED
    assert result["unresolved_concerns"] == []
    assert result["attempt_count"] == 1
    assert result["requirements_validation_passed"] is True
    assert len(result["job_requirements"]) == 2
    assert result["requirements_warnings"] == []
    assert llm.call_count == 4


def test_retry_then_success():
    """Metadata retry: extract → check(fail) → retry → check(pass) → req nodes → END."""
    wrong_extraction = json.dumps(_extracted_with_wrong_name())
    llm = MockLLMSequence([
        wrong_extraction,   # extract
        CHECK_FAIL_NAME,    # check → fail
        CORRECTED_NAME,     # retry
        CHECK_PASS,         # check → pass
        REQ_EXTRACT_PASS,   # extract_requirements
        REQ_VALIDATE_PASS,  # validate_requirements → ok
    ])
    graph = build_extraction_graph(llm=llm)

    result = graph.invoke(_full_initial_state())

    assert result["validation_passed"] is True
    assert result["extracted_data"]["company_info"]["company_name"] == "Acme Corporation"
    assert result["unresolved_concerns"] == []
    assert result["attempt_count"] == 2
    assert result["requirements_validation_passed"] is True
    assert llm.call_count == 6


def test_max_retries_exceeded():
    """All 3 metadata attempts fail → finalize → extract_requirements runs anyway."""
    wrong = json.dumps(_extracted_with_wrong_name())
    partial_still_wrong = json.dumps({"company_info": {"company_name": "Acme Corp"}})
    llm = MockLLMSequence([
        wrong,               # extract (attempt 1)
        CHECK_FAIL_NAME,     # check → fail
        partial_still_wrong, # retry (attempt 2)
        CHECK_FAIL_NAME,     # check → fail
        partial_still_wrong, # retry (attempt 3)
        CHECK_FAIL_NAME,     # check → fail → give_up → finalize
        REQ_EXTRACT_PASS,    # extract_requirements
        REQ_VALIDATE_PASS,   # validate_requirements → ok
    ])
    graph = build_extraction_graph(llm=llm)

    result = graph.invoke(_full_initial_state())

    assert result["validation_passed"] is False
    assert result["attempt_count"] == 3
    assert len(result["unresolved_concerns"]) > 0
    assert "company_info.company_name" in result["unresolved_concerns"][0]
    # Requirements extraction still ran
    assert result["requirements_validation_passed"] is True
    assert len(result["job_requirements"]) == 2
    assert llm.call_count == 8


def test_run_extraction_returns_full_state():
    """run_extraction() convenience wrapper returns all GraphState keys."""
    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),
        CHECK_PASS,
        REQ_EXTRACT_PASS,
        REQ_VALIDATE_PASS,
    ])

    result = run_extraction(
        job_description=SAMPLE_JD,
        resume="resume text",
        cover_letter=None,
        llm=llm,
    )

    expected_keys = {
        "job_description", "resume", "cover_letter",
        "extracted_data", "validation_passed", "validation_result",
        "attempt_count", "unresolved_concerns", "field_caveats",
        "job_requirements", "requirements_validation_passed",
        "requirements_validation_result", "requirements_attempt_count",
        "requirements_warnings",
    }
    assert expected_keys.issubset(result.keys())
    assert result["job_description"] == SAMPLE_JD
    assert result["resume"] == "resume text"
    assert result["cover_letter"] is None


def test_run_extraction_with_cover_letter():
    """Cover letter is carried in state unchanged."""
    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),
        CHECK_PASS,
        REQ_EXTRACT_PASS,
        REQ_VALIDATE_PASS,
    ])

    result = run_extraction(
        job_description=SAMPLE_JD,
        resume="resume text",
        cover_letter="Dear hiring manager...",
        llm=llm,
    )

    assert result["cover_letter"] == "Dear hiring manager..."


def test_extracted_data_has_required_structure():
    """extracted_data always has company_info and job_info sub-dicts."""
    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),
        CHECK_PASS,
        REQ_EXTRACT_PASS,
        REQ_VALIDATE_PASS,
    ])

    result = run_extraction(job_description=SAMPLE_JD, llm=llm)

    data = result["extracted_data"]
    assert "company_info" in data
    assert "job_info" in data
    assert "company_name" in data["company_info"]
    assert "work_environment" in data["job_info"]


def test_ambiguous_field_does_not_cause_retry():
    """Ambiguous field alone → check passes, no metadata retry."""
    check_with_ambiguous = json.dumps({
        "all_correct": True,
        "incorrect_fields": [],
        "ambiguous_fields": [
            {
                "field": "job_info.work_environment",
                "extracted_value": "remote",
                "explanation": "Correct for most staff, but JD notes hybrid near HQ",
            }
        ],
    })
    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),
        check_with_ambiguous,
        REQ_EXTRACT_PASS,
        REQ_VALIDATE_PASS,
    ])

    result = run_extraction(job_description=SAMPLE_JD, llm=llm)

    assert result["validation_passed"] is True
    assert result["unresolved_concerns"] == []
    assert len(result["field_caveats"]) == 1
    assert result["field_caveats"][0]["field"] == "job_info.work_environment"
    assert llm.call_count == 4


def test_requirements_retry_then_success():
    """Requirements validation fails once → correct → validate passes → END."""
    req_validate_fail = json.dumps({
        "all_correct": False,
        "issues": [
            {
                "type": "inaccurate",
                "label": "Python proficiency",
                "duplicate_of": None,
                "problem": "Description adds 5-year threshold not in JD",
                "correction": {
                    "label": "Python proficiency",
                    "description": "Experience with Python in a production environment",
                    "priority": "minimum_requirement",
                },
            }
        ],
    })
    corrected_reqs = json.dumps({
        "requirements": [
            {
                "label": "Python proficiency",
                "description": "Experience with Python in a production environment",
                "priority": "minimum_requirement",
            },
            {
                "label": "Team leadership",
                "description": "Experience leading small engineering teams",
                "priority": "preferred_requirement",
            },
        ]
    })

    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),  # extract
        CHECK_PASS,                    # check → ok
        REQ_EXTRACT_PASS,              # extract_requirements
        req_validate_fail,             # validate_requirements → retry
        corrected_reqs,                # correct_requirements
        REQ_VALIDATE_PASS,             # validate_requirements → ok
    ])
    graph = build_extraction_graph(llm=llm)

    result = graph.invoke(_full_initial_state())

    assert result["requirements_validation_passed"] is True
    assert result["requirements_warnings"] == []
    assert result["requirements_attempt_count"] == 2
    python_req = next(r for r in result["job_requirements"] if r["label"] == "Python proficiency")
    assert python_req["description"] == "Experience with Python in a production environment"
    assert llm.call_count == 6


def test_requirements_max_retries_exceeded():
    """Requirements validation fails all 3 attempts → finalize_requirements → warnings."""
    req_validate_fail = json.dumps({
        "all_correct": False,
        "issues": [
            {
                "type": "missing",
                "label": None,
                "duplicate_of": None,
                "problem": "Security clearance requirement missing",
                "correction": {
                    "label": "Security clearance",
                    "description": "Active security clearance required",
                    "priority": "minimum_requirement",
                },
            }
        ],
    })
    still_wrong = json.dumps({"requirements": SAMPLE_REQUIREMENTS})

    llm = MockLLMSequence([
        json.dumps(SAMPLE_EXTRACTED),  # extract
        CHECK_PASS,                    # check → ok
        REQ_EXTRACT_PASS,              # extract_requirements (attempt 1)
        req_validate_fail,             # validate → retry
        still_wrong,                   # correct (attempt 2)
        req_validate_fail,             # validate → retry
        still_wrong,                   # correct (attempt 3)
        req_validate_fail,             # validate → give_up → finalize_requirements
    ])
    graph = build_extraction_graph(llm=llm)

    result = graph.invoke(_full_initial_state())

    assert result["requirements_validation_passed"] is False
    assert result["requirements_attempt_count"] == 3
    assert len(result["requirements_warnings"]) == 1
    assert "did not fully pass" in result["requirements_warnings"][0]
    assert llm.call_count == 8
