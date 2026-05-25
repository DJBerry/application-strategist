"""Graph construction for the LangGraph-based evaluation pipeline.

build_extraction_graph() — assembles and compiles the StateGraph.
run_extraction()         — public entry point: initialises state and invokes graph.

Graph flow
----------
START → extract → check → [conditional]
                            ├─ "ok"      → extract_requirements
                            ├─ "retry"   → retry → check (loop)
                            └─ "give_up" → finalize → extract_requirements

extract_requirements → validate_requirements → [conditional]
                            ├─ "ok"      → extract_implicit_requirements
                            ├─ "retry"   → correct_requirements → validate_requirements (loop)
                            └─ "give_up" → finalize_requirements → extract_implicit_requirements

extract_implicit_requirements → validate_implicit_requirements → [conditional]
                            ├─ "ok"      → deduplicate_requirements
                            ├─ "retry"   → correct_implicit_requirements → validate_implicit_requirements (loop)
                            └─ "give_up" → finalize_implicit_requirements → deduplicate_requirements

deduplicate_requirements
    → score_qualifications
    → validate_qualification_scores → [conditional]
            ├─ "ok"      → finalize_qualification_scores → END
            ├─ "retry"   → recheck_qualification_scores → validate_qualification_scores (loop)
            └─ "give_up" → finalize_qualification_scores → END
"""

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from app_strategist.config import get_llm_provider
from app_strategist.graph.nodes import (
    finalize_node,
    finalize_requirements_node,
    finalize_implicit_requirements_node,
    finalize_qualification_scores_node,
    make_check_node,
    make_correct_requirements_node,
    make_correct_implicit_requirements_node,
    make_deduplicate_requirements_node,
    make_extract_node,
    make_extract_requirements_node,
    make_extract_implicit_requirements_node,
    make_retry_node,
    make_score_qualifications_node,
    make_validate_requirements_node,
    make_validate_implicit_requirements_node,
    make_validate_qualification_scores_node,
    make_recheck_qualification_scores_node,
    route_after_check,
    route_after_requirements_validation,
    route_implicit_requirements,
    route_after_qualification_scores_validation,
)
from app_strategist.graph.state import GraphState

if TYPE_CHECKING:
    from app_strategist.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def build_extraction_graph(llm: "LLMProvider | None" = None):
    """Assemble and compile the job description extraction graph.

    Args:
        llm: LLMProvider instance to use.  Defaults to the provider selected
             by the LLM_PROVIDER environment variable.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    _llm = llm or get_llm_provider()

    graph = StateGraph(GraphState)

    # --- metadata extraction nodes ---
    graph.add_node("extract", make_extract_node(_llm))
    graph.add_node("check", make_check_node(_llm))
    graph.add_node("retry", make_retry_node(_llm))
    graph.add_node("finalize", finalize_node)

    # --- requirements extraction nodes ---
    graph.add_node("extract_requirements", make_extract_requirements_node(_llm))
    graph.add_node("validate_requirements", make_validate_requirements_node(_llm))
    graph.add_node("correct_requirements", make_correct_requirements_node(_llm))
    graph.add_node("finalize_requirements", finalize_requirements_node)

    # --- implicit requirements extraction nodes ---
    graph.add_node("extract_implicit_requirements", make_extract_implicit_requirements_node(_llm))
    graph.add_node("validate_implicit_requirements", make_validate_implicit_requirements_node(_llm))
    graph.add_node("correct_implicit_requirements", make_correct_implicit_requirements_node(_llm))
    graph.add_node("finalize_implicit_requirements", finalize_implicit_requirements_node)

    # --- cross-list deduplication node ---
    graph.add_node("deduplicate_requirements", make_deduplicate_requirements_node(_llm))

    # --- qualification scoring nodes ---
    graph.add_node("score_qualifications", make_score_qualifications_node(_llm))
    graph.add_node("validate_qualification_scores", make_validate_qualification_scores_node(_llm))
    graph.add_node("recheck_qualification_scores", make_recheck_qualification_scores_node(_llm))
    graph.add_node("finalize_qualification_scores", finalize_qualification_scores_node)

    # --- metadata extraction loop ---
    graph.set_entry_point("extract")
    graph.add_edge("extract", "check")
    graph.add_conditional_edges(
        "check",
        route_after_check,
        {
            "ok": "extract_requirements",
            "retry": "retry",
            "give_up": "finalize",
        },
    )
    graph.add_edge("retry", "check")
    graph.add_edge("finalize", "extract_requirements")

    # --- requirements extraction loop ---
    graph.add_edge("extract_requirements", "validate_requirements")
    graph.add_conditional_edges(
        "validate_requirements",
        route_after_requirements_validation,
        {
            "ok": "extract_implicit_requirements",
            "retry": "correct_requirements",
            "give_up": "finalize_requirements",
        },
    )
    graph.add_edge("correct_requirements", "validate_requirements")
    graph.add_edge("finalize_requirements", "extract_implicit_requirements")

    # --- implicit requirements extraction loop ---
    graph.add_edge("extract_implicit_requirements", "validate_implicit_requirements")
    graph.add_conditional_edges(
        "validate_implicit_requirements",
        route_implicit_requirements,
        {
            "ok": "deduplicate_requirements",
            "retry": "correct_implicit_requirements",
            "give_up": "finalize_implicit_requirements",
        },
    )
    graph.add_edge("correct_implicit_requirements", "validate_implicit_requirements")
    graph.add_edge("finalize_implicit_requirements", "deduplicate_requirements")

    # --- qualification scoring loop ---
    graph.add_edge("deduplicate_requirements", "score_qualifications")
    graph.add_edge("score_qualifications", "validate_qualification_scores")
    graph.add_conditional_edges(
        "validate_qualification_scores",
        route_after_qualification_scores_validation,
        {
            "ok": "finalize_qualification_scores",
            "retry": "recheck_qualification_scores",
            "give_up": "finalize_qualification_scores",
        },
    )
    graph.add_edge("recheck_qualification_scores", "validate_qualification_scores")
    graph.add_edge("finalize_qualification_scores", END)

    return graph.compile()


def run_extraction(
    job_description: str,
    resume: str | None = None,
    cover_letter: str | None = None,
    llm: "LLMProvider | None" = None,
) -> dict:
    """Run the extraction graph and return the final state.

    Args:
        job_description: Full text of the job description.
        resume: Full text of the resume (carried in state for downstream nodes).
        cover_letter: Full text of the cover letter, or None.
        llm: LLMProvider to use.  Defaults to the configured provider.

    Returns:
        The final GraphState dict, including extracted_data, job_requirements,
        implicit_requirements, and any unresolved_concerns / requirements_warnings /
        implicit_requirements_warnings if max retries were exhausted.
    """
    compiled = build_extraction_graph(llm=llm)

    initial_state: GraphState = {
        "job_description": job_description,
        "resume": resume,
        "cover_letter": cover_letter,
        # metadata extraction
        "extracted_data": None,
        "validation_passed": False,
        "validation_result": None,
        "attempt_count": 0,
        "unresolved_concerns": [],
        "field_caveats": [],
        # requirements extraction
        "job_requirements": None,
        "requirements_validation_passed": False,
        "requirements_validation_result": None,
        "requirements_attempt_count": 0,
        "requirements_warnings": [],
        # implicit requirements extraction
        "implicit_requirements": None,
        "implicit_requirements_validation_passed": False,
        "implicit_requirements_validation_result": None,
        "implicit_requirements_attempt_count": 0,
        "implicit_requirements_warnings": [],
        # qualification scoring
        "qualifications_to_score": None,
        "qualification_scores": None,
        "qualification_scores_validation_passed": False,
        "qualification_scores_validation_result": None,
        "qualification_scores_attempt_count": 0,
        "qualification_scores_warnings": [],
        "qualification_scoring_result": None,
    }

    logger.debug("run_extraction: starting graph")
    result = compiled.invoke(initial_state)
    logger.debug(
        "run_extraction: finished — validation_passed=%s, attempts=%d, concerns=%d, "
        "requirements_validation_passed=%s, req_attempts=%d, req_warnings=%d, "
        "implicit_requirements_validation_passed=%s, impl_attempts=%d, impl_warnings=%d, "
        "qualification_scores_validation_passed=%s, qual_attempts=%d, qual_warnings=%d",
        result.get("validation_passed"),
        result.get("attempt_count", 0),
        len(result.get("unresolved_concerns", [])),
        result.get("requirements_validation_passed"),
        result.get("requirements_attempt_count", 0),
        len(result.get("requirements_warnings", [])),
        result.get("implicit_requirements_validation_passed"),
        result.get("implicit_requirements_attempt_count", 0),
        len(result.get("implicit_requirements_warnings", [])),
        result.get("qualification_scores_validation_passed"),
        result.get("qualification_scores_attempt_count", 0),
        len(result.get("qualification_scores_warnings", [])),
    )
    return result
