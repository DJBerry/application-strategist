"""Graph construction for the LangGraph-based evaluation pipeline.

build_extraction_graph() — assembles and compiles the StateGraph.
run_extraction()         — public entry point: initialises state and invokes graph.

Graph flow
----------
START → extract → check → [conditional]
                            ├─ "ok"       → END
                            ├─ "retry"    → retry → check (loop)
                            └─ "give_up"  → finalize → END
"""

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from app_strategist.config import get_llm_provider
from app_strategist.graph.nodes import (
    finalize_node,
    make_check_node,
    make_extract_node,
    make_retry_node,
    route_after_check,
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

    graph.add_node("extract", make_extract_node(_llm))
    graph.add_node("check", make_check_node(_llm))
    graph.add_node("retry", make_retry_node(_llm))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "check")
    graph.add_conditional_edges(
        "check",
        route_after_check,
        {
            "ok": END,
            "retry": "retry",
            "give_up": "finalize",
        },
    )
    graph.add_edge("retry", "check")
    graph.add_edge("finalize", END)

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
        The final GraphState dict, including extracted_data and any
        unresolved_concerns if max retries were exhausted.
    """
    compiled = build_extraction_graph(llm=llm)

    initial_state: GraphState = {
        "job_description": job_description,
        "resume": resume,
        "cover_letter": cover_letter,
        "extracted_data": None,
        "validation_passed": False,
        "validation_result": None,
        "attempt_count": 0,
        "unresolved_concerns": [],
    }

    logger.debug("run_extraction: starting graph")
    result = compiled.invoke(initial_state)
    logger.debug(
        "run_extraction: finished — validation_passed=%s, attempts=%d, concerns=%d",
        result.get("validation_passed"),
        result.get("attempt_count", 0),
        len(result.get("unresolved_concerns", [])),
    )
    return result
