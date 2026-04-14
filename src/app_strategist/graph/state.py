"""LangGraph state schema for the evaluation pipeline."""

from typing import TypedDict


class GraphState(TypedDict):
    """
    Shared state for the job application evaluation graph.

    Designed for incremental extension: future pipeline steps (resume analysis,
    scoring, etc.) should append new fields here rather than creating separate
    state types. All fields that are not available at graph entry should be
    initialised to None / [] / False / 0 in run_extraction().
    """

    # Raw inputs — set once at graph entry, carried through to all downstream nodes
    job_description: str
    resume: str | None
    cover_letter: str | None

    # Extraction output from the extract / retry nodes
    # Schema:
    #   {
    #     "company_info": {
    #       "company_name": str,
    #       "company_description": str,
    #       "company_mission": str
    #     },
    #     "job_info": {
    #       "title": str,
    #       "seniority_level": str,
    #       "location": str,
    #       "work_environment": "remote" | "hybrid" | "on-site" | "N/A"
    #     }
    #   }
    extracted_data: dict | None

    # Validation output from the check node
    validation_passed: bool
    # Schema:
    #   {
    #     "all_correct": bool,          # true when incorrect_fields is empty
    #     "incorrect_fields": [         # factually wrong / fabricated values
    #       {"field": str, "extracted_value": str, "explanation": str}
    #     ],
    #     "ambiguous_fields": [         # valid reading but has nuance/caveats
    #       {"field": str, "extracted_value": str, "explanation": str}
    #     ]
    #   }
    validation_result: dict | None

    # Retry tracking
    attempt_count: int       # incremented by extract_node and retry_node; max 3
    unresolved_concerns: list[str]  # populated by finalize_node when max retries exceeded

    # Ambiguous field caveats — set by check_node, accumulated across iterations
    # Each entry: {"field": str, "extracted_value": str, "explanation": str}
    # Carries nuance to downstream nodes and the renderer without blocking routing.
    field_caveats: list[dict]
