"""MCP server for resume validation tools."""

from app_strategist.mcp_server.resume_validator import (
    check_claim_in_document,
    get_audit_trail,
    log_validation_event,
    mcp,
    reset_audit_trail,
    validate_extraction_batch,
)

__all__ = [
    "mcp",
    "check_claim_in_document",
    "validate_extraction_batch",
    "log_validation_event",
    "get_audit_trail",
    "reset_audit_trail",
]
