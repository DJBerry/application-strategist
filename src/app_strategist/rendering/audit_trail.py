"""Audit trail formatting for validation log display."""

from typing import Any


def format_audit_trail(trail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw audit trail to display-friendly list of events.

    Returns list of {icon, label, color, detail} objects.
    detail is expandable/collapsible content.
    """
    EVENT_MAP = {
        "extraction_produced": ("\u2699\ufe0f", "Extraction produced", "blue", None),
        "validation_passed": ("\u2705", "Validation passed", "green", None),
        "validation_failed": ("\U0001f6a8", "Validation failed", "red", "failed_claims"),
        "recall_triggered": ("\u27f4", "Recall triggered", "orange", "corrections"),
        "retry_exhausted": ("\u26a0\ufe0f", "Retry exhausted", "yellow", "unresolvable_claims"),
        "final_output_accepted": ("\u2705", "Final output accepted", "green", None),
    }

    formatted: list[dict[str, Any]] = []
    for entry in trail:
        event_type = entry.get("event_type", "")
        attempt = entry.get("attempt_number", 0)
        details = entry.get("details", {})

        icon, label, color, detail_key = EVENT_MAP.get(event_type, ("\u2022", event_type or "event", "white", None))

        detail = None
        if detail_key and detail_key in details:
            val = details[detail_key]
            if detail_key == "failed_claims":
                lines = []
                for fc in (val if isinstance(val, list) else []):
                    c = fc.get("claim", "")
                    lines.append(f"  \u2022 {c[:80]}..." if len(c) > 80 else f"  \u2022 {c}")
                detail = "\n".join(lines)
            elif detail_key == "corrections":
                detail = "\n".join(
                    f"  \u2022 {c}" for c in (val if isinstance(val, list) else [str(val)])
                )
            elif detail_key == "unresolvable_claims":
                detail = "\n".join(
                    f"  \u2022 {uc.get('claim', '')}: {uc.get('correction_instruction', '')}"
                    for uc in (val if isinstance(val, list) else [])
                )
            else:
                detail = str(val)

        formatted.append({
            "icon": icon,
            "label": f"{label} (attempt {attempt})",
            "color": color,
            "detail": detail,
        })

    return formatted
