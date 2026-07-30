"""Structured-output schema + dependency-free validation/coercion.

The schema is the contract the VLM must fill. We validate raw parsed dicts against
it and coerce them into a ``MistakeAttribution`` with safe defaults, so a slightly
malformed model response degrades gracefully instead of crashing the pipeline.
"""

from __future__ import annotations

from typing import Any

from aiops.adjudication.types import MistakeAttribution, normalize_family

# JSON-schema-style contract (also embeddable in a prompt). Kept as plain data so
# no external validator dependency is required.
ATTRIBUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["has_error", "mistake_family", "description", "confidence"],
    "properties": {
        "has_error": {"type": "boolean"},
        "mistake_family": {"type": "string",
                           "description": "one of the mistake families or a free-text family"},
        "mistake_subtype": {"type": "string"},
        "description": {"type": "string", "description": "what specifically went wrong"},
        "expected": {"type": "string", "description": "what the step called for"},
        "observed": {"type": "string", "description": "what was actually seen"},
        "violated_role": {"type": "string",
                          "description": "ingredient | tool | quantity | temperature | order | timing | technique"},
        "evidence": {"type": "string", "description": "where in the frames the evidence is"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "evidence_sufficient": {"type": "boolean",
                                "description": "false if the frames/context do not let you "
                                "identify the specific mistake (then abstain, do not guess)"},
    },
}

_TYPE_CHECKS = {
    "boolean": lambda v: isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def validate_attribution(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, errors) for a parsed attribution dict against the schema."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, ["response is not a JSON object"]
    props = ATTRIBUTION_SCHEMA["properties"]
    for key in ATTRIBUTION_SCHEMA["required"]:
        if key not in payload:
            errors.append(f"missing required field '{key}'")
    for key, value in payload.items():
        spec = props.get(key)
        if not spec:
            continue  # unknown fields tolerated
        check = _TYPE_CHECKS.get(spec["type"])
        if check and not check(value):
            errors.append(f"field '{key}' should be {spec['type']}")
        if key == "confidence" and _TYPE_CHECKS["number"](value):
            if not 0.0 <= float(value) <= 1.0:
                errors.append("confidence out of [0,1]")
    return (len(errors) == 0), errors


def coerce_attribution(payload: dict[str, Any]) -> MistakeAttribution:
    """Build a MistakeAttribution from a (possibly imperfect) dict with defaults."""
    def s(key: str) -> str:
        v = payload.get(key, "")
        return v if isinstance(v, str) else ("" if v is None else str(v))

    conf = payload.get("confidence", 0.0)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0
    has_error = payload.get("has_error", True)
    if not isinstance(has_error, bool):
        has_error = str(has_error).strip().lower() in {"true", "1", "yes"}
    es = payload.get("evidence_sufficient", True)
    if not isinstance(es, bool):
        es = str(es).strip().lower() not in {"false", "0", "no"}
    return MistakeAttribution(
        has_error=has_error,
        mistake_family=s("mistake_family") or "Other",
        description=s("description"),
        confidence=conf,
        mistake_subtype=s("mistake_subtype"),
        expected=s("expected"),
        observed=s("observed"),
        violated_role=s("violated_role"),
        evidence=s("evidence"),
        rationale=s("rationale"),
        evidence_sufficient=es,
    )


def canonical_family(attr: MistakeAttribution) -> str:
    return normalize_family(attr.mistake_family)
