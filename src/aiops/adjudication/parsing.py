"""Parser stage — raw VLM text -> validated ``MistakeAttribution``.

Tolerant by design: strips code fences and prose, extracts the first JSON object,
validates against the schema, and coerces with safe defaults. On failure returns an
``AdjudicationResult`` with status ``schema_invalid`` (never raises), so a bad model
response degrades gracefully.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from aiops.adjudication.schema import coerce_attribution, validate_attribution
from aiops.adjudication.types import AdjudicationResult, EvidencePacket


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} block in text, tolerating fences/prose."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


@runtime_checkable
class ResponseParser(Protocol):
    def parse(self, raw: str, packet: EvidencePacket) -> AdjudicationResult:
        ...


class JsonResponseParser:
    """Extract + validate + coerce a JSON attribution from raw model text."""

    def __init__(self, strict: bool = False) -> None:
        # strict=True rejects on any schema error; strict=False coerces best-effort.
        self.strict = strict

    def parse(self, raw: str, packet: EvidencePacket) -> AdjudicationResult:
        blob = _extract_json_object(raw or "")
        if blob is None:
            return AdjudicationResult(event_id=packet.event_id, status="schema_invalid",
                                      raw_response=raw, error="no JSON object found")
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError as exc:
            return AdjudicationResult(event_id=packet.event_id, status="schema_invalid",
                                      raw_response=raw, error=f"json decode: {exc}")
        ok, errors = validate_attribution(payload)
        if not ok and self.strict:
            return AdjudicationResult(event_id=packet.event_id, status="schema_invalid",
                                      raw_response=raw, error="; ".join(errors))
        attribution = coerce_attribution(payload)
        return AdjudicationResult(event_id=packet.event_id, status="ok",
                                  attribution=attribution, raw_response=raw,
                                  error="" if ok else "coerced: " + "; ".join(errors))
