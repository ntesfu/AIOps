"""Prompt stage — turns an ``EvidencePacket`` into a model-agnostic ``Prompt``.

Two default builders: a compact zero-shot builder, and a chain-of-thought builder
(CoT improves procedural mistake reasoning — cf. TI-PREGO). Swap or add in-context
exemplars without touching the backend.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from aiops.adjudication.schema import ATTRIBUTION_SCHEMA
from aiops.adjudication.types import EvidencePacket, Prompt

_SYSTEM = (
    "You are a procedural-error adjudicator for an egocentric assembly/cooking "
    "monitor. A separate detector has already flagged a step as likely erroneous; "
    "your job is NOT to re-decide whether an error exists, but to attribute WHAT "
    "the mistake is. Reason only from the provided frames and step context. "
    "Respond with a single JSON object matching the given schema and nothing else."
)


def _context_block(packet: EvidencePacket) -> str:
    c = packet.candidate
    lines = [
        f"Step being performed: {c.step_description!r}",
        f"Detector flagged this step (score={c.detector_score:.2f}"
        + (f", suspected category={c.detector_category!r}" if c.detector_category else "")
        + ").",
    ]
    if packet.expected_action:
        lines.append(f"Expected action (from the procedure): {packet.expected_action!r}")
    if packet.procedure_context:
        lines.append("Surrounding steps: " + " | ".join(packet.procedure_context))
    if packet.frames:
        lines.append("Evidence frames provided (roles): "
                     + ", ".join(f.role for f in packet.frames))
    return "\n".join(lines)


@runtime_checkable
class PromptBuilder(Protocol):
    def build(self, packet: EvidencePacket) -> Prompt:
        ...


class ZeroShotPromptBuilder:
    def build(self, packet: EvidencePacket) -> Prompt:
        user = (
            _context_block(packet)
            + "\n\nAttribute the mistake. Return JSON with fields: "
            + ", ".join(ATTRIBUTION_SCHEMA["properties"].keys())
            + ".\nSchema:\n" + json.dumps(ATTRIBUTION_SCHEMA)
        )
        return Prompt(system=_SYSTEM, user=user, images=list(packet.frames),
                      response_schema=ATTRIBUTION_SCHEMA, packet=packet)


class ChainOfThoughtPromptBuilder:
    def build(self, packet: EvidencePacket) -> Prompt:
        user = (
            _context_block(packet)
            + "\n\nThink step by step about (1) what the step required, (2) what the "
            "frames show, (3) the delta between them — then output the final JSON "
            "attribution ONLY, matching the schema. Put reasoning in the 'rationale' "
            "field.\nSchema:\n" + json.dumps(ATTRIBUTION_SCHEMA)
        )
        return Prompt(system=_SYSTEM, user=user, images=list(packet.frames),
                      response_schema=ATTRIBUTION_SCHEMA, packet=packet)
