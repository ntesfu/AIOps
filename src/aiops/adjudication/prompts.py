"""Prompt stage - turns an ``EvidencePacket`` into a model-agnostic ``Prompt``.

Two default builders: a compact zero-shot builder, and a chain-of-thought builder
(CoT improves procedural mistake reasoning - cf. TI-PREGO). The pieces that shape
*which family the model picks* - one-line family definitions, a Missing-Step
disambiguation rule, and a few in-context exemplars - are module-level constants
injected into both builders, so they can be tuned or swapped without touching the
backend or the schema.

The exemplars are hand-authored generic scenarios (not drawn from any evaluation
split) so they carry no test-set leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from aiops.adjudication.schema import ATTRIBUTION_SCHEMA
from aiops.adjudication.types import MISTAKE_FAMILIES, EvidencePacket, Prompt

_FAMILIES_LINE = (
    "mistake_family MUST be exactly one of: " + ", ".join(MISTAKE_FAMILIES) + "."
)

# One-line definition per canonical family. The wording is chosen to force the
# key distinction the model kept getting wrong: an action performed *badly* is not
# a Missing Step - only an action that never happened at all is.
FAMILY_DEFINITIONS: dict[str, str] = {
    "Preparation Error": "an ingredient/tool was performed on but left in the wrong "
    "prepared state, or the wrong item was used (e.g. unpeeled when peeling was "
    "required, unwashed produce, wrong utensil).",
    "Measurement Error": "the action happened but with the wrong amount/quantity "
    "(too much or too little of an ingredient, wrong count/size).",
    "Order Error": "the step was done, but out of the correct sequence relative to "
    "other steps (before/after something it should not have been).",
    "Timing Error": "the action happened but for the wrong duration (too long or too "
    "short - under/over-cooked, mixed too briefly, waited too little/much).",
    "Technique Error": "the action happened but was executed by the wrong method or "
    "manner (wrong cutting style, improper grip/motion, wrong surface/tool use).",
    "Temperature Error": "the action happened but at the wrong temperature/heat "
    "setting (heat too high/low, wrong appliance setting).",
    "Missing Step": "the required action was NOT performed at all - it was entirely "
    "omitted or skipped. Use ONLY when the action never occurs in the frames.",
    "Other": "a genuine mistake that does not fit any family above.",
}

# The decision rule that fixes the observed Missing-Step over-prediction: the model
# was labelling any unmet goal as a Missing Step. First separate omitted-vs-attempted.
_DISAMBIGUATION = (
    "DECISION RULE (apply in order):\n"
    "1. Did the required action happen at all in the frames? If it NEVER happens "
    "(entirely skipped), and only then, the family is 'Missing Step'.\n"
    "2. If the action DID happen but the result is wrong, do NOT say 'Missing Step'. "
    "Instead pick the family that describes HOW it was wrong: wrong amount -> "
    "Measurement Error; wrong duration -> Timing Error; wrong heat -> Temperature "
    "Error; wrong method/manner -> Technique Error; wrong prepared state or wrong "
    "item -> Preparation Error; wrong sequence -> Order Error.\n"
    "A step whose goal was not achieved because it was done badly is a technique/"
    "measurement/timing/temperature/preparation failure, NOT a missing step."
)


@dataclass(frozen=True)
class FewShotExemplar:
    """A compact, hand-authored adjudication example (no eval-set provenance)."""

    step: str
    observation: str
    family: str
    why: str


# Generic exemplars that drill the omitted-vs-attempted distinction. Two of the
# three show an action that *happened but was wrong* being labelled by its manner
# (Technique / Measurement), and one shows a genuine Missing Step, so the model
# sees the boundary from both sides.
DEFAULT_FEWSHOT: tuple[FewShotExemplar, ...] = (
    FewShotExemplar(
        step="Dice the onion.",
        observation="The onion is cut, but into large uneven chunks rather than a "
        "fine dice.",
        family="Technique Error",
        why="The cutting action WAS performed, so it is not a Missing Step; it was "
        "the method (chunk vs dice) that was wrong.",
    ),
    FewShotExemplar(
        step="Add one teaspoon of salt to the sauce.",
        observation="The cook pours salt straight from the box, roughly a "
        "tablespoon.",
        family="Measurement Error",
        why="Salt WAS added, so not a Missing Step; the amount was wrong.",
    ),
    FewShotExemplar(
        step="Whisk the eggs before adding them to the pan.",
        observation="The cook cracks the eggs directly into the pan; no whisking "
        "bowl or whisking motion ever appears.",
        family="Missing Step",
        why="The whisking action never happens at all - it is truly omitted.",
    ),
)


def _definitions_block(families: Sequence[str]) -> str:
    lines = ["Mistake family definitions:"]
    for fam in families:
        d = FAMILY_DEFINITIONS.get(fam)
        if d:
            lines.append(f"- {fam}: {d}")
    return "\n".join(lines)


def _fewshot_block(exemplars: Sequence[FewShotExemplar]) -> str:
    if not exemplars:
        return ""
    lines = ["Examples (study the family boundary, do not copy verbatim):"]
    for i, ex in enumerate(exemplars, 1):
        lines.append(
            f"{i}. Step: {ex.step}\n"
            f"   Observed: {ex.observation}\n"
            f"   -> mistake_family: {ex.family} - {ex.why}"
        )
    return "\n".join(lines)


def _guidance(
    families: Sequence[str],
    exemplars: Sequence[FewShotExemplar],
    disambiguation: str,
) -> str:
    parts = [_definitions_block(families), disambiguation]
    fs = _fewshot_block(exemplars)
    if fs:
        parts.append(fs)
    return "\n\n".join(parts)


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


class _GuidedPromptBuilder:
    """Shared base: holds the injectable guidance (definitions + rule + exemplars)."""

    def __init__(
        self,
        *,
        families: Sequence[str] = MISTAKE_FAMILIES,
        few_shot: Optional[Sequence[FewShotExemplar]] = None,
        disambiguation: str = _DISAMBIGUATION,
    ) -> None:
        self.families = tuple(families)
        self.few_shot = tuple(DEFAULT_FEWSHOT if few_shot is None else few_shot)
        self.disambiguation = disambiguation

    def _guidance(self) -> str:
        return _guidance(self.families, self.few_shot, self.disambiguation)


class ZeroShotPromptBuilder(_GuidedPromptBuilder):
    def build(self, packet: EvidencePacket) -> Prompt:
        user = (
            _context_block(packet)
            + "\n\n" + self._guidance()
            + "\n\nAttribute the mistake. " + _FAMILIES_LINE
            + " Return JSON with fields: "
            + ", ".join(ATTRIBUTION_SCHEMA["properties"].keys())
            + ".\nSchema:\n" + json.dumps(ATTRIBUTION_SCHEMA)
        )
        return Prompt(system=_SYSTEM, user=user, images=list(packet.frames),
                      response_schema=ATTRIBUTION_SCHEMA, packet=packet)


class ChainOfThoughtPromptBuilder(_GuidedPromptBuilder):
    def build(self, packet: EvidencePacket) -> Prompt:
        user = (
            _context_block(packet)
            + "\n\n" + self._guidance()
            + "\n\nThink step by step about (1) what the step required, (2) whether "
            "the required action happened at all in the frames, (3) if it happened, "
            "how the result differs from what was required - then output the final "
            "JSON attribution ONLY, matching the schema. Put reasoning in the "
            "'rationale' field. " + _FAMILIES_LINE
            + "\nSchema:\n" + json.dumps(ATTRIBUTION_SCHEMA)
        )
        return Prompt(system=_SYSTEM, user=user, images=list(packet.frames),
                      response_schema=ATTRIBUTION_SCHEMA, packet=packet)
