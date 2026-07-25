"""Data types for the event-triggered VLM mistake adjudicator.

The adjudicator is a *pluggable* pipeline: a confirmed detection event becomes a
``Candidate``, an ``EvidencePacket`` is assembled, a ``Prompt`` is built, a
``VLMBackend`` produces raw text, a parser yields a ``MistakeAttribution``, and a
fuser attaches it to the primary ``Alert`` (enrichment only — it never raises or
suppresses the alert). Every stage is swappable; these types are the contracts
that pass between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Canonical mistake families (CaptainCook4D taxonomy); kept as strings so the
# schema stays open for other domains (e.g. assembly) without a code change.
MISTAKE_FAMILIES: tuple[str, ...] = (
    "Preparation Error",
    "Measurement Error",
    "Order Error",
    "Timing Error",
    "Technique Error",
    "Temperature Error",
    "Missing Step",
    "Other",
)

# Abstention sentinel — the model may return this (or set evidence_sufficient=false)
# when the frames/context do not let it identify the specific mistake. It is NOT a
# mistake family; it is scored separately (abstention), never as a wrong family.
ABSTAIN_FAMILY = "Insufficient Evidence"

_FAMILY_KEYWORDS = {
    "prepar": "Preparation Error",
    "measure": "Measurement Error",
    "quantit": "Measurement Error",
    "order": "Order Error",
    "sequenc": "Order Error",
    "timing": "Timing Error",
    "time": "Timing Error",
    "techniqu": "Technique Error",
    "temperatur": "Temperature Error",
    "missing": "Missing Step",
    "skip": "Missing Step",
}

_ABSTAIN_KEYS = (
    "insufficient", "cannot determine", "can't determine", "undetermined",
    "uncertain", "not enough", "indeterminate", "unable to determine",
)


def normalize_family(value: str) -> str:
    """Map a free-text family to the canonical taxonomy (best effort).

    Abstention phrasings collapse to ``ABSTAIN_FAMILY`` (checked before the fuzzy
    keyword pass, so "cannot determine the timing" abstains rather than mapping to
    Timing)."""
    if not value:
        return "Other"
    low = value.strip().lower()
    for fam in MISTAKE_FAMILIES:
        if low == fam.lower():
            return fam
    if low == ABSTAIN_FAMILY.lower() or any(k in low for k in _ABSTAIN_KEYS):
        return ABSTAIN_FAMILY
    for key, fam in _FAMILY_KEYWORDS.items():
        if key in low:
            return fam
    return "Other"


@dataclass(frozen=True)
class FrameRef:
    """A reference to an evidence frame, resolved lazily by a backend.

    ``ref`` is opaque (a path, URI, or cache id); ``role`` marks its place in the
    before/contact/effect story so prompts can label it.
    """

    ref: str
    role: str = "context"  # before | contact | effect | context
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class Candidate:
    """A confirmed detection event proposed for adjudication."""

    event_id: str
    recording_id: str
    step_id: int
    step_description: str
    detector_score: float
    detector_category: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePacket:
    """Everything the VLM sees for one event."""

    candidate: Candidate
    frames: list[FrameRef] = field(default_factory=list)
    expected_action: Optional[str] = None
    procedure_context: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return self.candidate.event_id


@dataclass
class Prompt:
    """A model-agnostic prompt. Backends consume system/user/images; ``packet``
    is carried for debugging and for the mock backend."""

    system: str
    user: str
    images: list[FrameRef] = field(default_factory=list)
    response_schema: dict[str, Any] = field(default_factory=dict)
    packet: Optional[EvidencePacket] = None


@dataclass
class MistakeAttribution:
    """Structured, validated output of the adjudicator."""

    has_error: bool
    mistake_family: str
    description: str
    confidence: float
    mistake_subtype: str = ""
    expected: str = ""
    observed: str = ""
    violated_role: str = ""
    evidence: str = ""
    rationale: str = ""
    evidence_sufficient: bool = True

    @property
    def abstained(self) -> bool:
        """True when the model declined to attribute (explicit flag or sentinel)."""
        return (not self.evidence_sufficient) or (
            normalize_family(self.mistake_family) == ABSTAIN_FAMILY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_error": self.has_error,
            "mistake_family": self.mistake_family,
            "mistake_subtype": self.mistake_subtype,
            "description": self.description,
            "expected": self.expected,
            "observed": self.observed,
            "violated_role": self.violated_role,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence_sufficient": self.evidence_sufficient,
        }


@dataclass
class AdjudicationResult:
    """The outcome of adjudicating one event (attribution + provenance)."""

    event_id: str
    status: str  # "ok" | "schema_invalid" | "backend_error" | "skipped"
    attribution: Optional[MistakeAttribution] = None
    raw_response: str = ""
    backend: str = ""
    latency_ms: Optional[float] = None
    error: str = ""


@dataclass
class Alert:
    """The primary detection alert. Immutable to the adjudicator."""

    event_id: str
    recording_id: str
    step_id: int
    detector_score: float
    detector_category: Optional[str] = None


@dataclass
class EnrichedAlert:
    """An alert with optional adjudication attached. The alert itself is never
    altered — ``adjudicated`` and ``attribution`` only *add* information."""

    alert: Alert
    adjudicated: bool = False
    attribution: Optional[MistakeAttribution] = None
    disagreement: bool = False  # VLM says no-error while the detector fired
