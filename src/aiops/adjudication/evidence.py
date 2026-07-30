"""Evidence stage — assembles the ``EvidencePacket`` a VLM will reason over.

Kept decoupled from pixel loading: a ``FrameProvider`` resolves an event to frame
references (paths / cache ids / base64), and a ``ProcedureContext`` supplies the
expected action + surrounding steps. Swap either without touching the pipeline.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

from aiops.adjudication.types import Candidate, EvidencePacket, FrameRef

# A frame provider maps a candidate to before/contact/effect frame references.
FrameProvider = Callable[[Candidate], list[FrameRef]]


@runtime_checkable
class ProcedureContext(Protocol):
    def expected_action(self, candidate: Candidate) -> Optional[str]:
        ...

    def surrounding_steps(self, candidate: Candidate) -> list[str]:
        ...


class NullProcedureContext:
    def expected_action(self, candidate: Candidate) -> Optional[str]:
        return candidate.step_description or None

    def surrounding_steps(self, candidate: Candidate) -> list[str]:
        return []


@runtime_checkable
class EvidenceBuilder(Protocol):
    def build(self, candidate: Candidate) -> EvidencePacket:
        ...


class DefaultEvidenceBuilder:
    """Assemble frames (via a provider) + procedure context into a packet."""

    def __init__(self, frame_provider: Optional[FrameProvider] = None,
                 context: Optional[ProcedureContext] = None) -> None:
        self.frame_provider = frame_provider or (lambda c: [])
        self.context = context or NullProcedureContext()

    def build(self, candidate: Candidate) -> EvidencePacket:
        return EvidencePacket(
            candidate=candidate,
            frames=list(self.frame_provider(candidate)),
            expected_action=self.context.expected_action(candidate),
            procedure_context=list(self.context.surrounding_steps(candidate)),
        )
