"""Fusion stage — attach attribution to the primary alert (enrichment only).

Hard invariant of the whole design: the adjudicator **cannot raise or suppress**
the primary alert. Fusion only *adds* the attribution and flags a disagreement
(VLM says no-error while the detector fired) for downstream review — it never
changes whether the alert fires.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from aiops.adjudication.types import AdjudicationResult, Alert, EnrichedAlert


@runtime_checkable
class Fuser(Protocol):
    def fuse(self, alert: Alert, result: Optional[AdjudicationResult]) -> EnrichedAlert:
        ...


class EnrichmentFuser:
    """Default fuser: attaches a successful attribution; records disagreement.

    ``min_confidence`` gates only whether the attribution is *attached* as
    enrichment — it never gates the alert itself.
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        self.min_confidence = min_confidence

    def fuse(self, alert: Alert, result: Optional[AdjudicationResult]) -> EnrichedAlert:
        if result is None or result.status != "ok" or result.attribution is None:
            return EnrichedAlert(alert=alert, adjudicated=False)
        attr = result.attribution
        if attr.confidence < self.min_confidence:
            return EnrichedAlert(alert=alert, adjudicated=True, attribution=None)
        return EnrichedAlert(
            alert=alert,
            adjudicated=True,
            attribution=attr,
            disagreement=(attr.has_error is False),
        )
