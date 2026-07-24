"""The adjudication pipeline — wires the swappable stages.

    trigger -> evidence -> prompt -> backend -> parser -> (fusion)

Every stage is injected; ``build_default_pipeline`` gives a fully working default
(mock backend) so the pipeline runs end-to-end with no API/GPU. Backend errors are
caught per-event (status ``backend_error``) so one bad event never sinks a batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

from aiops.adjudication.backends import MockVLMBackend, VLMBackend
from aiops.adjudication.evidence import DefaultEvidenceBuilder, EvidenceBuilder
from aiops.adjudication.fusion import EnrichmentFuser, Fuser
from aiops.adjudication.parsing import JsonResponseParser, ResponseParser
from aiops.adjudication.prompts import PromptBuilder, ZeroShotPromptBuilder
from aiops.adjudication.triggers import ScoreThresholdTrigger, Trigger
from aiops.adjudication.types import (
    AdjudicationResult,
    Alert,
    Candidate,
    EnrichedAlert,
)


@dataclass
class AdjudicationPipeline:
    trigger: Trigger
    evidence_builder: EvidenceBuilder
    prompt_builder: PromptBuilder
    backend: VLMBackend
    parser: ResponseParser
    fuser: Fuser = None  # optional; only needed for fuse()

    def adjudicate(self, candidates: Sequence[Candidate]) -> list[AdjudicationResult]:
        """Run trigger -> evidence -> prompt -> backend -> parse for each event."""
        # Warm up the backend before any evidence building. Some backends (e.g.
        # a torch VLM) must initialize CUDA before decord decodes frames, or the
        # process segfaults on init order.
        warmup = getattr(self.backend, "load", None)
        if callable(warmup):
            warmup()
        results: list[AdjudicationResult] = []
        for cand in self.trigger.select(candidates):
            packet = self.evidence_builder.build(cand)
            prompt = self.prompt_builder.build(packet)
            t0 = time.perf_counter()
            try:
                raw = self.backend.generate(prompt)
            except Exception as exc:  # never let one event sink the batch
                results.append(AdjudicationResult(
                    event_id=cand.event_id, status="backend_error",
                    backend=getattr(self.backend, "name", "?"), error=str(exc)))
                continue
            res = self.parser.parse(raw, packet)
            res.backend = getattr(self.backend, "name", "?")
            res.latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(res)
        return results

    def fuse(self, alerts: Sequence[Alert],
             results: Sequence[AdjudicationResult]) -> list[EnrichedAlert]:
        """Attach adjudications to alerts (enrichment only, never suppresses)."""
        if self.fuser is None:
            raise ValueError("pipeline has no fuser; pass one to fuse alerts")
        by_id = {r.event_id: r for r in results}
        return [self.fuser.fuse(a, by_id.get(a.event_id)) for a in alerts]


def build_default_pipeline(
    backend: Optional[VLMBackend] = None,
    *,
    trigger: Optional[Trigger] = None,
    evidence_builder: Optional[EvidenceBuilder] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    parser: Optional[ResponseParser] = None,
    fuser: Optional[Fuser] = None,
) -> AdjudicationPipeline:
    """A working pipeline; every stage overridable. Defaults to the mock backend."""
    return AdjudicationPipeline(
        trigger=trigger or ScoreThresholdTrigger(0.5),
        evidence_builder=evidence_builder or DefaultEvidenceBuilder(),
        prompt_builder=prompt_builder or ZeroShotPromptBuilder(),
        backend=backend or MockVLMBackend(),
        parser=parser or JsonResponseParser(),
        fuser=fuser or EnrichmentFuser(),
    )
