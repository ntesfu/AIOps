"""Runnable demo of the pluggable VLM mistake adjudicator (offline, mock backend).

    PYTHONPATH=src python scripts/demo_adjudication.py

Shows the full flow — trigger -> evidence -> prompt -> backend -> parse -> fuse ->
evaluate — with the mock backend, and prints how to swap in a real VLM. No API/GPU
needed. This is the entry point to iterate each stage independently.
"""

from __future__ import annotations

import json

from aiops.adjudication import (
    Alert,
    Candidate,
    FrameRef,
    ScoreThresholdTrigger,
    build_default_pipeline,
    evaluate_attributions,
)
from aiops.adjudication.evidence import DefaultEvidenceBuilder
from aiops.adjudication.prompts import ChainOfThoughtPromptBuilder

# In production these come from the detector + tracker; here, a few CaptainCook4D-like
# flagged step segments. detector_category is the detector's *guess* (not ground truth).
CANDIDATES = [
    Candidate("1_10#4", "1_10", 4, "Microwave the ramekin cup uncovered on high for 30 seconds",
              detector_score=0.88, detector_category="Timing Error"),
    Candidate("1_10#8", "1_10", 8, "Add 1 tsp of salt to the mixture",
              detector_score=0.71, detector_category="Measurement Error"),
    Candidate("1_10#1", "1_10", 1, "Pour 1 egg into the ramekin cup",
              detector_score=0.34, detector_category="Preparation Error"),  # below threshold
]

# Ground truth (CaptainCook4D ships tag + free-text description per error).
GROUND_TRUTH = {
    "1_10#4": {"family": "Timing Error", "description": "microwaved for 60 seconds instead of 30"},
    "1_10#8": {"family": "Measurement Error", "description": "added a tablespoon of salt, not a teaspoon"},
}


def frame_provider(cand: Candidate) -> list[FrameRef]:
    # Real builder resolves before/contact/effect frames (paths or base64). Stubbed.
    return [FrameRef(f"{cand.recording_id}@{cand.step_id}:{role}", role)
            for role in ("before", "contact", "effect")]


def main() -> None:
    pipe = build_default_pipeline(
        trigger=ScoreThresholdTrigger(0.5),          # only adjudicate flagged events
        evidence_builder=DefaultEvidenceBuilder(frame_provider=frame_provider),
        prompt_builder=ChainOfThoughtPromptBuilder(),
        # backend defaults to MockVLMBackend(); swap: ClaudeVLMBackend() / QwenVLBackend()
    )

    results = pipe.adjudicate(CANDIDATES)
    print(f"=== adjudicated {len(results)}/{len(CANDIDATES)} events (threshold 0.5) ===")
    for r in results:
        a = r.attribution
        print(f"\n[{r.event_id}] status={r.status} backend={r.backend} "
              f"latency={r.latency_ms:.1f}ms")
        if a:
            print("  " + json.dumps(a.to_dict()))

    # Fusion — enrichment only, never suppresses an alert.
    alerts = [Alert(c.event_id, c.recording_id, c.step_id, c.detector_score, c.detector_category)
              for c in CANDIDATES if c.detector_score >= 0.5]
    enriched = pipe.fuse(alerts, results)
    print("\n=== fused alerts (alert always survives) ===")
    for e in enriched:
        fam = e.attribution.mistake_family if e.attribution else "-"
        print(f"  {e.alert.event_id}: alert kept, adjudicated={e.adjudicated}, "
              f"attributed_family={fam}, disagreement={e.disagreement}")

    scores = evaluate_attributions(results, GROUND_TRUTH)
    print("\n=== attribution metrics vs ground truth ===")
    print(f"  coverage={scores.coverage}  family_acc={scores.family_accuracy}  "
          f"desc_F1={scores.description_f1}  mean_conf={scores.mean_confidence}")
    print("\n(swap MockVLMBackend -> ClaudeVLMBackend()/QwenVLBackend() for real attribution.)")


if __name__ == "__main__":
    main()
