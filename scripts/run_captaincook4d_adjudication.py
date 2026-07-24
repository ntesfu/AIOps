"""Run the VLM mistake adjudicator on CaptainCook4D error segments.

Wires the CaptainCook4D adapter (frames + procedure context + ground truth) into
the pluggable pipeline with a real VLM backend, and reports attribution metrics
(family accuracy, description token-F1) against the ground-truth error descriptions.

    HF_HOME=/media/.../hf_cache PYTHONPATH=src python scripts/run_captaincook4d_adjudication.py \
        --annotations-root .../captaincook4d/annotations \
        --data-root        .../captaincook4d/data \
        --frame-cache      .../captaincook4d/adj_frames \
        --backend qwen --limit 20 --prompt cot
"""

from __future__ import annotations

import argparse
import json

from aiops.adjudication import (
    AllEventsTrigger,
    ChainOfThoughtPromptBuilder,
    EnrichmentFuser,
    JsonResponseParser,
    MockVLMBackend,
    QwenVLBackend,
    ZeroShotPromptBuilder,
    evaluate_attributions,
)
from aiops.adjudication.captaincook4d import (
    CaptainCook4DFrameProvider,
    CaptainCook4DProcedureContext,
    build_candidates,
    ground_truth,
)
from aiops.adjudication.evidence import DefaultEvidenceBuilder
from aiops.adjudication.pipeline import AdjudicationPipeline
from aiops.data.captaincook4d import filter_segments, load_recording_split, load_segments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--frame-cache", required=True)
    p.add_argument("--split-by", default="person")
    p.add_argument("--which", default="test", choices=["train", "val", "test"])
    p.add_argument("--backend", default="qwen", choices=["qwen", "mock"])
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--prompt", default="cot", choices=["cot", "zeroshot"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    segments = load_segments(args.annotations_root)
    split = load_recording_split(args.annotations_root, args.split_by)
    subset = filter_segments(segments, split[args.which])
    candidates = build_candidates(subset, errors_only=True)[: args.limit]
    gt = ground_truth(args.annotations_root)
    print(f"segments={len(subset)} error-candidates={len(candidates)} (limit {args.limit}), "
          f"split={args.split_by}/{args.which}", flush=True)

    backend = (MockVLMBackend() if args.backend == "mock"
               else QwenVLBackend(model_id=args.model_id))
    prompt_builder = (ChainOfThoughtPromptBuilder() if args.prompt == "cot"
                      else ZeroShotPromptBuilder())
    pipe = AdjudicationPipeline(
        trigger=AllEventsTrigger(),
        evidence_builder=DefaultEvidenceBuilder(
            frame_provider=CaptainCook4DFrameProvider(args.data_root, args.frame_cache),
            context=CaptainCook4DProcedureContext(segments)),
        prompt_builder=prompt_builder,
        backend=backend,
        parser=JsonResponseParser(),
        fuser=EnrichmentFuser(),
    )

    results = pipe.adjudicate(candidates)
    for r in results[:6]:
        g = gt.get(r.event_id, {})
        a = r.attribution
        print(f"\n[{r.event_id}] status={r.status} lat={r.latency_ms and round(r.latency_ms)}ms")
        print(f"  GT   family={g.get('family')!r} desc={g.get('description')!r}")
        if a:
            print(f"  PRED family={a.mistake_family!r} desc={a.description!r} conf={a.confidence}")

    scores = evaluate_attributions(results, gt)
    print("\n=== attribution metrics ===")
    print(f"  n={scores.n} coverage={scores.coverage} family_acc={scores.family_accuracy} "
          f"desc_F1={scores.description_f1} mean_conf={scores.mean_confidence}", flush=True)
    if args.out:
        payload = {"scores": scores.__dict__,
                   "results": [{"event_id": r.event_id, "status": r.status,
                                "attribution": r.attribution.to_dict() if r.attribution else None,
                                "gt": gt.get(r.event_id)} for r in results]}
        with open(args.out, "w", encoding="utf-8") as h:
            json.dump(payload, h, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
