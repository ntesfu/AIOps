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
from collections import Counter, OrderedDict

from aiops.adjudication import (
    AllEventsTrigger,
    ChainOfThoughtPromptBuilder,
    EnrichmentFuser,
    JsonResponseParser,
    MockVLMBackend,
    QwenVLBackend,
    ZeroShotPromptBuilder,
    evaluate_attributions,
    normalize_family,
)
from aiops.adjudication.captaincook4d import (
    CaptainCook4DFrameProvider,
    CaptainCook4DProcedureContext,
    build_candidates,
    ground_truth,
)
from aiops.adjudication.evidence import DefaultEvidenceBuilder
from aiops.adjudication.pipeline import AdjudicationPipeline
from aiops.adjudication.schema import canonical_family
from aiops.data.captaincook4d import filter_segments, load_recording_split, load_segments


def diverse_order(candidates, per_recording=None):
    """Round-robin candidates across recordings so a small limit spans many videos
    (the raw segment order clumps by recording). ``per_recording`` optionally caps
    how many events any single recording may contribute."""
    groups: "OrderedDict[str, list]" = OrderedDict()
    for c in candidates:
        groups.setdefault(c.recording_id, []).append(c)
    if per_recording:
        for rid in groups:
            groups[rid] = groups[rid][:per_recording]
    ordered, i = [], 0
    while True:
        added = False
        for lst in groups.values():
            if i < len(lst):
                ordered.append(lst[i])
                added = True
        if not added:
            break
        i += 1
    return ordered


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
    p.add_argument("--per-recording", type=int, default=3,
                   help="cap events per recording before the limit (diversity)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    segments = load_segments(args.annotations_root)
    split = load_recording_split(args.annotations_root, args.split_by)
    subset = filter_segments(segments, split[args.which])
    all_cands = build_candidates(subset, errors_only=True)
    candidates = diverse_order(all_cands, per_recording=args.per_recording)[: args.limit]
    n_recs = len({c.recording_id for c in candidates})
    gt = ground_truth(args.annotations_root)
    print(f"segments={len(subset)} error-candidates={len(all_cands)} "
          f"selected={len(candidates)} across {n_recs} recordings (limit {args.limit}, "
          f"per_recording={args.per_recording}), split={args.split_by}/{args.which}", flush=True)

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
    for r in results[:12]:
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

    # Family distributions + per-family recall — makes the Missing-Step over-
    # prediction bias (and its closure) directly visible.
    ok = [r for r in results if r.status == "ok" and r.attribution is not None
          and r.event_id in gt]
    pred_dist = Counter(canonical_family(r.attribution) for r in ok)
    gt_dist = Counter(normalize_family(gt[r.event_id]["family"]) for r in ok)
    per_fam_total: Counter = Counter()
    per_fam_hit: Counter = Counter()
    for r in ok:
        g = normalize_family(gt[r.event_id]["family"])
        per_fam_total[g] += 1
        if canonical_family(r.attribution) == g:
            per_fam_hit[g] += 1
    print("\n  predicted family distribution:")
    for fam, c in pred_dist.most_common():
        print(f"    {fam:20s} {c}")
    print("  ground-truth family distribution:")
    for fam, c in gt_dist.most_common():
        print(f"    {fam:20s} {c}")
    print("  per-family recall (hits / gt-count):")
    for fam, tot in per_fam_total.most_common():
        print(f"    {fam:20s} {per_fam_hit[fam]}/{tot}")
    miss_pred = pred_dist.get("Missing Step", 0)
    miss_gt = gt_dist.get("Missing Step", 0)
    print(f"\n  Missing-Step: predicted {miss_pred}, actual {miss_gt} "
          f"(over-prediction factor {miss_pred / miss_gt:.2f})" if miss_gt else
          f"\n  Missing-Step: predicted {miss_pred}, actual 0", flush=True)
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
