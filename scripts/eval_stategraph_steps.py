#!/usr/bin/env python3
"""Eval-only pass for a StateGraph-PSR checkpoint, reporting the step-level baseline.

Loads a trained checkpoint, rebuilds the val/test loader exactly as training does,
and runs the (now step-instrumented) ``evaluate()``. Prints the fine metrics and the
new ``step_*`` metrics (the Track A day-one baseline + Viterbi lift). Forward pass
only — no training, a light GPU workload.

    HF unneeded. Example:
    PYTHONPATH=src python scripts/eval_stategraph_steps.py \
        --checkpoint .../best_checkpoint.pt \
        --cache-index .../index.json --split val --batch-size 4 --sequence-length 384
"""

from __future__ import annotations

import argparse
import json

from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    pad_stategraph_batch,
    read_cache_index,
)
from aiops.models.stategraph_psr import (
    StateGraphPSRConfig,
    build_dual_expert_stategraph_psr,
    build_stategraph_psr,
)
from aiops.training.train_stategraph_psr import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-index", required=True)
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=384)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument(
        "--near-online-lag-seconds",
        type=float,
        default=1.5,
        help="Trailing lookahead for the primary near-online STEP decode.",
    )
    p.add_argument(
        "--step-transition-self-bias",
        type=float,
        default=6.0,
        help="Viterbi self-transition log-score bonus (validation-selected default: 6).",
    )
    p.add_argument(
        "--step-dump-dir",
        default=None,
        help="Optional directory for compressed per-recording STEP posteriors and targets.",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("architecture_type") == "dual_expert_stategraph_psr":
        action_config = StateGraphPSRConfig(**ckpt["action_model_config"])
        config = StateGraphPSRConfig(**ckpt["event_model_config"])
        model = build_dual_expert_stategraph_psr(
            action_config, config,
            ckpt["action_transition_matrix"], ckpt["event_transition_matrix"],
        ).to(device).eval()
    else:
        config = StateGraphPSRConfig(**ckpt["model_config"])
        model = build_stategraph_psr(config, ckpt["transition_matrix"]).to(device).eval()
    missing = model.load_state_dict(ckpt["model_state"], strict=False)
    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        print(f"[load_state_dict] missing={list(missing.missing_keys)} "
              f"unexpected={list(missing.unexpected_keys)}", flush=True)
    missing_step_head = [
        key for key in getattr(missing, "missing_keys", [])
        if "psr_step_head." in key
    ]
    if missing_step_head:
        raise SystemExit(
            "checkpoint has no trained STEP head; refusing to report metrics from "
            f"randomly initialized parameters: {missing_step_head}"
        )

    metadata, records = read_cache_index(args.cache_index)
    wanted = {"val", "validation"} if args.split == "val" else {args.split}
    split_records = [r for r in records if r.split.lower() in wanted]
    if not split_records:
        raise SystemExit(f"no records in split={args.split!r}")
    fps = float(metadata.get("fps", 0.0) or 0.0)
    stride = float(metadata.get("stride_frames", 1) or 1)
    seconds_per_step = (stride / fps) if fps > 0 else 0.5

    loader = DataLoader(
        StateGraphCacheDataset(
            split_records,
            sequence_length=args.sequence_length,
            sequence_stride=max(1, args.sequence_length // 2),
            training=False,
            preload=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=pad_stategraph_batch,
    )
    print(f"split={args.split} recordings={len(split_records)} "
          f"seconds_per_step={seconds_per_step:.3f} device={device}", flush=True)

    use_amp = device.type == "cuda" and args.precision == "bf16"
    amp_dtype = torch.bfloat16
    metrics = evaluate(
        model, loader, device, use_amp, amp_dtype, config.num_components,
        seconds_per_step=seconds_per_step,
        calibrate_events=True, calibrate_state=True,
        step_lag_seconds=args.near_online_lag_seconds,
        step_transition_self_bias=args.step_transition_self_bias,
        step_dump_dir=args.step_dump_dir,
    )

    step_keys = [k for k in metrics if k.startswith("step_")]
    print("\n=== FINE action metrics ===", flush=True)
    for k in ("frame_accuracy", "raw_frame_accuracy", "edit", "f1@10", "f1@25", "f1@50"):
        if k in metrics:
            print(f"  {k:24s} {metrics[k]:.2f}")
    print("\n=== STEP-level metrics (primary) ===", flush=True)
    for k in step_keys:
        print(f"  {k:28s} {metrics[k]:.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as h:
            json.dump({k: float(v) for k, v in metrics.items()}, h, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
