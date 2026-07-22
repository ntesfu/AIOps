#!/usr/bin/env python3
"""Compose action and event checkpoints, calibrate on validation, and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    pad_stategraph_batch,
    read_cache_index,
)
from aiops.evaluation.evaluate_stategraph_checkpoint import _json_safe
from aiops.models.stategraph_psr import (
    StateGraphPSRConfig,
    build_dual_expert_stategraph_psr,
)
from aiops.training.train_stategraph_psr import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose a strong action expert and mistake event expert validation-only."
    )
    parser.add_argument("--action-checkpoint", required=True)
    parser.add_argument("--event-checkpoint", required=True)
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-validation", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-incorrect-false-alerts-per-minute", type=float, default=2.0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    action_checkpoint = torch.load(args.action_checkpoint, map_location=device, weights_only=False)
    event_checkpoint = torch.load(args.event_checkpoint, map_location=device, weights_only=False)
    action_config = StateGraphPSRConfig(**action_checkpoint["model_config"])
    event_config = StateGraphPSRConfig(**event_checkpoint["model_config"])
    model = build_dual_expert_stategraph_psr(
        action_config,
        event_config,
        action_checkpoint["transition_matrix"],
        event_checkpoint["transition_matrix"],
    ).to(device)
    model.action_expert.load_state_dict(action_checkpoint["model_state"])
    model.event_expert.load_state_dict(event_checkpoint["model_state"])
    model.eval()

    metadata, records = read_cache_index(args.cache_index)
    validation_records = [
        record for record in records if record.split.lower() in {"val", "validation"}
    ]
    if not validation_records:
        raise ValueError("Cache has no validation records.")
    loader = DataLoader(
        StateGraphCacheDataset(
            validation_records,
            sequence_length=args.sequence_length,
            sequence_stride=max(1, args.sequence_length // 2),
            training=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_stategraph_batch,
    )
    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    metrics = evaluate(
        model,
        loader,
        device,
        use_amp,
        amp_dtype,
        event_config.num_components,
        seconds_per_step=float(metadata.get("stride_frames", 5.0))
        / max(float(metadata.get("fps", 10.0)), 1e-6),
        calibrate_events=True,
        calibrate_latency=True,
        calibrate_state=True,
        max_incorrect_false_alerts_per_minute=args.max_incorrect_false_alerts_per_minute,
    )
    event_thresholds = [
        metrics["correct_event_threshold"],
        metrics["incorrect_event_threshold"],
        metrics["remove_event_threshold"],
    ]
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture_type": "dual_expert_stategraph_psr",
            "action_model_config": action_config.to_dict(),
            "event_model_config": event_config.to_dict(),
            "action_transition_matrix": torch.as_tensor(
                action_checkpoint["transition_matrix"]
            ).cpu(),
            "event_transition_matrix": torch.as_tensor(
                event_checkpoint["transition_matrix"]
            ).cpu(),
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "action_checkpoint": str(Path(args.action_checkpoint)),
            "event_checkpoint": str(Path(args.event_checkpoint)),
            "epoch": {
                "action": action_checkpoint.get("epoch"),
                "event": event_checkpoint.get("epoch"),
            },
            "event_thresholds": event_thresholds,
            "validation_metrics": metrics,
        },
        output_checkpoint,
    )
    payload = {
        "architecture_type": "dual_expert_stategraph_psr",
        "action_checkpoint": str(Path(args.action_checkpoint).resolve()),
        "event_checkpoint": str(Path(args.event_checkpoint).resolve()),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "recordings": len(validation_records),
        "metrics": metrics,
    }
    rendered = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    output_validation = Path(args.output_validation)
    output_validation.parent.mkdir(parents=True, exist_ok=True)
    output_validation.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
