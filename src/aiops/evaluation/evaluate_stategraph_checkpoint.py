from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    pad_stategraph_batch,
    read_cache_index,
)
from aiops.models.stategraph_psr import StateGraphPSRConfig, build_stategraph_psr
from aiops.training.train_stategraph_psr import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a StateGraph-PSR checkpoint with the current decoder."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = StateGraphPSRConfig(**checkpoint["model_config"])
    model = build_stategraph_psr(config, checkpoint["transition_matrix"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metadata, records = read_cache_index(args.cache_index)
    split_names = {"val", "validation"} if args.split == "val" else {"test"}
    selected = [record for record in records if record.split.lower() in split_names]
    if not selected:
        raise ValueError(f"Cache has no records for split '{args.split}'.")
    loader = DataLoader(
        StateGraphCacheDataset(
            selected,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_length,
            training=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_stategraph_batch,
    )
    use_amp = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    if args.split == "test" and not checkpoint.get("event_thresholds"):
        raise ValueError(
            "Test evaluation requires validation-calibrated event_thresholds in the checkpoint."
        )
    metrics = evaluate(
        model,
        loader,
        device,
        use_amp,
        amp_dtype,
        config.num_components,
        seconds_per_step=float(metadata.get("stride_frames", 5.0))
        / max(float(metadata.get("fps", 10.0)), 1e-6),
        event_thresholds=checkpoint.get("event_thresholds"),
        calibrate_events=args.split == "val",
        calibrate_latency=args.split == "val",
        state_incorrect_threshold=(
            checkpoint.get("validation_metrics", {}).get("state_incorrect_threshold")
            if args.split == "test"
            else None
        ),
        calibrate_state=args.split == "val",
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "recordings": len(selected),
        "decoder": "single-outcome completion peaks with temporal suppression",
        "metrics": metrics,
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
