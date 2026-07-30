#!/usr/bin/env python3
"""Benchmark the trainable StateGraph head without claiming backbone latency."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from aiops.data.stategraph_cache import read_cache_index
from aiops.models.stategraph_psr import (
    StateGraphPSRConfig,
    build_dual_expert_stategraph_psr,
    build_stategraph_psr,
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark StateGraph temporal-head latency and peak VRAM."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-index", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.iterations) <= 0 or args.warmup < 0:
        parser.error("batch size, sequence length, and iterations must be positive; warmup cannot be negative")

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture_type") == "dual_expert_stategraph_psr":
        action_config = StateGraphPSRConfig(**checkpoint["action_model_config"])
        config = StateGraphPSRConfig(**checkpoint["event_model_config"])
        model = build_dual_expert_stategraph_psr(
            action_config,
            config,
            checkpoint["action_transition_matrix"],
            checkpoint["event_transition_matrix"],
        ).to(device).eval()
    else:
        config = StateGraphPSRConfig(**checkpoint["model_config"])
        model = build_stategraph_psr(config, checkpoint["transition_matrix"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    dtype = torch.float32
    if device.type == "cuda" and args.precision == "bf16":
        dtype = torch.bfloat16
    elif device.type == "cuda" and args.precision == "fp16":
        dtype = torch.float16
    shape = (args.batch_size, args.sequence_length)
    motion = torch.randn(*shape, config.motion_dim, device=device)
    appearance = torch.randn(*shape, config.appearance_dim, device=device)
    sensor = torch.randn(*shape, config.sensor_dim, device=device)
    motion_aux = (
        torch.randn(*shape, config.motion_aux_dim, device=device)
        if config.motion_aux_dim > 0
        else None
    )
    valid = torch.ones(*shape, dtype=torch.bool, device=device)
    modality_mask = torch.ones(*shape, 3, dtype=torch.bool, device=device)
    use_amp = device.type == "cuda" and args.precision != "fp32"

    def forward() -> None:
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=use_amp
        ):
            model(
                motion,
                appearance,
                sensor,
                valid_mask=valid,
                modality_mask=modality_mask,
                motion_aux=motion_aux,
            )

    for _ in range(args.warmup):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings_ms: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append(1000.0 * (time.perf_counter() - started))

    metadata = None
    update_interval_ms = None
    clip_context_ms = None
    if args.cache_index:
        metadata, _ = read_cache_index(args.cache_index)
        fps = float(metadata.get("fps", 0.0))
        if fps > 0:
            update_interval_ms = 1000.0 * float(metadata.get("stride_frames", 1)) / fps
            clip_context_ms = 1000.0 * float(metadata.get("clip_frames", 1)) / fps
    mean_ms = statistics.fmean(timings_ms)
    payload = {
        "scope": "StateGraph temporal head only; excludes video decoding and frozen visual backbones",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "precision": args.precision,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "iterations": args.iterations,
        "latency_ms": {
            "mean": mean_ms,
            "median": statistics.median(timings_ms),
            "p95": _percentile(timings_ms, 0.95),
            "maximum": max(timings_ms),
        },
        "feature_steps_per_second": (
            1000.0 * args.batch_size * args.sequence_length / mean_ms
        ),
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else None
        ),
        "source_update_interval_ms": update_interval_ms,
        "source_clip_context_ms": clip_context_ms,
        "head_meets_update_interval_at_p95": (
            _percentile(timings_ms, 0.95) <= update_interval_ms
            if update_interval_ms is not None
            else None
        ),
        "cache_metadata": metadata,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
