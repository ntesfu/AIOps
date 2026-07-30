#!/usr/bin/env python3
"""Measure genuine video decode plus Swin3D/ConvNeXt feature throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aiops.features.assembly101_cache import BatchedConvNeXtExtractor
from aiops.features.assembly101_video import Swin3DFeatureExtractor
from aiops.features.assembly101_video_cache import _extract_recording


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Assembly101 causal visual feature pipeline on one video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if min(args.target_fps, args.clip_frames, args.stride_frames, args.batch_size) <= 0:
        parser.error("fps, clip/stride frames, and batch size must be positive")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    motion_extractor = Swin3DFeatureExtractor(device, args.precision)
    appearance_extractor = BatchedConvNeXtExtractor(device, args.precision)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    motion, appearance, sampled_frames = _extract_recording(
        Path(args.video),
        motion_extractor,
        appearance_extractor,
        target_fps=args.target_fps,
        clip_frames=args.clip_frames,
        stride_frames=args.stride_frames,
        batch_size=args.batch_size,
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    clips = len(sampled_frames)
    represented_seconds = (
        (int(sampled_frames[-1]) - int(sampled_frames[0]) + args.stride_frames)
        / args.target_fps
    )
    update_interval_ms = 1000.0 * args.stride_frames / args.target_fps
    payload = {
        "scope": "video decode, causal clip construction, Swin3D-S motion, JPEG current-frame conversion, and ConvNeXt-T appearance",
        "video": str(Path(args.video).resolve()),
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "target_fps": args.target_fps,
        "clip_frames": args.clip_frames,
        "stride_frames": args.stride_frames,
        "clips": clips,
        "motion_shape": list(motion.shape),
        "appearance_shape": list(appearance.shape),
        "elapsed_seconds": elapsed,
        "mean_ms_per_clip": 1000.0 * elapsed / clips,
        "clips_per_second": clips / elapsed,
        "represented_video_seconds": represented_seconds,
        "video_seconds_processed_per_wall_second": represented_seconds / elapsed,
        "source_update_interval_ms": update_interval_ms,
        "meets_update_interval_by_mean_throughput": 1000.0 * elapsed / clips <= update_interval_ms,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if str(device).startswith("cuda")
            else None
        ),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
