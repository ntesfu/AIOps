from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.assembly101 import (
    dense_assembly_targets,
    read_coarse_action_segments,
    read_mistake_segments,
)
from aiops.data.procedure_schema import CACHE_SCHEMA_VERSION, EVENT_OUTCOME_NAMES
from aiops.data.stategraph_cache import StateGraphCacheRecord, save_cache_record, write_cache_index
from aiops.features.assembly101_cache import BatchedConvNeXtExtractor
from aiops.features.assembly101_video import Swin3DFeatureExtractor, iter_causal_clips


def _official_actions(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["action_id"]))
    names = [str(row["action_cls"]).strip().lower() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Official coarse action map contains duplicate class names")
    return names, rows


def _extract_recording(
    video: Path,
    motion_extractor: Swin3DFeatureExtractor,
    appearance_extractor: BatchedConvNeXtExtractor,
    *,
    target_fps: float,
    clip_frames: int,
    stride_frames: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    motion_rows: list[np.ndarray] = []
    appearance_rows: list[np.ndarray] = []
    sampled_frames: list[int] = []
    clip_batch: list[np.ndarray] = []

    def flush() -> None:
        if not clip_batch:
            return
        motion_rows.append(motion_extractor.extract(clip_batch))
        encoded = []
        for clip in clip_batch:
            # ConvNeXt supplies complementary spatial detail from the current
            # frame; Swin3D sees the complete past-only temporal window.
            ok, payload = cv2.imencode(".jpg", cv2.cvtColor(clip[-1], cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("Could not encode current ego frame")
            encoded.append(payload.tobytes())
        appearance_rows.append(appearance_extractor.extract(encoded, batch_size=len(encoded)))
        clip_batch.clear()

    for clip in iter_causal_clips(
        video,
        target_fps=target_fps,
        clip_frames=clip_frames,
        stride_frames=stride_frames,
        output_size=(224, 224),
    ):
        sampled_frames.append(clip.end_frame)
        clip_batch.append(clip.frames)
        if len(clip_batch) == batch_size:
            flush()
    flush()
    if not sampled_frames:
        raise RuntimeError(f"No clips decoded from {video}")
    return (
        np.concatenate(motion_rows).astype(np.float32),
        np.concatenate(appearance_rows).astype(np.float32),
        np.asarray(sampled_frames, dtype=np.int64),
    )


def build_video_cache(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest_path.parent
    annotation_root = data_root / "official_hf" / "annotations"
    mistake_root = data_root / "annotations" / "mistake-detection"
    actions, action_rows = _official_actions(annotation_root / "coarse-annotations" / "actions.csv")
    action_ids = ["__background__", *actions]
    action_to_index = {name: index for index, name in enumerate(action_ids)}
    records_payload = [
        row
        for index, row in enumerate(manifest["records"])
        if index % args.num_shards == args.shard_index
    ]
    if args.recording_id:
        requested = set(args.recording_id)
        records_payload = [row for row in records_payload if row["recording_id"] in requested]
        missing = requested - {row["recording_id"] for row in records_payload}
        if missing:
            raise ValueError(f"Requested recording IDs are absent from this shard: {sorted(missing)}")
    if args.max_recordings is not None:
        records_payload = records_payload[: args.max_recordings]
    all_mistakes = {
        row["recording_id"]: read_mistake_segments(mistake_root / f"{row['recording_id']}.csv")
        for row in manifest["records"]
    }
    components = sorted({segment.subject for values in all_mistakes.values() for segment in values})
    component_to_index = {name: index for index, name in enumerate(components)}

    motion_extractor = Swin3DFeatureExtractor(args.device, args.precision)
    appearance_extractor = BatchedConvNeXtExtractor(args.device, args.precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_records: list[StateGraphCacheRecord] = []
    peak_gib = 0.0
    for number, row in enumerate(records_payload, 1):
        rid = row["recording_id"]
        destination = output_dir / row["split"] / f"{rid}.npz"
        print(f"[{number}/{len(records_payload)}] {row['split']}/{rid}", flush=True)
        if destination.is_file() and args.skip_existing:
            with np.load(destination, allow_pickle=False) as arrays:
                length = len(arrays["step"])
        else:
            motion, appearance, sampled_frames = _extract_recording(
                data_root / row["video_relative_path"],
                motion_extractor,
                appearance_extractor,
                target_fps=args.target_fps,
                clip_frames=args.clip_frames,
                stride_frames=args.stride_frames,
                batch_size=args.feature_batch_size,
            )
            coarse_paths = [
                annotation_root / "coarse-annotations" / "coarse_labels" / name
                for name in row["coarse_label_files"]
            ]
            targets = dense_assembly_targets(
                sampled_frames,
                all_mistakes[rid],
                action_to_index,
                component_to_index,
                background_index=0,
                action_segments=read_coarse_action_segments(coarse_paths),
            )
            length = len(sampled_frames)
            save_cache_record(
                destination,
                motion=motion,
                appearance=appearance,
                sensor=np.zeros((length, 1), dtype=np.float32),
                modality_mask=np.column_stack(
                    (np.ones((length, 2), dtype=np.bool_), np.zeros(length, dtype=np.bool_))
                ),
                timestamps=sampled_frames.astype(np.float32) / args.target_fps,
                **targets,
            )
        if motion_extractor.device.type == "cuda":
            peak_gib = max(
                peak_gib,
                motion_extractor.torch.cuda.max_memory_reserved(motion_extractor.device) / 2**30,
            )
        cache_records.append(
            StateGraphCacheRecord(
                recording_id=rid,
                split=row["split"],
                path=destination,
                num_steps=len(action_ids),
                motion_dim=motion_extractor.output_dim,
                appearance_dim=appearance_extractor.output_dim,
                sensor_dim=1,
                num_components=len(components),
                num_completion_components=len(components),
                num_frames=length,
            )
        )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": "Assembly101",
        "manifest": str(manifest_path),
        "manifest_selection": manifest["selection_name"],
        "label_contract": "complete_coarse_action_plus_part_to_part_mistake_and_state",
        "event_outcomes": list(EVENT_OUTCOME_NAMES),
        "completion_components": components,
        "state_components": components,
        "action_ids": action_ids,
        "action_descriptions": action_ids,
        "action_taxonomy": action_rows,
        "background_action_id": "__background__",
        "state_names": ["incorrect", "not_completed", "correct"],
        "fps": args.target_fps,
        "stride_frames": args.stride_frames,
        "clip_frames": args.clip_frames,
        "causal_clips": True,
        "camera": "e1",
        "feature_backends": {
            "motion": "torchvision_swin3d_s_kinetics400_v1",
            "appearance": "torchvision_convnext_tiny_imagenet1k_v1",
            "sensor": None,
        },
        "peak_feature_vram_gib": peak_gib,
        "shard": {"index": args.shard_index, "count": args.num_shards},
    }
    index_path = output_dir / f"index.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    write_cache_index(index_path, cache_records, metadata)
    return {"index": str(index_path), "recordings": len(cache_records), "peak_vram_gib": peak_gib}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build genuine 30-fps Assembly101 ego cache.")
    parser.add_argument("--manifest", default="data/raw/assembly101/ego30_manifest.json")
    parser.add_argument("--output-dir", default="data/processed/assembly101_ego30_stategraph")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--recording-id", action="append")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    print(json.dumps(build_video_cache(args), indent=2))


if __name__ == "__main__":
    main()
