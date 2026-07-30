#!/usr/bin/env python3
"""Rebuild cached sensors from raw hands and pose without recomputing video."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from aiops.data.industreal import (
    attach_industreal_auxiliary,
    discover_industreal_recordings,
)
from aiops.data.stategraph_cache import read_cache_index, write_cache_index
from aiops.features.industreal_cache import _sample_sensors


def rebuild_sensors(
    cache_index: str | Path,
    data_root: str | Path,
    auxiliary_root: str | Path,
    output_dir: str | Path,
    *,
    sensor_dim: int = 113,
    max_recordings: int | None = None,
) -> dict[str, object]:
    if sensor_dim <= 0:
        raise ValueError("sensor_dim must be positive.")
    metadata, records = read_cache_index(cache_index)
    recordings = attach_industreal_auxiliary(
        discover_industreal_recordings(data_root), auxiliary_root
    )
    by_id = {recording.recording_id: recording for recording in recordings}
    selected = records[:max_recordings] if max_recordings else records
    destination_root = Path(output_dir).expanduser().resolve()
    rebuilt_records = []
    for ordinal, record in enumerate(selected, start=1):
        recording = by_id.get(record.recording_id)
        if recording is None:
            raise FileNotFoundError(
                f"Could not resolve raw recording {record.recording_id}."
            )
        with np.load(record.path, allow_pickle=False) as source:
            if "prediction_frame_indices" not in source.files:
                raise ValueError(
                    f"{record.recording_id} lacks prediction_frame_indices."
                )
            arrays = {name: source[name] for name in source.files}
            frame_indices = source["prediction_frame_indices"].astype(np.int64)
        sensor, sensor_present = _sample_sensors(
            recording, frame_indices, sensor_dim
        )
        modality_mask = arrays["modality_mask"].astype(np.bool_, copy=True)
        modality_mask[:, 2] = sensor_present
        arrays["sensor"] = sensor
        arrays["modality_mask"] = modality_mask
        destination = destination_root / record.split / f"{record.recording_id}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        rebuilt_records.append(
            replace(record, path=destination, sensor_dim=sensor_dim)
        )
        print(
            f"[{ordinal}/{len(selected)}] {record.split}/{record.recording_id} "
            f"sensor coverage={float(sensor_present.mean()):.3f}",
            flush=True,
        )

    output_metadata = dict(metadata)
    output_metadata["sensor_contract"] = {
        "version": 2,
        "sources": ["hands", "headset_pose"],
        "gaze_excluded": True,
        "hand_dim": 104,
        "pose_dim": 9,
        "sensor_dim": sensor_dim,
        "frame_alignment": "nearest_current_sample",
    }
    index_path = destination_root / "index.json"
    write_cache_index(index_path, rebuilt_records, output_metadata)
    return {
        "index": str(index_path),
        "recordings": len(rebuilt_records),
        "sensor_dim": sensor_dim,
        "gaze_excluded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace legacy cached sensors with hands + headset pose."
    )
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--auxiliary-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sensor-dim", type=int, default=113)
    parser.add_argument("--max-recordings", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            rebuild_sensors(
                args.cache_index,
                args.data_root,
                args.auxiliary_root,
                args.output_dir,
                sensor_dim=args.sensor_dim,
                max_recordings=args.max_recordings,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
