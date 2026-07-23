#!/usr/bin/env python3
"""Attach strictly aligned ROI features to an existing StateGraph cache."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from aiops.data.stategraph_cache import read_cache_index, write_cache_index


def attach_motion_aux(
    cache_index: str | Path, features_root: str | Path, output_dir: str | Path
) -> dict[str, object]:
    metadata, records = read_cache_index(cache_index)
    feature_root = Path(features_root).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    manifest_path = feature_root / "roi_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ROI manifest: {manifest_path}")
    roi_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        roi_manifest.get("causal") is not True
        or int(roi_manifest.get("max_future_offset_frames", 1)) > 0
    ):
        raise ValueError("ROI feature manifest does not prove current/past-only extraction.")

    output_records = []
    auxiliary_dim: int | None = None
    for ordinal, record in enumerate(records, start=1):
        roi_path = feature_root / record.split / f"{record.recording_id}.npz"
        if not roi_path.is_file():
            raise FileNotFoundError(f"Missing ROI features: {roi_path}")
        with np.load(record.path, allow_pickle=False) as base, np.load(
            roi_path, allow_pickle=False
        ) as roi:
            auxiliary = roi["motion"].astype(np.float32)
            if len(auxiliary) != len(base["step"]):
                raise ValueError(
                    f"{record.recording_id}: ROI rows {len(auxiliary)} do not match "
                    f"cache rows {len(base['step'])}."
                )
            if "prediction_frame_indices" not in base.files:
                raise ValueError(f"{record.recording_id}: base cache lacks prediction provenance.")
            if not np.array_equal(
                base["prediction_frame_indices"], roi["prediction_frame_indices"]
            ):
                raise ValueError(
                    f"{record.recording_id}: ROI prediction frames are not exactly aligned."
                )
            if (
                np.max(
                    roi["source_frame_indices"]
                    - roi["prediction_frame_indices"][:, None]
                )
                > 0
            ):
                raise ValueError(f"{record.recording_id}: ROI features contain future frames.")
            if auxiliary_dim is None:
                auxiliary_dim = int(auxiliary.shape[1])
            elif auxiliary.shape[1] != auxiliary_dim:
                raise ValueError("ROI feature dimensions differ across recordings.")
            arrays = {name: base[name] for name in base.files}
            arrays["motion_aux"] = auxiliary
        destination = destination_root / record.split / f"{record.recording_id}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        output_records.append(
            replace(
                record,
                path=destination,
                motion_aux_dim=int(auxiliary.shape[1]),
            )
        )
        print(f"[{ordinal}/{len(records)}] {record.split}/{record.recording_id}", flush=True)

    output_metadata = dict(metadata)
    output_metadata["feature_backends"] = {
        **dict(metadata.get("feature_backends", {})),
        "motion_aux": roi_manifest.get("backend", "industreal_roi"),
    }
    output_metadata["motion_aux_provenance"] = {
        "kind": "current_frame_roi",
        "format_version": int(roi_manifest.get("format_version", 0)),
        "causality_verified": True,
        "max_future_offset_frames": int(
            roi_manifest.get("max_future_offset_frames", 0)
        ),
        "manifest": str(manifest_path),
        "roi_names": roi_manifest.get("roi_names", []),
        "feature_contract": roi_manifest.get("feature_contract"),
        "source_contract": roi_manifest.get("source_contract"),
    }
    index_path = destination_root / "index.json"
    write_cache_index(index_path, output_records, output_metadata)
    return {
        "index": str(index_path),
        "recordings": len(output_records),
        "motion_aux_dim": auxiliary_dim or 0,
        "causal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            attach_motion_aux(args.cache_index, args.features_root, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
