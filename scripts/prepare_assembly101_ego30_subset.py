#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


HF_REPO = "cvml-nus/assembly101"
HF_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
# e1 is the left/primary ego view. The headset serial changed during capture.
E1_CAMERAS = ("HMC_84346135_mono10bit.mp4", "HMC_21176875_mono10bit.mp4")


def _recording_id(label_or_recording: str) -> str:
    name = Path(label_or_recording).name
    for prefix in ("assembly_", "disassembly_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.removesuffix(".txt").removesuffix(".csv")


def _view_map(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "/" not in line:
            continue
        sequence, camera = line.split("/", 1)
        rid = _recording_id(sequence)
        if camera in E1_CAMERAS and camera not in result[rid]:
            result[rid].append(camera)
    return dict(result)


def _coarse_labels(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.glob("*.txt")):
        result[_recording_id(path.name)].append(path.name)
    return dict(result)


def build_manifest(
    old_manifest: Path,
    annotation_root: Path,
    output: Path,
    max_per_actor: int = 10,
    additional_train_actors: int = 0,
) -> dict:
    old = json.loads(old_manifest.read_text(encoding="utf-8"))
    actor_split = {row["actor_id"]: row["split"] for row in old["records"]}
    old_ids = {row["recording_id"] for row in old["records"]}
    coarse_root = annotation_root / "coarse-annotations"
    views = _view_map(coarse_root / "coarse_seq_views.txt")
    coarse = _coarse_labels(coarse_root / "coarse_labels")
    mistake_root = old_manifest.parent / "annotations" / "mistake-detection"

    candidates: dict[str, list[dict]] = defaultdict(list)
    for mistake_path in sorted(mistake_root.glob("*.csv")):
        rid = mistake_path.stem
        actor = rid.split("_user_id_", 1)[0].rsplit("_", 1)[-1]
        if rid not in views or rid not in coarse:
            continue
        rows = max(0, len(mistake_path.read_text(encoding="utf-8").splitlines()) - 1)
        camera = views[rid][0]
        candidates[actor].append(
            {
                "recording_id": rid,
                "split": actor_split.get(actor, "train"),
                "actor_id": actor,
                "toy_id": rid.split("-", 1)[-1].split("_", 1)[0][:-1],
                "mistake_annotation_rows": rows,
                "coarse_label_files": sorted(coarse[rid]),
                "camera": "e1",
                "camera_file": camera,
                "hf_path": f"recordings/{rid}/{camera}",
                "video_relative_path": f"official_hf/recordings/{rid}/{camera}",
            }
        )

    new_actor_ranking = sorted(
        (actor for actor in candidates if actor not in actor_split),
        key=lambda actor: (
            -sum(row["mistake_annotation_rows"] for row in candidates[actor]),
            actor,
        ),
    )
    selected_new_actors = set(new_actor_ranking[:additional_train_actors])
    selected_actors = set(actor_split) | selected_new_actors
    records = []
    for actor in sorted(selected_actors):
        # Retain prior recordings where possible; fill the expansion with the
        # most mistake-rich recordings, then break ties deterministically.
        ranked = sorted(
            candidates[actor],
            key=lambda row: (
                row["recording_id"] not in old_ids,
                -row["mistake_annotation_rows"],
                row["recording_id"],
            ),
        )
        records.extend(ranked[:max_per_actor])
    records.sort(key=lambda row: ({"train": 0, "val": 1, "test": 2}[row["split"]], row["recording_id"]))
    counts = Counter(row["split"] for row in records)
    payload = {
        "format_version": 1,
        "selection_name": "assembly101_ego_e1_30fps_actor_disjoint_v1",
        "dataset": {"repo_id": HF_REPO, "revision": HF_REVISION, "license": "CC BY-NC 4.0"},
        "sampling": {
            "source_fps": 60,
            "annotation_and_decode_fps": 30,
            "clip_frames": 32,
            "stride_frames": 8,
            "camera": "e1 only",
        },
        "selection": {
            "actor_disjoint": True,
            "max_recordings_per_actor": max_per_actor,
            "additional_train_actors": sorted(selected_new_actors),
            "actors": {split: sorted({r["actor_id"] for r in records if r["split"] == split}) for split in ("train", "val", "test")},
            "recordings": dict(counts),
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def download(
    payload: dict, root: Path, splits: set[str] | None = None, workers: int = 8
) -> None:
    from huggingface_hub import snapshot_download

    dataset = payload["dataset"]
    chosen = [row for row in payload["records"] if splits is None or row["split"] in splits]
    snapshot_download(
        repo_id=dataset["repo_id"],
        repo_type="dataset",
        revision=dataset["revision"],
        allow_patterns=[row["hf_path"] for row in chosen],
        local_dir=root / "official_hf",
        max_workers=workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the one-view Assembly101 ego30 subset.")
    parser.add_argument("--old-manifest", default="data/raw/assembly101/subset_manifest.json")
    parser.add_argument("--annotation-root", default="data/raw/assembly101/official_hf/annotations")
    parser.add_argument("--output", default="data/raw/assembly101/ego30_manifest.json")
    parser.add_argument("--max-per-actor", type=int, default=10)
    parser.add_argument(
        "--additional-train-actors",
        type=int,
        default=0,
        help="Add this many highest-mistake actors to train while preserving existing val/test actors.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-splits", nargs="*", choices=("train", "val", "test"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    payload = build_manifest(
        Path(args.old_manifest),
        Path(args.annotation_root),
        Path(args.output),
        args.max_per_actor,
        args.additional_train_actors,
    )
    print(json.dumps({"selection": payload["selection"], "output": args.output}, indent=2))
    if args.download:
        download(
            payload,
            Path(args.output).parent,
            set(args.download_splits or ()) or None,
            args.workers,
        )


if __name__ == "__main__":
    main()
