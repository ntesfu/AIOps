from __future__ import annotations

import argparse
import json
import shutil
import ssl
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATASET_URL = "http://www.crcv.ucf.edu/datasets/ugur/TinyVIRAT.zip"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    context = ssl._create_unverified_context() if url.startswith("https://") else None
    with urllib.request.urlopen(url, timeout=60, context=context) as response:
        with output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def infer_label(path: str) -> str:
    parts = Path(path).parts
    for part in reversed(parts[:-1]):
        if part and not part.startswith("__") and part.lower() not in {"train", "test", "val", "videos"}:
            return part
    return Path(path).stem.split("_")[0]


def extract_subset(zip_path: Path, output_dir: Path, max_videos: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    rows: list[dict[str, str]] = []

    with zipfile.ZipFile(zip_path) as archive:
        video_members = [
            member
            for member in archive.namelist()
            if not member.endswith("/") and Path(member).suffix.lower() in VIDEO_EXTENSIONS
        ]
        video_members.sort()

        for member in video_members[:max_videos]:
            label = infer_label(member)
            target = output_dir / "videos" / label / Path(member).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            rows.append({"video_path": str(target), "label": label, "source_member": member})

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return manifest_path


def extract_labeled_subset(zip_path: Path, output_dir: Path, max_classes: int, max_per_class: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    rows: list[dict[str, str]] = []

    with zipfile.ZipFile(zip_path) as archive:
        payload = json.loads(archive.read("TinyVIRAT/tiny_train.json"))
        by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in payload["tubes"]:
            labels = item.get("label") or []
            if not labels:
                continue
            label = str(labels[0])
            by_label[label].append(item)

        selected_labels = sorted(by_label, key=lambda label: len(by_label[label]), reverse=True)[:max_classes]
        for label in selected_labels:
            for item in by_label[label][:max_per_class]:
                relative_path = str(item["path"])
                member = f"TinyVIRAT/videos/train/{relative_path}"
                safe_name = "__".join(Path(relative_path).parts)
                target = output_dir / "videos" / label / safe_name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with archive.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                rows.append(
                    {
                        "video_path": str(target),
                        "label": label,
                        "source_member": member,
                    }
                )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract a tiny TinyVIRAT subset.")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--archive", default="data/raw/TinyVIRAT.zip")
    parser.add_argument("--output-dir", default="data/processed/tinyvirat_subset")
    parser.add_argument("--max-videos", type=int, default=24)
    parser.add_argument("--max-classes", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, default=8)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    download_file(args.url, archive_path)
    manifest_path = extract_labeled_subset(
        archive_path,
        Path(args.output_dir),
        max_classes=args.max_classes,
        max_per_class=args.max_per_class,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
