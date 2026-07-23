#!/usr/bin/env python3
"""Selectively extract small IndustReal metadata files from recording ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


DEFAULT_NAMES = frozenset({"OD_labels.json", "hands.csv", "pose.csv"})


def extract_auxiliary(
    archives_dir: str | Path,
    output_root: str | Path,
    names: frozenset[str] = DEFAULT_NAMES,
) -> dict[str, object]:
    source = Path(archives_dir).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    extracted: list[dict[str, object]] = []
    skipped = 0
    for archive_path in sorted(source.glob("*.zip")):
        split = archive_path.stem.split("_p", 1)[0].lower()
        if split not in {"train", "val", "validation", "test"}:
            continue
        split = "val" if split == "validation" else split
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if member.is_dir() or path.name not in names or len(path.parts) != 2:
                    continue
                recording_id = path.parts[0]
                target = destination / "recordings" / split / recording_id / path.name
                resolved = target.resolve()
                if destination not in resolved.parents:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                if target.is_file() and target.stat().st_size == member.file_size:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                digest = hashlib.sha256()
                with archive.open(member) as reader, temporary.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
                        digest.update(chunk)
                temporary.replace(target)
                extracted.append(
                    {
                        "archive": archive_path.name,
                        "member": member.filename,
                        "path": str(target.relative_to(destination)),
                        "bytes": member.file_size,
                        "sha256": digest.hexdigest(),
                    }
                )
    manifest = {
        "archives_dir": str(source),
        "output_root": str(destination),
        "names": sorted(names),
        "extracted": extracted,
        "extracted_files": len(extracted),
        "skipped_existing_files": skipped,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "auxiliary_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(extract_auxiliary(args.archives_dir, args.output_root), indent=2))


if __name__ == "__main__":
    main()
