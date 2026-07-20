#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ANNOTATION_URL = (
    "https://github.com/assembly-101/assembly101-mistake-detection/"
    "archive/refs/heads/main.zip"
)
ANNOTATION_SHA256 = "f30c55a1cdf6f17fbd2516e598e038b5c7cec3532a61eb01a5cc99360d92e05e"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_record(
    data_root: Path, row: dict[str, Any], clean_retry: int = 1
) -> dict[str, Any]:
    destination = data_root / row["archive_relative_path"]
    expected_size = int(row["expected_size"])
    expected_hash = str(row["sha256_lfs_oid"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and sha256(destination) == expected_hash
    ):
        return {"recording_id": row["recording_id"], "status": "verified", "bytes": expected_size}

    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and not partial.exists():
        # Preserve useful bytes from interrupted older download commands.
        destination.replace(partial)
    if partial.exists() and partial.stat().st_size >= expected_size:
        # A few CDN redirects ignore Range and append a complete object to an
        # old partial. Such an oversized file cannot be repaired by resuming.
        partial.unlink()
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--silent",
        "--show-error",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        row["url"],
    ]
    subprocess.run(command, check=True)
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        if actual_size > expected_size:
            partial.unlink()
            if clean_retry > 0:
                return download_record(data_root, row, clean_retry=clean_retry - 1)
        raise RuntimeError(
            f"{row['recording_id']}: size {actual_size} does not match {expected_size}"
        )
    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        partial.unlink()
        if clean_retry > 0:
            return download_record(data_root, row, clean_retry=clean_retry - 1)
        raise RuntimeError(
            f"{row['recording_id']}: SHA-256 {actual_hash} does not match {expected_hash}"
        )
    partial.replace(destination)
    return {"recording_id": row["recording_id"], "status": "downloaded", "bytes": actual_size}


def ensure_official_annotations(data_root: Path) -> dict[str, Any]:
    destination = data_root / "annotations" / "mistake-detection"
    existing = list(destination.glob("*.csv")) if destination.is_dir() else []
    if len(existing) == 328:
        return {"status": "verified", "csv_files": len(existing), "path": str(destination)}
    archive = data_root / "archives" / "assembly101-mistake-detection-main.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or sha256(archive) != ANNOTATION_SHA256:
        partial = archive.with_suffix(".zip.part")
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "8",
                "--retry-all-errors",
                "--silent",
                "--show-error",
                "--output",
                str(partial),
                ANNOTATION_URL,
            ],
            check=True,
        )
        if sha256(partial) != ANNOTATION_SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError("Official mistake-annotation archive failed SHA-256 verification")
        partial.replace(archive)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        members = [
            name
            for name in handle.namelist()
            if "/annots/" in name and name.lower().endswith(".csv")
        ]
        if len(members) != 328:
            raise RuntimeError(f"Official annotation archive has {len(members)} CSVs, expected 328")
        for member in members:
            (destination / Path(member).name).write_bytes(handle.read(member))
    return {"status": "downloaded", "csv_files": len(members), "path": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume and verify the reproducible public Assembly101 fallback subset."
    )
    parser.add_argument("--manifest", default="data/raw/assembly101/subset_manifest.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotation_result = ensure_official_annotations(manifest_path.parent)
    print(json.dumps({"annotations": annotation_result}), flush=True)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_record, manifest_path.parent, row): row
            for row in payload["records"]
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                failures.append({"recording_id": row["recording_id"], "error": str(exc)})
                print(json.dumps(failures[-1]), flush=True)
    summary = {
        "verified": len(results),
        "failed": len(failures),
        "bytes": sum(result["bytes"] for result in results),
        "annotations": annotation_result,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
