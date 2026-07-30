"""Download and safely extract the official IndustReal v2 split archives.

The 4TU landing page occasionally returns an HTTP-200 maintenance document for
file URLs. This downloader validates ZIP signatures and published MD5 digests
so such responses can never be mistaken for dataset content.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


DATASET_UUID = "b008dd74-020d-4ea4-a8ba-7bb60769d224"


@dataclass(frozen=True)
class Archive:
    split: str
    name: str
    file_uuid: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return f"https://data.4tu.nl/file/{DATASET_UUID}/{self.file_uuid}"


ARCHIVES = (
    Archive("train", "train_p1.zip", "f58a5f18-9a71-4443-8a48-e59530df9ee8", 6_371_363_172, "b283383ee97cc92bd3410c09c3c1b4bc"),
    Archive("train", "train_p2.zip", "5aae5141-cece-41db-9681-47acd5300705", 4_801_035_968, "0d07877b0755b226c28b8c5952d6c0dd"),
    Archive("train", "train_p3.zip", "4e2ed830-a1fe-40fa-9c5a-2aeae3964616", 4_414_811_027, "606509e026f501240595ff9805e9a662"),
    Archive("train", "train_p4.zip", "3ff5409e-165f-4354-9fa4-a4fde639a7a6", 4_837_382_030, "40193a8fee14fad551ff6ed7b140c29f"),
    Archive("val", "val_p1.zip", "bb336949-248c-4ae6-82ef-107dbe61d10f", 3_783_132_550, "a6d9464b6bbe5b7c4c2c05927ad95844"),
    Archive("val", "val_p2.zip", "e899de9f-313a-482b-a6b5-e2de4f38412e", 6_222_333_683, "0db912bd9565084e77090adabf78b7a4"),
    Archive("test", "test_p1.zip", "ec7a4f94-7c1d-497d-b75a-4e694b7a9f7e", 7_535_688_635, "04f53405ca41d85681ef9323c588d271"),
    Archive("test", "test_p2.zip", "d07789b6-83b1-429d-be0c-95edba210757", 5_952_711_048, "c76b3ca8a03f2c67764cb9c16eff358d"),
    Archive("test", "test_p3.zip", "31f88cb6-a2e7-44df-ac51-5aaa0a148fc3", 10_236_464_186, "7a81827d26293520d3da87ff77c6abd4"),
)


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path, archive: Archive, verify_md5: bool = True) -> None:
    actual_size = path.stat().st_size
    if actual_size != archive.size:
        prefix = path.read_bytes()[:4]
        kind = "HTML/maintenance response" if prefix.startswith(b"<!DO") else "partial file"
        raise RuntimeError(
            f"{archive.name}: expected {archive.size} bytes, got {actual_size} ({kind})"
        )
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"{archive.name}: payload is not a ZIP archive")
    if verify_md5:
        actual_md5 = md5sum(path)
        if actual_md5 != archive.md5:
            raise RuntimeError(
                f"{archive.name}: MD5 mismatch (expected {archive.md5}, got {actual_md5})"
            )


def download_archive(archive: Archive, archive_dir: Path, retries: int) -> Path:
    destination = archive_dir / archive.name
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        validate_archive(destination, archive)
        print(f"verified {destination}", flush=True)
        return destination

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "AIOps-IndustReal/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(archive.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                content_type = response.headers.get_content_type()
                if content_type == "text/html":
                    raise RuntimeError("4TU returned an HTML maintenance page")
                if offset and status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            if partial.stat().st_size == archive.size:
                partial.replace(destination)
                validate_archive(destination, archive)
                print(f"downloaded and verified {destination}", flush=True)
                return destination
            raise RuntimeError(
                f"incomplete response: {partial.stat().st_size}/{archive.size} bytes"
            )
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(f"failed to download {archive.name}: {exc}") from exc
            delay = min(60, 5 * attempt)
            print(f"{archive.name}: attempt {attempt}/{retries} failed: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def safe_extract(path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if not destination.is_relative_to(output_root):
                raise RuntimeError(f"unsafe archive member in {path.name}: {member.filename}")
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("data/raw/industreal_archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/industreal"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    if args.retries <= 0:
        parser.error("--retries must be positive")

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [archive for archive in ARCHIVES if archive.split in set(args.splits)]
    for archive in selected:
        path = download_archive(archive, args.archive_dir, args.retries)
        if not args.no_extract:
            safe_extract(path, args.output_dir)
            print(f"extracted {path.name} into {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
