from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssemblySegment:
    video_id: str
    step_id: str
    label: str
    start_s: float
    end_s: float


def read_segments_csv(path: str | Path) -> list[AssemblySegment]:
    """Read a normalized Assembly101-style CSV.

    Expected columns: video_id, step_id, label, start_s, end_s.
    Dataset-specific raw annotation conversion will be added after the
    downloaded annotation layout is available locally.
    """

    segments: list[AssemblySegment] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            segments.append(
                AssemblySegment(
                    video_id=row["video_id"],
                    step_id=row["step_id"],
                    label=row["label"],
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                )
            )
    return segments

