from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemporalSegment:
    step_id: str
    label: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TemporalVideo:
    video_path: Path
    segments: tuple[TemporalSegment, ...]
    dataset: str = "unknown"
    procedure: str = "unknown"


def read_temporal_manifest(path: str | Path) -> list[TemporalVideo]:
    rows: list[TemporalVideo] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                TemporalVideo(
                    video_path=Path(payload["video_path"]),
                    dataset=payload.get("dataset", "unknown"),
                    procedure=payload.get("procedure", "unknown"),
                    segments=tuple(
                        TemporalSegment(
                            step_id=segment["step_id"],
                            label=segment.get("label", segment["step_id"]),
                            start_s=float(segment["start_s"]),
                            end_s=float(segment["end_s"]),
                        )
                        for segment in payload["segments"]
                    ),
                )
            )
    return rows


def write_temporal_manifest(rows: list[TemporalVideo], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload: dict[str, Any] = {
                "video_path": str(row.video_path),
                "dataset": row.dataset,
                "procedure": row.procedure,
                "segments": [
                    {
                        "step_id": segment.step_id,
                        "label": segment.label,
                        "start_s": segment.start_s,
                        "end_s": segment.end_s,
                    }
                    for segment in row.segments
                ],
            }
            handle.write(json.dumps(payload) + "\n")

