"""CaptainCook4D loader: step segments, error labels, and official splits.

CaptainCook4D (Peddi et al., NeurIPS 2024) is an egocentric procedural-activity
dataset with dense, real execution errors — 5.7K step segments, ~2.5K error
instances across eight categories.  We use it as an *architecture-validation*
corpus for the error-first StateVerify line: unlike IndustReal (19 real error
events), it supplies enough held-out errors to separate a data-scarcity failure
from a method failure.

Two annotation files fully describe every recording and align positionally:

* ``complete_step_annotations.json`` — dict ``recording_id -> {activity_id,
  activity_name, person_id, environment, steps:[{step_id, start_time, end_time,
  description, has_errors}]}``.
* ``error_annotations.json`` — list of ``{recording_id, is_error,
  step_annotations:[{step_id, start_time, end_time, errors:[{tag, description}]}]}``.
  The per-step ``errors`` list carries the fine-grained category tags.

The two files list steps in the same order, so we join them by position.

**Missing-Step** errors are encoded with ``start_time == end_time == -1``: the
step was never performed, so there is no observable video segment.  Those are a
step-*absence* signal (procedure-residual territory), not a visual anomaly, so
``load_segments`` drops them by default and flags them via ``is_missing``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Canonical order from annotations/annotation_csv/error_category_idx.csv.
ERROR_CATEGORIES: tuple[str, ...] = (
    "Preparation Error",
    "Measurement Error",
    "Order Error",
    "Timing Error",
    "Technique Error",
    "Temperature Error",
    "Missing Step",
    "Other",
)

SPLIT_FILES: dict[str, str] = {
    "person": "person_data_split_combined.json",
    "recordings": "recordings_data_split_combined.json",
    "environment": "environment_data_split_combined.json",
    "recipes": "recipes_data_split_combined.json",
}

MISSING_TIME = -1.0


@dataclass(frozen=True)
class CaptainCook4DSegment:
    """One step segment (the unit of the error-recognition benchmark)."""

    recording_id: str
    activity_id: int
    activity_name: str
    person_id: str
    environment: str
    step_index: int  # position within the recording's step list
    step_id: int
    start_time: float
    end_time: float
    description: str
    has_errors: bool
    error_tags: tuple[str, ...]
    is_missing: bool  # start_time == -1 (skipped step, no observable segment)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def label(self) -> int:
        return 1 if self.has_errors else 0


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_segments(
    annotations_root: str | Path,
    *,
    include_missing: bool = False,
) -> list[CaptainCook4DSegment]:
    """Return every step segment with its error label and metadata.

    ``annotations_root`` is the cloned ``annotations`` directory (contains
    ``annotation_json/``).  Segments keep their in-recording order.  Skipped
    (missing) steps are excluded unless ``include_missing`` is set.
    """
    root = Path(annotations_root)
    json_dir = root / "annotation_json"
    complete = _load_json(json_dir / "complete_step_annotations.json")
    errors = _load_json(json_dir / "error_annotations.json")
    if not isinstance(complete, dict):
        raise ValueError("complete_step_annotations.json must be a dict")
    error_by_id = {row["recording_id"]: row for row in errors}

    segments: list[CaptainCook4DSegment] = []
    for recording_id, meta in complete.items():
        steps = meta["steps"]
        err_steps = error_by_id.get(recording_id, {}).get("step_annotations", [])
        for index, step in enumerate(steps):
            tags: tuple[str, ...] = ()
            if index < len(err_steps):
                tags = tuple(
                    tag["tag"] for tag in (err_steps[index].get("errors") or [])
                )
            is_missing = float(step["start_time"]) == MISSING_TIME
            if is_missing and not include_missing:
                continue
            segments.append(
                CaptainCook4DSegment(
                    recording_id=recording_id,
                    activity_id=int(meta["activity_id"]),
                    activity_name=str(meta.get("activity_name", "")),
                    person_id=str(meta["person_id"]),
                    environment=str(meta["environment"]),
                    step_index=index,
                    step_id=int(step["step_id"]),
                    start_time=float(step["start_time"]),
                    end_time=float(step["end_time"]),
                    description=str(step.get("description", "")),
                    has_errors=bool(step["has_errors"]),
                    error_tags=tags,
                    is_missing=is_missing,
                )
            )
    return segments


def load_recording_split(
    annotations_root: str | Path,
    split_by: str = "person",
) -> dict[str, set[str]]:
    """Return ``{"train"/"val"/"test": {recording_id}}`` for an official split.

    ``split_by`` is one of ``person``, ``recordings``, ``environment``,
    ``recipes`` (the ``combined`` variants that include error recordings).
    ``person`` is the participant-disjoint split — the direct analogue of the
    IndustReal operator-disjoint screen.
    """
    if split_by not in SPLIT_FILES:
        raise ValueError(f"split_by must be one of {sorted(SPLIT_FILES)}")
    path = Path(annotations_root) / "data_splits" / SPLIT_FILES[split_by]
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a dict of split -> recording ids")
    return {name: set(ids) for name, ids in payload.items()}


def filter_segments(
    segments: Iterable[CaptainCook4DSegment],
    recording_ids: set[str],
) -> list[CaptainCook4DSegment]:
    return [seg for seg in segments if seg.recording_id in recording_ids]


def video_path(data_root: str | Path, recording_id: str) -> Path:
    """Path to the downloaded GoPro 360p clip for a recording."""
    return (
        Path(data_root)
        / "captain_cook_4d"
        / "gopro"
        / "resolution_360p"
        / f"{recording_id}_360p.mp4"
    )
