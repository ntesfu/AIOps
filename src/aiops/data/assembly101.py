from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_RECORDING_PATTERN = re.compile(
    r"^nusar-2021_action_both_(?P<actor>\d+)-(?P<toy>[a-z]\d+[a-z]?)_"
)


@dataclass(frozen=True)
class AssemblySegment:
    """One official Assembly101 part-to-part mistake annotation.

    The public files contain frame indices at 30 fps and intentionally have no
    header.  ``subject`` and ``reference`` are the two interacting toy parts.
    """

    recording_id: str
    start_frame: int
    end_frame: int
    verb: str
    subject: str
    reference: str
    label: str
    remark: str = ""

    @property
    def action_id(self) -> str:
        # This is the paper's coarse, visually recognizable action level.  The
        # second working object remains available for state-graph topology but
        # is not folded into a 358-way long-tailed classifier.
        return f"{self.verb}:{self.subject}"

    @property
    def action_description(self) -> str:
        return f"{self.verb} {self.subject}"

    @property
    def outcome_index(self) -> int:
        # Canonical StateGraph outcomes are [correct, incorrect, remove].  A
        # correction attachment restores a valid state; a correction detach
        # removes the erroneous attachment.  Mistakes remain mistakes even if
        # their physical operation is a detach.
        if self.label == "mistake":
            return 1
        if self.label == "correction" and self.verb == "detach":
            return 2
        return 0


@dataclass(frozen=True)
class CoarseActionSegment:
    """One complete action interval from the official coarse annotations."""

    start_frame: int
    end_frame: int
    action: str


def read_coarse_action_segments(paths: Iterable[str | Path]) -> list[CoarseActionSegment]:
    segments: list[CoarseActionSegment] = []
    for source in paths:
        path = Path(source)
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, 1):
                values = line.strip().split("\t")
                if not values or all(not value.strip() for value in values):
                    continue
                if len(values) < 3:
                    raise ValueError(f"{path}:{row_number} has fewer than three columns")
                start, end = int(values[0]), int(values[1])
                if end <= start:
                    raise ValueError(f"{path}:{row_number} has a non-positive interval")
                segments.append(CoarseActionSegment(start, end, _normalize(values[2])))
    return sorted(segments, key=lambda item: (item.start_frame, item.end_frame, item.action))


def parse_recording_identity(recording_id: str) -> tuple[str, str]:
    match = _RECORDING_PATTERN.match(recording_id)
    if match is None:
        raise ValueError(f"Unrecognized Assembly101 recording ID: {recording_id!r}")
    return match.group("actor"), match.group("toy")


def read_mistake_segments(path: str | Path) -> list[AssemblySegment]:
    """Read an official headerless ``assembly101-mistake-detection`` CSV."""

    annotation_path = Path(path)
    recording_id = annotation_path.stem
    segments: list[AssemblySegment] = []
    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 6:
                raise ValueError(
                    f"{annotation_path}:{row_number} has {len(row)} columns; expected at least 6"
                )
            try:
                start_frame, end_frame = int(row[0]), int(row[1])
            except ValueError as exc:
                raise ValueError(
                    f"{annotation_path}:{row_number} has non-integer frame bounds"
                ) from exc
            if end_frame <= start_frame:
                # One upstream file contains a transposed end timestamp.  Do
                # not silently make a negative segment; retain a one-frame
                # event so the anomaly is represented and record-level order
                # remains causal.
                end_frame = start_frame + 1
            verb, subject, reference, label = (
                _normalize(value) for value in row[2:6]
            )
            if label not in {"correct", "mistake", "correction"}:
                raise ValueError(
                    f"{annotation_path}:{row_number} has unknown label {label!r}"
                )
            # One official row has ``interior`` in the verb column for a
            # correction immediately following an erroneous attachment.  All
            # other removal corrections use ``detach``; repair that documented
            # upstream typo deterministically.
            if label == "correction" and verb not in {"attach", "detach"}:
                verb = "detach"
            segments.append(
                AssemblySegment(
                    recording_id=recording_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    verb=verb,
                    subject=subject,
                    reference=reference,
                    label=label,
                    remark=_normalize(" ".join(row[6:])),
                )
            )
    return sorted(segments, key=lambda item: (item.start_frame, item.end_frame))


def dense_assembly_targets(
    sampled_frames: np.ndarray,
    segments: Iterable[AssemblySegment],
    action_to_index: dict[str, int],
    component_to_index: dict[str, int],
    *,
    background_index: int,
    action_segments: Iterable[CoarseActionSegment] | None = None,
) -> dict[str, np.ndarray]:
    """Project sparse part-to-part annotations onto sampled visual frames.

    Completion/outcome supervision is placed once, at the final sampled row of
    each action. Persistent component state is then carried forward causally.
    """

    frames = np.asarray(sampled_frames, dtype=np.int64)
    if frames.ndim != 1 or frames.size == 0:
        raise ValueError("sampled_frames must be a non-empty one-dimensional array")
    ordered = list(segments)
    length = len(frames)
    components = len(component_to_index)
    step = np.full(length, background_index, dtype=np.int64)
    boundary = np.zeros(length, dtype=np.float32)
    completion = np.zeros((length, components), dtype=np.float32)
    outcome = np.full((length, components), -100, dtype=np.int64)
    state = np.ones((length, components), dtype=np.int64)
    # Global component vocabulary spans many different toys. Components absent
    # from this recording are unknown, not "not completed"; supervising them
    # as pending overwhelms the rare incorrect/correct states and teaches toy
    # identity instead of assembly state.
    active_components = {component_to_index[segment.subject] for segment in ordered}
    state_mask = np.zeros((length, components), dtype=np.bool_)
    if active_components:
        state_mask[:, sorted(active_components)] = True
    current_state = np.ones(components, dtype=np.int64)
    cursor = 0

    complete_actions = list(action_segments) if action_segments is not None else None
    if complete_actions is not None:
        for action in complete_actions:
            inside = np.flatnonzero((frames >= action.start_frame) & (frames < action.end_frame))
            if inside.size:
                step[inside] = action_to_index[action.action]
                boundary[int(inside[0])] = 1.0

    for segment in ordered:
        while cursor < length and frames[cursor] < segment.start_frame:
            state[cursor] = current_state
            cursor += 1
        inside = np.flatnonzero(
            (frames >= segment.start_frame) & (frames < segment.end_frame)
        )
        if inside.size and complete_actions is None:
            step[inside] = action_to_index[segment.action_id]
            boundary[int(inside[0])] = 1.0
            event_row = int(inside[-1])
        elif inside.size:
            event_row = int(inside[-1])
        else:
            event_row = int(np.argmin(np.abs(frames - (segment.end_frame - 1))))
        component = component_to_index[segment.subject]
        # A handful of sub-second correction actions collapse onto the same
        # 1 fps mirror frame as the preceding mistake. Preserve both ordered
        # events by assigning the later one to the next free causal row.
        while event_row + 1 < length and outcome[event_row, component] >= 0:
            event_row += 1
        completion[event_row, component] = 1.0
        outcome[event_row, component] = segment.outcome_index

        while cursor <= event_row and cursor < length:
            state[cursor] = current_state
            cursor += 1
        if segment.verb == "detach":
            current_state[component] = 1
        elif segment.label == "mistake":
            current_state[component] = 0
        else:
            current_state[component] = 2
        if event_row < length:
            state[event_row] = current_state

    while cursor < length:
        state[cursor] = current_state
        cursor += 1
    return {
        "step": step,
        "boundary": boundary,
        "completion": completion,
        "component_outcome": outcome,
        "state": state,
        "state_mask": state_mask,
    }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
