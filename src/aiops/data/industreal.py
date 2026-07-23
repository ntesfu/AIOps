from __future__ import annotations

import csv
import json
import re
from dataclasses import replace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


OUTCOME_NAMES = ("background", "correct", "incorrect", "remove")
STATE_NAMES = ("incorrect", "not_completed", "correct")


@dataclass(frozen=True)
class IndustRealRecording:
    recording_id: str
    split: str
    root: Path
    rgb_dir: Path | None
    video_path: Path | None
    ar_labels: Path | None
    psr_labels: Path | None
    psr_raw_labels: Path | None
    object_labels: Path | None
    hands: Path | None
    gaze: Path | None
    pose: Path | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


@dataclass(frozen=True)
class CompletionEvent:
    frame_index: int
    step_id: str
    description: str
    outcome: str


@dataclass(frozen=True)
class ActionSegment:
    start_frame: int
    end_frame: int
    action_id: str
    description: str


@dataclass(frozen=True)
class IndustRealAudit:
    root: Path
    official_layout: bool
    recording_count: int
    split_counts: dict[str, int]
    labeled_recordings: int
    video_only_recordings: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "official_layout": self.official_layout,
            "recording_count": self.recording_count,
            "split_counts": self.split_counts,
            "labeled_recordings": self.labeled_recordings,
            "video_only_recordings": self.video_only_recordings,
            "warnings": list(self.warnings),
        }


def discover_industreal_recordings(root: str | Path) -> list[IndustRealRecording]:
    root_path = Path(root).expanduser().resolve()
    recordings_root = root_path / "recordings"
    if recordings_root.is_dir():
        recordings: list[IndustRealRecording] = []
        for split_dir in sorted(path for path in recordings_root.iterdir() if path.is_dir()):
            for recording_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                recordings.append(_official_recording(recording_dir, split_dir.name))
        return recordings

    # Some archives contain a single outer folder.
    children = [path for path in root_path.iterdir() if path.is_dir()] if root_path.is_dir() else []
    for child in children:
        if (child / "recordings").is_dir():
            return discover_industreal_recordings(child)

    # Fallback for custom HoloLens bundles. Labels must be supplied through a
    # cache/manifest before training; these rows are useful for auditing.
    video_files = sorted(
        path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi"}
    ) if root_path.exists() else []
    return [
        IndustRealRecording(
            recording_id=video.stem,
            split="unassigned",
            root=video.parent,
            rgb_dir=None,
            video_path=video,
            ar_labels=None,
            psr_labels=None,
            psr_raw_labels=None,
            object_labels=None,
            hands=None,
            gaze=None,
            pose=None,
        )
        for video in video_files
    ]


def attach_industreal_auxiliary(
    recordings: Iterable[IndustRealRecording], auxiliary_root: str | Path
) -> list[IndustRealRecording]:
    """Attach selectively extracted release metadata without duplicating RGB."""

    root = Path(auxiliary_root).expanduser().resolve()
    base = root / "recordings" if (root / "recordings").is_dir() else root
    attached: list[IndustRealRecording] = []
    for recording in recordings:
        directory = base / recording.split / recording.recording_id
        attached.append(
            replace(
                recording,
                object_labels=recording.object_labels
                or _first_existing(directory, ("OD_labels.json",)),
                hands=recording.hands or _first_existing(directory, ("hands.csv",)),
                gaze=recording.gaze or _first_existing(directory, ("gaze.csv",)),
                pose=recording.pose or _first_existing(directory, ("pose.csv",)),
            )
        )
    return attached


def audit_industreal_root(root: str | Path) -> IndustRealAudit:
    root_path = Path(root).expanduser().resolve()
    recordings = discover_industreal_recordings(root_path)
    split_counts: dict[str, int] = {}
    labeled = 0
    video_only = 0
    for recording in recordings:
        split_counts[recording.split] = split_counts.get(recording.split, 0) + 1
        if recording.psr_labels or recording.ar_labels:
            labeled += 1
        if recording.video_path is not None and recording.psr_labels is None:
            video_only += 1
    official = any(
        recording.split != "unassigned"
        and (recording.rgb_dir is not None or recording.video_path is not None)
        for recording in recordings
    )
    warnings: list[str] = []
    if not recordings:
        warnings.append("No recordings were found.")
    if not official and video_only:
        warnings.append(
            "Video files were found, but the official IndustReal recordings/split/annotation tree was not. "
            "Create a StateGraph cache from a temporal manifest or point --data-root at the extracted official release."
        )
    if official and labeled < len(recordings):
        warnings.append("Some official-layout recordings are missing AR/PSR annotations.")
    return IndustRealAudit(
        root=root_path,
        official_layout=official,
        recording_count=len(recordings),
        split_counts=split_counts,
        labeled_recordings=labeled,
        video_only_recordings=video_only,
        warnings=tuple(warnings),
    )


def read_completion_events(path: str | Path) -> list[CompletionEvent]:
    rows = _read_csv(Path(path))
    events: list[CompletionEvent] = []
    for ordinal, row in enumerate(rows):
        frame_value = _first_value(
            row,
            ("image", "image_name", "frame", "frame_id", "filename", "completed_frame", "column_0"),
        )
        frame_index = _parse_frame_index(frame_value, fallback=ordinal)
        step_value = _first_value(
            row,
            ("completed_step_id", "step_id", "action_id", "id", "label", "step", "column_1"),
        )
        description = str(
            _first_value(
                row,
                ("description", "step_description", "action_description", "name", "label", "column_2"),
            )
            or step_value
        )
        step_id = str(step_value if step_value not in (None, "") else description).strip()
        if not step_id:
            continue
        events.append(
            CompletionEvent(
                frame_index=frame_index,
                step_id=step_id,
                description=description.strip(),
                outcome=infer_outcome(description),
            )
        )
    return sorted(events, key=lambda event: event.frame_index)


def read_action_segments(path: str | Path) -> list[ActionSegment]:
    """Read dense, point-wise, or start/end action annotations.

    Point-wise annotations are returned as one-frame segments. The densifier
    below carries each label forward until the next annotated frame, which is
    also correct for datasets that export one row per video frame.
    """
    rows = _read_csv(Path(path))
    segments: list[ActionSegment] = []
    for ordinal, row in enumerate(rows):
        start_value = _first_value(
            row,
            (
                "start_frame",
                "start frame",
                "start",
                "frame_start",
                "column_3",
                "image",
                "image_name",
                "frame",
                "frame_id",
                "column_0",
            ),
        )
        end_value = _first_value(
            row, ("end_frame", "end frame", "end", "stop_frame", "frame_end", "column_4")
        )
        action_value = _first_value(
            row,
            ("action_id", "activity_id", "class_id", "step_id", "label", "action", "column_1"),
        )
        description = str(
            _first_value(
                row,
                ("description", "action_description", "activity", "name", "label", "column_2"),
            )
            or action_value
            or ""
        ).strip()
        action_id = str(action_value if action_value not in (None, "") else description).strip()
        if not action_id:
            continue
        start = _parse_frame_index(start_value, fallback=ordinal)
        end = _parse_frame_index(end_value, fallback=start) if end_value not in (None, "") else start
        if end < start:
            start, end = end, start
        segments.append(ActionSegment(start, end, action_id, description))
    return sorted(segments, key=lambda item: (item.start_frame, item.end_frame, item.action_id))


def read_raw_states(path: str | Path, num_components: int = 11) -> dict[int, np.ndarray]:
    rows = _read_csv(Path(path))
    states: dict[int, np.ndarray] = {}
    for ordinal, row in enumerate(rows):
        frame_value = _first_value(
            row, ("image", "image_name", "frame", "frame_id", "filename", "column_0")
        )
        frame_index = _parse_frame_index(frame_value, fallback=ordinal)
        values: list[int] = []
        for key, value in row.items():
            if key.lower() in {
                "image",
                "image_name",
                "frame",
                "frame_id",
                "filename",
                "recording",
                "column_0",
            }:
                continue
            try:
                numeric = int(float(str(value).strip()))
            except (TypeError, ValueError):
                continue
            if numeric in {-1, 0, 1}:
                values.append(numeric)
        if len(values) >= num_components:
            # Convert -1/0/1 to CE indices 0/1/2.
            states[frame_index] = np.asarray(values[-num_components:], dtype=np.int64) + 1
    return states


def read_numeric_timeseries(path: str | Path) -> dict[int, np.ndarray]:
    rows = _read_csv(Path(path))
    result: dict[int, np.ndarray] = {}
    for ordinal, row in enumerate(rows):
        frame_value = _first_value(
            row, ("image", "image_name", "frame", "frame_id", "filename", "column_0")
        )
        frame_index = _parse_frame_index(frame_value, fallback=ordinal)
        values: list[float] = []
        for key, value in row.items():
            if key.lower() in {
                "image",
                "image_name",
                "frame",
                "frame_id",
                "filename",
                "recording",
                "column_0",
            }:
                continue
            try:
                values.append(float(str(value).strip()))
            except (TypeError, ValueError):
                continue
        if values:
            result[frame_index] = np.asarray(values, dtype=np.float32)
    return result


def dense_labels_from_events(
    frame_indices: Iterable[int],
    events: list[CompletionEvent],
    step_to_index: dict[str, int],
    mode: str = "active_until_next",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(list(frame_indices), dtype=np.int64)
    step = np.full(frames.shape, -100, dtype=np.int64)
    outcome = np.full(frames.shape, -100, dtype=np.int64)
    boundary = np.zeros(frames.shape, dtype=np.float32)
    if not events or frames.size == 0:
        return step, outcome, boundary
    if mode not in {"active_until_next", "state_after_completion"}:
        raise ValueError(f"Unsupported completion labeling mode: {mode}")
    outcome_to_index = {name: index for index, name in enumerate(OUTCOME_NAMES)}
    for index, event in enumerate(events):
        if event.step_id not in step_to_index:
            continue
        previous_frame = events[index - 1].frame_index if index > 0 else int(frames.min()) - 1
        next_frame = events[index + 1].frame_index if index + 1 < len(events) else int(frames.max()) + 1
        if mode == "active_until_next":
            selected = (frames > previous_frame) & (frames <= event.frame_index)
        else:
            selected = (frames >= event.frame_index) & (frames < next_frame)
        step[selected] = step_to_index[event.step_id]
        outcome[selected] = outcome_to_index.get(event.outcome, outcome_to_index["background"])
        nearest = int(np.argmin(np.abs(frames - event.frame_index)))
        boundary[nearest] = 1.0
    return step, outcome, boundary


def dense_action_labels(
    frame_indices: Iterable[int],
    segments: list[ActionSegment],
    action_to_index: dict[str, int],
    background_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert action annotations into a mutually-exclusive segmentation target."""
    frames = np.asarray(list(frame_indices), dtype=np.int64)
    fill_value = -100 if background_index is None else int(background_index)
    action = np.full(frames.shape, fill_value, dtype=np.int64)
    if not segments or frames.size == 0:
        return action, np.zeros(frames.shape, dtype=np.float32)
    point_annotations = all(segment.start_frame == segment.end_frame for segment in segments)
    if point_annotations:
        for index, segment in enumerate(segments):
            end = segments[index + 1].start_frame if index + 1 < len(segments) else int(frames.max()) + 1
            selected = (frames >= segment.start_frame) & (frames < end)
            if segment.action_id in action_to_index:
                action[selected] = action_to_index[segment.action_id]
    else:
        for segment in segments:
            if segment.action_id in action_to_index:
                action[(frames >= segment.start_frame) & (frames <= segment.end_frame)] = action_to_index[
                    segment.action_id
                ]
    boundary = np.zeros(frames.shape, dtype=np.float32)
    valid = action >= 0
    changes = np.flatnonzero(valid & np.r_[True, action[1:] != action[:-1]])
    boundary[changes] = 1.0
    return action, boundary


def multi_component_targets_from_events(
    frame_indices: Iterable[int],
    events: list[CompletionEvent],
    resolve_component,
    num_components: int,
    event_outcomes: tuple[str, ...] = ("correct", "incorrect", "remove"),
) -> tuple[np.ndarray, np.ndarray]:
    """Create collision-safe multi-label completion and per-part outcome targets."""
    frames = np.asarray(list(frame_indices), dtype=np.int64)
    completion = np.zeros((len(frames), num_components), dtype=np.float32)
    component_outcome = np.full((len(frames), num_components), -100, dtype=np.int64)
    outcome_to_index = {name: index for index, name in enumerate(event_outcomes)}
    severity = {"correct": 0, "remove": 1, "incorrect": 2}
    for event in events:
        if frames.size == 0:
            break
        component = int(resolve_component(event.step_id, event.description))
        if not 0 <= component < num_components:
            raise ValueError(f"Resolved completion component {component} is outside [0, {num_components}).")
        row = int(np.argmin(np.abs(frames - event.frame_index)))
        outcome = outcome_to_index[event.outcome]
        previous = int(component_outcome[row, component])
        if previous < 0 or severity[event.outcome] > severity[event_outcomes[previous]]:
            component_outcome[row, component] = outcome
        completion[row, component] = 1.0
    return completion, component_outcome


def infer_outcome(description: str) -> str:
    text = description.lower()
    if any(token in text for token in ("incorrect", "wrong", "error", "fault", "missing")):
        return "incorrect"
    if any(token in text for token in ("remove", "detach", "disassemble", "uninstall")):
        return "remove"
    if any(token in text for token in ("background", "idle", "none")):
        return "background"
    return "correct"


def write_audit(root: str | Path, output: str | Path) -> IndustRealAudit:
    audit = audit_industreal_root(root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    return audit


def _official_recording(recording_dir: Path, split: str) -> IndustRealRecording:
    return IndustRealRecording(
        recording_id=recording_dir.name,
        split=split,
        root=recording_dir,
        rgb_dir=_optional_dir(recording_dir / "rgb"),
        video_path=_find_recording_video(recording_dir),
        ar_labels=_first_existing(recording_dir, ("AR_labels.csv", "ar_labels.csv")),
        psr_labels=_first_existing(recording_dir, ("PSR_labels_with_errors.csv", "PSR_labels.csv")),
        psr_raw_labels=_first_existing(recording_dir, ("PSR_labels_raw.csv",)),
        object_labels=_first_existing(recording_dir, ("OD_labels.json",)),
        hands=_first_existing(recording_dir, ("hands.csv",)),
        gaze=_first_existing(recording_dir, ("gaze.csv",)),
        pose=_first_existing(recording_dir, ("pose.csv",)),
    )


def _optional_dir(path: Path) -> Path | None:
    return path if path.is_dir() else None


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _find_recording_video(recording_dir: Path) -> Path | None:
    # The public release commonly keeps annotations in
    # recordings/{split}/{id} while placing {id}.mp4 at the dataset root.
    search_roots = [recording_dir]
    if len(recording_dir.parents) >= 3:
        search_roots.append(recording_dir.parents[2])
    for root in search_roots:
        for suffix in (".mp4", ".mov", ".avi"):
            candidate = root / f"{recording_dir.name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect=dialect)
        raw_rows = [[str(value).strip() for value in row] for row in reader if row]
        if not raw_rows:
            return []
        known_headers = {
            "image",
            "image_name",
            "frame",
            "frame_id",
            "filename",
            "completed_frame",
            "completed_step_id",
            "step_id",
            "action_id",
            "activity_id",
            "class_id",
            "start",
            "start_frame",
            "end",
            "end_frame",
            "description",
            "step_description",
            "recording",
        }
        first = [value.lower().strip() for value in raw_rows[0]]
        has_header = any(value in known_headers for value in first)
        if has_header:
            fieldnames = [value or f"column_{index}" for index, value in enumerate(raw_rows[0])]
            data_rows = raw_rows[1:]
        else:
            fieldnames = [f"column_{index}" for index in range(max(map(len, raw_rows)))]
            data_rows = raw_rows
        return [
            {
                fieldnames[index]: value
                for index, value in enumerate(row)
                if index < len(fieldnames)
            }
            for row in data_rows
        ]
    return []


def _first_value(row: dict[str, str], aliases: Iterable[str]) -> str | None:
    normalized = {key.lower().strip(): value for key, value in row.items()}
    for alias in aliases:
        if alias in normalized and normalized[alias] not in (None, ""):
            return normalized[alias]
    return None


def _parse_frame_index(value: object, fallback: int) -> int:
    if value is None:
        return fallback
    text = str(value)
    matches = re.findall(r"\d+", Path(text).stem)
    if matches:
        return int(matches[-1])
    try:
        return int(float(text))
    except ValueError:
        return fallback


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit an extracted IndustReal or HoloLens dataset root.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    audit = audit_industreal_root(args.data_root)
    if args.output:
        write_audit(args.data_root, args.output)
    print(json.dumps(audit.to_dict(), indent=2))


if __name__ == "__main__":
    main()
