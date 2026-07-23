from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class StateGraphCacheRecord:
    recording_id: str
    split: str
    path: Path
    num_steps: int
    motion_dim: int
    appearance_dim: int
    sensor_dim: int
    num_components: int
    num_completion_components: int = 0
    # `num_steps` is the legacy name for the action-class count. Sequence
    # length varies by recording and is stored separately for tooling.
    num_frames: int = 0
    motion_aux_dim: int = 0


def read_cache_index(path: str | Path) -> tuple[dict[str, Any], list[StateGraphCacheRecord]]:
    index_path = Path(path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        StateGraphCacheRecord(
            recording_id=row["recording_id"],
            split=row["split"],
            path=(index_path.parent / row["path"]).resolve(),
            num_steps=int(row["num_steps"]),
            motion_dim=int(row["motion_dim"]),
            appearance_dim=int(row["appearance_dim"]),
            sensor_dim=int(row["sensor_dim"]),
            num_components=int(row.get("num_components", 11)),
            num_completion_components=int(row.get("num_completion_components", 0)),
            num_frames=int(row.get("num_frames", 0)),
            motion_aux_dim=int(row.get("motion_aux_dim", 0)),
        )
        for row in payload["records"]
    ]
    return payload.get("metadata", {}), records


def write_cache_index(
    path: str | Path,
    records: Iterable[StateGraphCacheRecord],
    metadata: dict[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        try:
            relative_path = record.path.resolve().relative_to(output_path.parent.resolve())
        except ValueError:
            relative_path = record.path.resolve()
        rows.append(
            {
                "recording_id": record.recording_id,
                "split": record.split,
                "path": str(relative_path),
                "num_steps": record.num_steps,
                "motion_dim": record.motion_dim,
                "appearance_dim": record.appearance_dim,
                "sensor_dim": record.sensor_dim,
                "num_components": record.num_components,
                "num_completion_components": record.num_completion_components,
                "num_frames": record.num_frames,
                "motion_aux_dim": record.motion_aux_dim,
            }
        )
    output_path.write_text(json.dumps({"metadata": metadata, "records": rows}, indent=2) + "\n", encoding="utf-8")


def save_cache_record(
    path: str | Path,
    *,
    motion: np.ndarray,
    appearance: np.ndarray,
    sensor: np.ndarray,
    modality_mask: np.ndarray,
    step: np.ndarray,
    completion: np.ndarray,
    component_outcome: np.ndarray,
    state: np.ndarray,
    state_mask: np.ndarray,
    boundary: np.ndarray,
    timestamps: np.ndarray,
    motion_aux: np.ndarray | None = None,
    prediction_frame_indices: np.ndarray | None = None,
    motion_source_frame_indices: np.ndarray | None = None,
) -> None:
    length = len(step)
    arrays = {
        "motion": np.asarray(motion, dtype=np.float32),
        "appearance": np.asarray(appearance, dtype=np.float32),
        "sensor": np.asarray(sensor, dtype=np.float32),
        "modality_mask": np.asarray(modality_mask, dtype=np.bool_),
        "step": np.asarray(step, dtype=np.int64),
        "completion": np.asarray(completion, dtype=np.float32),
        "component_outcome": np.asarray(component_outcome, dtype=np.int64),
        "state": np.asarray(state, dtype=np.int64),
        "state_mask": np.asarray(state_mask, dtype=np.bool_),
        "boundary": np.asarray(boundary, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float32),
    }
    if motion_aux is not None:
        arrays["motion_aux"] = np.asarray(motion_aux, dtype=np.float32)
    if prediction_frame_indices is not None:
        arrays["prediction_frame_indices"] = np.asarray(prediction_frame_indices, dtype=np.int64)
    if motion_source_frame_indices is not None:
        arrays["motion_source_frame_indices"] = np.asarray(
            motion_source_frame_indices, dtype=np.int64
        )
    for name, array in arrays.items():
        if array.shape[0] != length:
            raise ValueError(f"{name} has {array.shape[0]} rows, expected {length}")
    if arrays["step"].ndim != 1:
        raise ValueError(f"step must have shape [time], got {arrays['step'].shape}")
    if arrays["completion"].ndim != 2:
        raise ValueError(f"completion must have shape [time, component], got {arrays['completion'].shape}")
    if arrays["component_outcome"].shape != arrays["completion"].shape:
        raise ValueError("component_outcome must have the same shape as completion")
    if arrays["state"].shape != arrays["state_mask"].shape:
        raise ValueError("state_mask must have the same shape as state")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


class StateGraphCacheDataset:
    def __init__(
        self,
        records: list[StateGraphCacheRecord],
        sequence_length: int = 256,
        sequence_stride: int = 192,
        training: bool = False,
        seed: int = 7,
        preload: bool = False,
        event_centered_crops_per_event: int = 0,
        event_centered_crop_radius: int = 0,
    ) -> None:
        if sequence_length <= 0 or sequence_stride <= 0:
            raise ValueError("sequence_length and sequence_stride must be positive")
        if event_centered_crops_per_event < 0:
            raise ValueError("event_centered_crops_per_event cannot be negative")
        if event_centered_crop_radius < 0:
            raise ValueError("event_centered_crop_radius cannot be negative")
        if event_centered_crops_per_event and not training:
            raise ValueError("event-centered crops are training-only")
        if event_centered_crop_radius > sequence_length // 2:
            raise ValueError("event_centered_crop_radius cannot exceed half of sequence_length")
        self.records = records
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.training = training
        self.seed = seed
        self.epoch = 0
        self._preloaded: list[dict[str, np.ndarray]] | None = None
        if preload:
            self._preloaded = []
            for record in records:
                with np.load(record.path, allow_pickle=False) as arrays:
                    self._preloaded.append(
                        {name: arrays[name].copy() for name in arrays.files}
                    )
        self.windows: list[tuple[int, int]] = []
        self.incorrect_event_windows: set[int] = set()
        self.event_centered_window_indices: set[int] = set()
        self.incorrect_event_count = 0
        for record_index, record in enumerate(records):
            with self._open_record(record_index) as arrays:
                length = int(arrays["step"].shape[0])
            if length <= sequence_length:
                self.windows.append((record_index, 0))
            else:
                starts = list(range(0, max(1, length - sequence_length + 1), sequence_stride))
                final_start = length - sequence_length
                if not starts or starts[-1] != final_start:
                    starts.append(final_start)
                self.windows.extend((record_index, start) for start in starts)
            with self._open_record(record_index) as arrays:
                # Keep one entry per supervised component event. Multiple
                # incorrect components at the same timestamp are distinct
                # targets and intentionally receive distinct crop slots.
                event_rows = np.where(arrays["component_outcome"] == 1)[0]
            self.incorrect_event_count += len(event_rows)
            if event_centered_crops_per_event:
                if event_centered_crops_per_event == 1:
                    offsets = np.asarray([0], dtype=np.int64)
                else:
                    offsets = np.rint(
                        np.linspace(
                            -event_centered_crop_radius,
                            event_centered_crop_radius,
                            event_centered_crops_per_event,
                        )
                    ).astype(np.int64)
                for event_row in event_rows:
                    for offset in offsets:
                        start = int(
                            np.clip(
                                int(event_row) - sequence_length // 2 + int(offset),
                                0,
                                max(0, length - sequence_length),
                            )
                        )
                        window_index = len(self.windows)
                        self.windows.append((record_index, start))
                        self.event_centered_window_indices.add(window_index)
        for window_index, (record_index, start) in enumerate(self.windows):
            with self._open_record(record_index) as arrays:
                end = min(len(arrays["component_outcome"]), start + self.sequence_length)
                if (arrays["component_outcome"][start:end] == 1).any():
                    self.incorrect_event_windows.add(window_index)

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic but different temporal crop set each epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    @contextmanager
    def _open_record(self, record_index: int):
        if self._preloaded is not None:
            yield self._preloaded[record_index]
            return
        with np.load(self.records[record_index].path, allow_pickle=False) as arrays:
            yield arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start = self.windows[index]
        record = self.records[record_index]
        with self._open_record(record_index) as arrays:
            length = int(arrays["step"].shape[0])
            if (
                self.training
                and length > self.sequence_length
                and index not in self.incorrect_event_windows
            ):
                # Workers are recreated for each DataLoader iteration, so the
                # epoch set by the trainer is copied into every worker. Large
                # coprime multipliers prevent adjacent indices/epochs from
                # receiving correlated crops while retaining exact resume.
                rng = np.random.default_rng(
                    self.seed + 1_000_003 * self.epoch + 9_176 * index
                )
                jitter = int(rng.integers(-self.sequence_stride // 4, self.sequence_stride // 4 + 1))
                start = int(np.clip(start + jitter, 0, length - self.sequence_length))
            end = min(length, start + self.sequence_length)
            names = arrays.files if hasattr(arrays, "files") else arrays.keys()
            sample = {name: arrays[name][start:end].copy() for name in names}
        sample["recording_id"] = record.recording_id
        sample["start_index"] = start
        return sample

    def window_sampling_weights(
        self,
        incorrect_outcome_index: int = 1,
        rare_window_boost: float = 4.0,
    ) -> np.ndarray:
        """Return deterministic weights that expose scarce event windows more often."""
        if rare_window_boost < 1.0:
            raise ValueError("rare_window_boost must be at least 1")
        labels: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for record_index, record in enumerate(self.records):
            with self._open_record(record_index) as arrays:
                labels[record_index] = (
                    arrays["component_outcome"].copy(),
                    arrays["state"].copy(),
                    arrays["state_mask"].copy(),
                )
        weights = np.ones(len(self.windows), dtype=np.float64)
        for window_index, (record_index, start) in enumerate(self.windows):
            component_outcome, state, state_mask = labels[record_index]
            end = min(len(component_outcome), start + self.sequence_length)
            has_incorrect_outcome = bool(
                (component_outcome[start:end] == incorrect_outcome_index).any()
            )
            if has_incorrect_outcome:
                weights[window_index] = rare_window_boost
        return weights

    def rare_window_indices(self) -> list[int]:
        """Return windows containing a supervised incorrect completion event."""

        return sorted(self.incorrect_event_windows)

    def event_centered_indices(self) -> list[int]:
        """Return the synthetic, training-only crops centered near incorrect events."""

        return sorted(self.event_centered_window_indices)

    def matched_correct_window_candidates(
        self,
        incorrect_outcome_index: int = 1,
        correct_outcome_index: int = 0,
    ) -> dict[int, list[int]]:
        """Match rare windows to correct installs of the same component.

        Candidates with the same action at the labelled event are preferred,
        followed by cross-recording candidates.  The method uses training
        records only; validation records live in a separate dataset instance.
        """

        labels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for record_index in range(len(self.records)):
            with self._open_record(record_index) as arrays:
                labels[record_index] = (
                    arrays["component_outcome"].copy(),
                    arrays["step"].copy(),
                )

        correct: dict[int, list[tuple[int, int, int]]] = {}
        incorrect: dict[int, set[tuple[int, int]]] = {}
        for window_index, (record_index, start) in enumerate(self.windows):
            outcomes, steps = labels[record_index]
            end = min(len(outcomes), start + self.sequence_length)
            window_outcomes = outcomes[start:end]
            for row_offset, component in np.argwhere(
                window_outcomes == correct_outcome_index
            ):
                action = int(steps[start + int(row_offset)])
                correct.setdefault(int(component), []).append(
                    (window_index, action, record_index)
                )
            if window_index in self.incorrect_event_windows:
                signatures = {
                    (int(component), int(steps[start + int(row_offset)]))
                    for row_offset, component in np.argwhere(
                        window_outcomes == incorrect_outcome_index
                    )
                }
                if signatures:
                    incorrect[window_index] = signatures

        matches: dict[int, list[int]] = {}
        for rare_index, signatures in incorrect.items():
            rare_record = self.windows[rare_index][0]
            ranked: dict[int, tuple[int, int, int]] = {}
            for component, action in signatures:
                for candidate, candidate_action, candidate_record in correct.get(
                    component, ()
                ):
                    key = (
                        candidate_action != action,
                        candidate_record == rare_record,
                        candidate,
                    )
                    if candidate not in ranked or key < ranked[candidate]:
                        ranked[candidate] = key
            if ranked:
                matches[rare_index] = sorted(ranked, key=ranked.__getitem__)
        return matches


def pad_stategraph_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for batching cached StateGraph features.") from exc
    max_length = max(len(sample["step"]) for sample in samples)

    def pad_array(name: str, value: float | int | bool) -> np.ndarray:
        rows = []
        for sample in samples:
            array = sample[name]
            width = max_length - len(array)
            pad_width = [(0, width)] + [(0, 0)] * (array.ndim - 1)
            rows.append(np.pad(array, pad_width, mode="constant", constant_values=value))
        return np.stack(rows)

    valid_mask = np.stack(
        [np.arange(max_length) < len(sample["step"]) for sample in samples]
    )
    step = pad_array("step", -100)
    next_step = np.stack([_next_distinct_labels(row) for row in step])
    next_step[~valid_mask] = -100
    batch = {
        "motion": torch.from_numpy(pad_array("motion", 0.0)),
        "appearance": torch.from_numpy(pad_array("appearance", 0.0)),
        "sensor": torch.from_numpy(pad_array("sensor", 0.0)),
        "modality_mask": torch.from_numpy(pad_array("modality_mask", False)),
        "step": torch.from_numpy(step),
        "completion": torch.from_numpy(pad_array("completion", 0.0)),
        "component_outcome": torch.from_numpy(pad_array("component_outcome", -100)),
        "state": torch.from_numpy(pad_array("state", 1)),
        "state_mask": torch.from_numpy(pad_array("state_mask", False)),
        "boundary": torch.from_numpy(pad_array("boundary", 0.0)),
        "next_step": torch.from_numpy(next_step),
        "valid_mask": torch.from_numpy(valid_mask),
        "recording_id": [sample["recording_id"] for sample in samples],
        "start_index": [sample["start_index"] for sample in samples],
    }
    if "motion_aux" in samples[0]:
        batch["motion_aux"] = torch.from_numpy(pad_array("motion_aux", 0.0))
    return batch


def _next_distinct_labels(labels: np.ndarray) -> np.ndarray:
    result = np.full_like(labels, -100)
    next_label = -100
    for index in range(len(labels) - 1, -1, -1):
        current = int(labels[index])
        if current < 0:
            continue
        if index + 1 < len(labels):
            following = int(labels[index + 1])
            if following >= 0 and following != current:
                next_label = following
        result[index] = next_label
    return result


def build_transition_matrix(
    records: list[StateGraphCacheRecord],
    num_steps: int,
    smoothing: float = 0.02,
) -> np.ndarray:
    # Keep impossible edges at exactly zero so the graph regularizer can
    # distinguish valid from invalid transitions. Smoothing is applied only
    # to observed edges (and self-loops), not to the complete graph.
    counts = np.zeros((num_steps, num_steps), dtype=np.float64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            labels = arrays["step"].astype(np.int64)
        labels = labels[labels >= 0]
        if labels.size == 0:
            continue
        compact = labels[np.r_[True, labels[1:] != labels[:-1]]]
        for previous, current in zip(compact[:-1], compact[1:]):
            counts[int(previous), int(current)] += 1.0
        for label in np.unique(labels):
            counts[int(label), int(label)] += 1.0
    observed = counts > 0
    counts[observed] += smoothing
    empty_rows = counts.sum(axis=1) == 0
    counts[empty_rows, empty_rows] = 1.0
    return (counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)
