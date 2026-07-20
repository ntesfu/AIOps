from __future__ import annotations

import json
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
    outcome: np.ndarray,
    state: np.ndarray,
    state_mask: np.ndarray,
    boundary: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    length = len(step)
    arrays = {
        "motion": np.asarray(motion, dtype=np.float32),
        "appearance": np.asarray(appearance, dtype=np.float32),
        "sensor": np.asarray(sensor, dtype=np.float32),
        "modality_mask": np.asarray(modality_mask, dtype=np.bool_),
        "step": np.asarray(step, dtype=np.int64),
        "outcome": np.asarray(outcome, dtype=np.int64),
        "state": np.asarray(state, dtype=np.int64),
        "state_mask": np.asarray(state_mask, dtype=np.bool_),
        "boundary": np.asarray(boundary, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float32),
    }
    for name, array in arrays.items():
        if array.shape[0] != length:
            raise ValueError(f"{name} has {array.shape[0]} rows, expected {length}")
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
    ) -> None:
        if sequence_length <= 0 or sequence_stride <= 0:
            raise ValueError("sequence_length and sequence_stride must be positive")
        self.records = records
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.training = training
        self.seed = seed
        self.windows: list[tuple[int, int]] = []
        for record_index, record in enumerate(records):
            with np.load(record.path, allow_pickle=False) as arrays:
                length = int(arrays["step"].shape[0])
            if length <= sequence_length:
                self.windows.append((record_index, 0))
            else:
                starts = list(range(0, max(1, length - sequence_length + 1), sequence_stride))
                final_start = length - sequence_length
                if not starts or starts[-1] != final_start:
                    starts.append(final_start)
                self.windows.extend((record_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, start = self.windows[index]
        record = self.records[record_index]
        with np.load(record.path, allow_pickle=False) as arrays:
            length = int(arrays["step"].shape[0])
            if self.training and length > self.sequence_length:
                rng = np.random.default_rng(self.seed + index)
                jitter = int(rng.integers(-self.sequence_stride // 4, self.sequence_stride // 4 + 1))
                start = int(np.clip(start + jitter, 0, length - self.sequence_length))
            end = min(length, start + self.sequence_length)
            sample = {name: arrays[name][start:end].copy() for name in arrays.files}
        sample["recording_id"] = record.recording_id
        sample["start_index"] = start
        return sample


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
    next_step = np.full_like(step, -100)
    next_step[:, :-1] = step[:, 1:]
    next_step[~valid_mask] = -100
    return {
        "motion": torch.from_numpy(pad_array("motion", 0.0)),
        "appearance": torch.from_numpy(pad_array("appearance", 0.0)),
        "sensor": torch.from_numpy(pad_array("sensor", 0.0)),
        "modality_mask": torch.from_numpy(pad_array("modality_mask", False)),
        "step": torch.from_numpy(step),
        "outcome": torch.from_numpy(pad_array("outcome", -100)),
        "state": torch.from_numpy(pad_array("state", 1)),
        "state_mask": torch.from_numpy(pad_array("state_mask", False)),
        "boundary": torch.from_numpy(pad_array("boundary", 0.0)),
        "next_step": torch.from_numpy(next_step),
        "valid_mask": torch.from_numpy(valid_mask),
        "recording_id": [sample["recording_id"] for sample in samples],
        "start_index": [sample["start_index"] for sample in samples],
    }


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
