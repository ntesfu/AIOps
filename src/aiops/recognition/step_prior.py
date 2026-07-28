"""Train-derived transition legality for coarse STEP decoding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from aiops.recognition.step_taxonomy import densify_completion_to_steps


@dataclass(frozen=True)
class StepTransitionPrior:
    """Observed transitions plus safe soft/hard decoder constraints."""

    counts: np.ndarray
    allowed: np.ndarray
    penalties: np.ndarray
    metadata: dict[str, Any]

    @property
    def forbidden(self) -> np.ndarray:
        return ~self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "counts": self.counts.astype(np.int64).tolist(),
            "allowed": self.allowed.astype(bool).tolist(),
            "penalties": self.penalties.astype(float).tolist(),
        }


def build_step_transition_prior(
    records: Iterable[Any],
    num_steps: int,
    *,
    soft_penalty: float = -6.0,
    background_step: int = 0,
) -> StepTransitionPrior:
    """Build a deterministic prior from train completion targets only.

    Non-training records are filtered before their paths are opened. Observed
    transitions are derived from the same run-up-densified completion targets
    used by the dedicated STEP loss. Self transitions and all transitions
    to/from background remain safe even when absent from a finite train split.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not 0 <= background_step < num_steps:
        raise ValueError("background_step must index a valid STEP class")
    if not np.isfinite(soft_penalty) or soft_penalty > 0:
        raise ValueError("soft_penalty must be finite and non-positive")

    train_records = sorted(
        (record for record in records if str(record.split).lower() == "train"),
        key=lambda record: (str(record.recording_id), str(record.path)),
    )
    if not train_records:
        raise ValueError("at least one training record is required")

    counts = np.zeros((num_steps, num_steps), dtype=np.int64)
    rows = 0
    for record in train_records:
        with np.load(record.path, allow_pickle=False) as arrays:
            completion = np.asarray(arrays["completion"])
        targets = densify_completion_to_steps(
            completion, background_step=background_step
        )
        if targets.size:
            if targets.min() < 0 or targets.max() >= num_steps:
                raise ValueError(
                    f"{record.recording_id} has a run-up STEP target outside "
                    f"[0, {num_steps})"
                )
            rows += int(targets.size)
        if targets.size > 1:
            np.add.at(counts, (targets[:-1], targets[1:]), 1)

    allowed = counts > 0
    np.fill_diagonal(allowed, True)
    allowed[background_step, :] = True
    allowed[:, background_step] = True
    penalties = np.where(allowed, 0.0, float(soft_penalty)).astype(np.float64)
    recording_ids = [str(record.recording_id) for record in train_records]
    source_digest = hashlib.sha256(
        json.dumps(recording_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "format_version": 1,
        "source_split": "train",
        "source_recordings": len(train_records),
        "source_recording_ids_sha256": source_digest,
        "source_rows": rows,
        "num_steps": int(num_steps),
        "background_step": int(background_step),
        "soft_penalty": float(soft_penalty),
        "target_derivation": "run_up_densified_completion_v1",
        "safety": "self_and_background_transitions_always_allowed",
    }
    return StepTransitionPrior(counts, allowed, penalties, metadata)
