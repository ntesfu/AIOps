from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiops.data.stategraph_cache import StateGraphCacheRecord
from aiops.recognition import build_step_transition_prior, quantize_lookahead


def _record(path: Path, recording_id: str, split: str) -> StateGraphCacheRecord:
    return StateGraphCacheRecord(
        recording_id=recording_id,
        split=split,
        path=path,
        num_steps=3,
        motion_dim=1,
        appearance_dim=1,
        sensor_dim=1,
        num_components=2,
        num_completion_components=2,
    )


def test_lookahead_floor_respects_strict_1_5_second_contract() -> None:
    budget = quantize_lookahead(1.5, 8.0 / 30.0)
    assert budget.total_steps == 5
    assert budget.decoder_steps == 5
    assert budget.effective_seconds == pytest.approx(4.0 / 3.0)
    assert budget.effective_seconds <= budget.requested_seconds


def test_lookahead_exact_multiple_zero_and_neural_reservation() -> None:
    exact = quantize_lookahead(1.2, 0.4, neural_steps=1)
    assert exact.total_steps == 3
    assert exact.neural_steps == 1
    assert exact.decoder_steps == 2
    assert exact.effective_seconds == pytest.approx(1.2)
    assert exact.effective_seconds <= exact.requested_seconds
    zero = quantize_lookahead(0.0, 0.25)
    assert zero.total_steps == zero.decoder_steps == 0
    assert zero.effective_seconds == 0.0


@pytest.mark.parametrize(
    ("seconds", "step", "neural"),
    [
        (-0.1, 0.25, 0),
        (float("nan"), 0.25, 0),
        (1.0, 0.0, 0),
        (1.0, float("inf"), 0),
        (0.5, 0.25, -1),
        (0.5, 0.25, 3),
    ],
)
def test_lookahead_rejects_invalid_arguments(
    seconds: float, step: float, neural: int
) -> None:
    with pytest.raises(ValueError):
        quantize_lookahead(seconds, step, neural_steps=neural)


def test_transition_prior_is_train_only_deterministic_and_safe(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    completion = np.zeros((7, 2), dtype=np.float32)
    completion[2, 0] = 1.0
    completion[5, 1] = 1.0
    np.savez_compressed(train_path, completion=completion)
    # A missing validation path proves the builder filters by split before I/O.
    records = [
        _record(train_path, "train-b", "train"),
        _record(tmp_path / "must-not-be-opened.npz", "val-a", "val"),
    ]

    first = build_step_transition_prior(records, 3, soft_penalty=-4.0)
    second = build_step_transition_prior(reversed(records), 3, soft_penalty=-4.0)
    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(first.allowed, second.allowed)
    assert first.metadata == second.metadata
    assert first.counts[1, 2] == 1
    assert first.allowed.shape == (3, 3)
    assert np.diag(first.allowed).all()
    assert first.allowed[0].all() and first.allowed[:, 0].all()
    assert (first.penalties[first.allowed] == 0.0).all()
    assert (first.penalties[~first.allowed] == -4.0).all()
    assert first.metadata["source_split"] == "train"
    assert first.metadata["source_recordings"] == 1


def test_transition_prior_rejects_invalid_shape_or_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training record"):
        build_step_transition_prior([], 3)
    bad_path = tmp_path / "bad.npz"
    bad_completion = np.zeros((3, 3), dtype=np.float32)
    bad_completion[1, 2] = 1.0
    np.savez_compressed(bad_path, completion=bad_completion)
    with pytest.raises(ValueError, match="outside"):
        build_step_transition_prior([_record(bad_path, "bad", "train")], 2)
