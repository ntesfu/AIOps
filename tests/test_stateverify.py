from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from aiops.stateverify import (
    BeliefTrackerConfig,
    ComponentState,
    TripleResidualEvidence,
    TypedComponentBeliefTracker,
    canonicalize_stategraph_emissions,
)


def _residuals(
    procedure: np.ndarray, execution: np.ndarray, effect: np.ndarray
) -> TripleResidualEvidence:
    return TripleResidualEvidence.from_arrays(
        procedure, execution, effect, num_components=execution.shape[1]
    )


def test_stategraph_state_order_is_mapped_to_canonical_contract() -> None:
    historical = np.asarray([[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]])
    canonical = canonicalize_stategraph_emissions(historical)
    np.testing.assert_allclose(
        canonical,
        np.asarray([[[0.2, 0.1, 0.7], [0.3, 0.6, 0.1]]]),
    )


def test_single_frame_incorrect_spike_does_not_fire_alert() -> None:
    emissions = np.asarray(
        [
            [[0.9, 0.09, 0.01]],
            [[0.05, 0.05, 0.90]],
            [[0.05, 0.94, 0.01]],
        ]
    )
    high = np.asarray([[0.1], [0.95], [0.1]])
    trace = TypedComponentBeliefTracker(
        1,
        BeliefTrackerConfig(
            confirmation_steps=2,
            alert_threshold=0.55,
            residual_strength=1.0,
        ),
    ).run(emissions, _residuals(high, high, high))
    assert not trace.alert_active.any()
    assert not trace.events


def test_sustained_effect_residual_fires_attributed_alert_and_recovery() -> None:
    emissions = np.asarray(
        [
            [[0.95, 0.04, 0.01]],
            [[0.05, 0.05, 0.90]],
            [[0.02, 0.03, 0.95]],
            [[0.02, 0.97, 0.01]],
            [[0.02, 0.97, 0.01]],
        ]
    )
    procedure = np.full((5, 1), 0.2)
    execution = np.asarray([[0.2], [0.65], [0.70], [0.1], [0.1]])
    effect = np.asarray([[0.1], [0.95], [0.98], [0.05], [0.05]])
    tracker = TypedComponentBeliefTracker(
        1,
        BeliefTrackerConfig(
            confirmation_steps=2,
            recovery_confirmation_steps=2,
            alert_threshold=0.40,
            recovery_threshold=0.65,
            residual_strength=1.0,
        ),
    )
    trace = tracker.run(emissions, _residuals(procedure, execution, effect))
    assert [event.kind for event in trace.events] == ["alert", "recovery"]
    alert = trace.events[0]
    assert alert.step == 2
    assert alert.attribution == "effect"
    assert alert.state == ComponentState.INSTALLED_INCORRECT.name.lower()
    assert trace.events[1].step == 4
    assert not trace.alert_active[-1, 0]


def test_components_are_filtered_independently() -> None:
    emissions = np.asarray(
        [
            [[0.95, 0.04, 0.01], [0.95, 0.04, 0.01]],
            [[0.02, 0.03, 0.95], [0.03, 0.96, 0.01]],
            [[0.02, 0.03, 0.95], [0.03, 0.96, 0.01]],
        ]
    )
    procedure = np.full((3, 2), 0.2)
    execution = np.asarray([[0.1, 0.1], [0.9, 0.1], [0.9, 0.1]])
    effect = np.asarray([[0.1, 0.1], [0.95, 0.05], [0.95, 0.05]])
    trace = TypedComponentBeliefTracker(
        2,
        BeliefTrackerConfig(
            confirmation_steps=2,
            alert_threshold=0.40,
            residual_strength=1.0,
        ),
    ).run(emissions, _residuals(procedure, execution, effect))
    assert trace.alert_active[-1].tolist() == [True, False]
    assert [event.component for event in trace.events] == [0]


def test_missing_residuals_are_neutral_and_auditable() -> None:
    emissions = np.asarray([[[0.1, 0.1, 0.8]], [[0.05, 0.05, 0.9]]])
    missing = np.full((2, 1), np.nan)
    effect = np.asarray([[0.9], [0.95]])
    residuals = TripleResidualEvidence.from_arrays(
        missing, missing, effect, num_components=1
    )
    trace = TypedComponentBeliefTracker(
        1,
        BeliefTrackerConfig(
            confirmation_steps=1,
            alert_threshold=0.5,
            residual_strength=1.0,
        ),
    ).run(emissions, residuals)
    event = trace.events[0]
    assert event.attribution == "effect"
    assert event.residuals["procedure"] is None
    assert event.residuals["execution"] is None


def test_offline_cli_writes_machine_readable_trace(tmp_path: Path) -> None:
    input_path = tmp_path / "emissions.npz"
    output_path = tmp_path / "trace.json"
    np.savez_compressed(
        input_path,
        state_emissions=np.asarray(
            [[[0.95, 0.04, 0.01]], [[0.02, 0.03, 0.95]]]
        ),
        procedure_residual=np.asarray([0.1, 0.9]),
        execution_residual=np.asarray([[0.1], [0.9]]),
        effect_residual=np.asarray([[0.1], [0.98]]),
        component_names=np.asarray(["front_pin"]),
        component_outcome=np.asarray([[-100], [1]], dtype=np.int64),
        seconds_per_step=np.asarray(0.5, dtype=np.float32),
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_stateverify_tracker.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--confirmation-steps",
            "1",
            "--alert-threshold",
            "0.5",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["component_names"] == ["front_pin"]
    assert report["summary"]["alerts"] == 1
    assert report["events"][0]["component_name"] == "front_pin"
    assert report["incorrect_alert_metrics"]["recall_percent"] == 100.0
    assert report["incorrect_alert_metrics"]["false_alerts_per_minute"] == 0.0
