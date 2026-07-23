from __future__ import annotations

import numpy as np

from aiops.evaluation.stateverify_prototypes import PrototypeGroup
from aiops.evaluation.stateverify_streaming import (
    StreamingConfig,
    build_streaming_evidence,
    component_anomaly_scores,
    match_streaming_alerts,
    predicted_step_anomaly_scores,
    run_streaming_tracker,
)


def test_component_anomaly_scoring_uses_only_component_fallback():
    bank = {
        (0, None): PrototypeGroup(
            centers=np.asarray([[1.0, 0.0]], dtype=np.float32),
            residual_median=0.0,
            residual_scale=0.1,
            samples=10,
        ),
        (0, 9): PrototypeGroup(
            centers=np.asarray([[0.0, 1.0]], dtype=np.float32),
            residual_median=0.0,
            residual_scale=0.1,
            samples=10,
        ),
    }
    features = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    scores, mask = component_anomaly_scores(features, bank)
    assert mask.all()
    assert scores[0, 0] < scores[1, 0]


def test_streaming_evidence_is_masked_until_completion_candidate():
    anomaly = np.asarray([[4.0], [4.0]])
    available = np.ones_like(anomaly, dtype=bool)
    effect = np.asarray([[[0.9, 0.05, 0.05, 0.0]], [[0.1, 0.2, 0.7, 0.0]]])
    evidence = build_streaming_evidence(
        anomaly,
        available,
        effect,
        StreamingConfig(completion_threshold=0.5),
    )
    assert not evidence.execution_mask[0, 0]
    assert evidence.execution_mask[1, 0]
    assert evidence.effect[1, 0] > 0.7
    assert evidence.effect_mask[1, 0]


def test_low_confidence_supervised_effect_cannot_veto_execution_residual():
    anomaly = np.asarray([[4.0]])
    available = np.ones_like(anomaly, dtype=bool)
    effect = np.asarray([[[0.1, 0.895, 0.005, 0.0]]])
    evidence = build_streaming_evidence(
        anomaly,
        available,
        effect,
        StreamingConfig(
            completion_threshold=0.5,
            effect_positive_threshold=0.5,
        ),
    )
    assert evidence.execution_mask[0, 0]
    assert not evidence.effect_mask[0, 0]


def test_predicted_step_scoring_uses_step_bank_without_label_input():
    bank = {
        (0, None): PrototypeGroup(
            centers=np.asarray([[1.0, 0.0]], dtype=np.float32),
            residual_median=0.0,
            residual_scale=0.1,
            samples=10,
        ),
        (0, 4): PrototypeGroup(
            centers=np.asarray([[0.0, 1.0]], dtype=np.float32),
            residual_median=0.0,
            residual_scale=0.1,
            samples=10,
        ),
    }
    features = np.asarray([[[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32)
    scores, mask, sources = predicted_step_anomaly_scores(
        features, np.asarray([4, 8]), bank
    )
    assert mask.all()
    assert scores[0, 0] < scores[1, 0]
    assert sources["component_step"] == 1
    assert sources["component_fallback"] == 1


def test_matching_is_recording_and_component_aware():
    targets = [("r1", 10, 2, 5.0), ("r2", 10, 2, 5.0)]
    alerts = [("r1", 11, 2, 5.4), ("r1", 11, 1, 5.4)]
    metrics = match_streaming_alerts(
        alerts, targets, duration_minutes=2.0, tolerance_seconds=1.0
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_alerts_per_minute"] == 0.5


def test_tracker_cannot_start_incorrect_transition_without_completion_candidate():
    state = np.asarray([[[0.01, 0.01, 0.98]]] * 6)
    anomaly = np.asarray([[8.0]] * 6)
    available = np.ones_like(anomaly, dtype=bool)
    effect = np.asarray([[[0.99, 0.005, 0.005, 0.0]]] * 6)
    trace = run_streaming_tracker(
        state,
        anomaly,
        available,
        effect,
        StreamingConfig(
            completion_threshold=0.5,
            alert_threshold=0.45,
            confirmation_steps=1,
        ),
    )
    assert not any(event.kind == "alert" for event in trace.events)


def test_open_set_execution_can_promote_incorrect_emission_at_candidate():
    state = np.asarray([[[0.01, 0.98, 0.01]]] * 3)
    anomaly = np.asarray([[8.0]] * 3)
    available = np.ones_like(anomaly, dtype=bool)
    effect = np.asarray([[[0.01, 0.98, 0.01, 0.0]]] * 3)
    trace = run_streaming_tracker(
        state,
        anomaly,
        available,
        effect,
        StreamingConfig(
            completion_threshold=0.5,
            anomaly_threshold=1.0,
            alert_threshold=0.45,
            confirmation_steps=1,
            promote_execution_to_state_emission=True,
        ),
    )
    assert any(event.kind == "alert" for event in trace.events)
