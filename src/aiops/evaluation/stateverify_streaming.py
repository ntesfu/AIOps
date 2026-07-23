from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from aiops.evaluation.stateverify_prototypes import (
    PrototypeGroup,
    prototype_anomaly_score,
)
from aiops.stateverify import (
    BeliefTrackerConfig,
    TripleResidualEvidence,
    TypedComponentBeliefTracker,
)


@dataclass(frozen=True)
class StreamingConfig:
    completion_threshold: float = 0.25
    anomaly_threshold: float = 1.0
    anomaly_temperature: float = 0.75
    alert_threshold: float = 0.60
    confirmation_steps: int = 2

    def validate(self) -> None:
        if not 0.0 <= self.completion_threshold <= 1.0:
            raise ValueError("completion_threshold must lie in [0, 1].")
        if self.anomaly_temperature <= 0:
            raise ValueError("anomaly_temperature must be positive.")
        if not 0.0 < self.alert_threshold < 1.0:
            raise ValueError("alert_threshold must lie in (0, 1).")
        if self.confirmation_steps <= 0:
            raise ValueError("confirmation_steps must be positive.")


def component_anomaly_scores(
    component_features: np.ndarray,
    bank: dict[tuple[int, int | None], PrototypeGroup],
) -> tuple[np.ndarray, np.ndarray]:
    """Score every frame against component-only normal prototypes.

    No action/step label is accepted here: this is the leakage-free online path.
    """
    features = np.asarray(component_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("component_features must have shape [time, component, feature].")
    scores = np.full(features.shape[:2], np.nan, dtype=np.float32)
    mask = np.zeros(features.shape[:2], dtype=np.bool_)
    for component in range(features.shape[1]):
        group = bank.get((component, None))
        if group is None:
            continue
        scores[:, component] = prototype_anomaly_score(
            features[:, component], group
        )
        mask[:, component] = True
    return scores, mask


def predicted_step_anomaly_scores(
    component_features: np.ndarray,
    predicted_steps: np.ndarray,
    bank: dict[tuple[int, int | None], PrototypeGroup],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Score with causal predicted steps, falling back to component prototypes."""
    features = np.asarray(component_features, dtype=np.float32)
    steps = np.asarray(predicted_steps, dtype=np.int64)
    if features.ndim != 3:
        raise ValueError("component_features must have shape [time, component, feature].")
    if steps.shape != (features.shape[0],):
        raise ValueError("predicted_steps must have shape [time].")
    scores = np.full(features.shape[:2], np.nan, dtype=np.float32)
    mask = np.zeros(features.shape[:2], dtype=np.bool_)
    sources = {"component_step": 0, "component_fallback": 0, "missing": 0}
    for row, step in enumerate(steps):
        for component in range(features.shape[1]):
            key = (component, int(step))
            source = "component_step"
            if key not in bank:
                key = (component, None)
                source = "component_fallback"
            group = bank.get(key)
            if group is None:
                sources["missing"] += 1
                continue
            scores[row, component] = prototype_anomaly_score(
                features[row, component][None], group
            )[0]
            mask[row, component] = True
            sources[source] += 1
    return scores, mask, sources


def build_streaming_evidence(
    anomaly_scores: np.ndarray,
    anomaly_available: np.ndarray,
    effect_probabilities: np.ndarray,
    config: StreamingConfig,
) -> TripleResidualEvidence:
    config.validate()
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    available = np.asarray(anomaly_available, dtype=np.bool_)
    effect = np.asarray(effect_probabilities, dtype=np.float64)
    if anomaly.ndim != 2 or available.shape != anomaly.shape:
        raise ValueError("Anomaly scores and availability must have shape [time, component].")
    if effect.shape != anomaly.shape + (4,):
        raise ValueError("effect_probabilities must have shape [time, component, 4].")
    completion = effect[..., 1] + effect[..., 2]
    candidate = (completion >= config.completion_threshold) & available
    execution = _sigmoid(
        (anomaly - config.anomaly_threshold) / config.anomaly_temperature
    )
    conditional_incorrect = effect[..., 2] / np.maximum(completion, 1e-8)
    zeros = np.zeros_like(anomaly)
    return TripleResidualEvidence.from_arrays(
        zeros,
        execution,
        conditional_incorrect,
        num_components=anomaly.shape[1],
        procedure_mask=np.zeros_like(candidate),
        execution_mask=candidate,
        effect_mask=candidate,
    )


def run_streaming_tracker(
    state_probabilities: np.ndarray,
    anomaly_scores: np.ndarray,
    anomaly_available: np.ndarray,
    effect_probabilities: np.ndarray,
    config: StreamingConfig,
):
    evidence = build_streaming_evidence(
        anomaly_scores, anomaly_available, effect_probabilities, config
    )
    tracker_config = BeliefTrackerConfig(
        alert_threshold=config.alert_threshold,
        confirmation_steps=config.confirmation_steps,
    )
    return TypedComponentBeliefTracker(
        np.asarray(state_probabilities).shape[1], tracker_config
    ).run(state_probabilities, evidence)


def match_streaming_alerts(
    alerts: Iterable[tuple[str, int, int, float]],
    targets: Iterable[tuple[str, int, int, float]],
    *,
    duration_minutes: float,
    tolerance_seconds: float,
) -> dict[str, float | int | None]:
    """One-to-one, recording/component-aware event matching."""
    predictions = list(alerts)
    truth = list(targets)
    unmatched = set(range(len(truth)))
    delays: list[float] = []
    matched = 0
    for recording, _row, component, timestamp in predictions:
        candidates = [
            index
            for index in unmatched
            if truth[index][0] == recording
            and truth[index][2] == component
            and abs(timestamp - truth[index][3]) <= tolerance_seconds
        ]
        if not candidates:
            continue
        selected = min(
            candidates, key=lambda index: abs(timestamp - truth[index][3])
        )
        unmatched.remove(selected)
        matched += 1
        delays.append(timestamp - truth[selected][3])
    false_alerts = len(predictions) - matched
    false_negatives = len(truth) - matched
    precision = matched / max(len(predictions), 1)
    recall = matched / max(len(truth), 1)
    return {
        "true_events": len(truth),
        "predicted_alerts": len(predictions),
        "true_positives": matched,
        "false_positives": false_alerts,
        "false_negatives": false_negatives,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * 2.0 * precision * recall / max(precision + recall, 1e-12),
        "false_alerts_per_minute": false_alerts / max(duration_minutes, 1e-8),
        "mean_signed_delay_seconds": float(np.mean(delays)) if delays else None,
        "mean_absolute_delay_seconds": (
            float(np.mean(np.abs(delays))) if delays else None
        ),
        "tolerance_seconds": tolerance_seconds,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))
