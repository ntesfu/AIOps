from __future__ import annotations

import numpy as np

from aiops.evaluation.stateverify_prototypes import (
    binary_average_precision,
    calibrate_threshold,
    fit_prototype_bank,
    fit_prototype_group,
    load_prototype_bank,
    prototype_anomaly_score,
    save_prototype_bank,
    score_with_bank,
    threshold_metrics,
)


def test_prototype_group_scores_outlier_above_normal_examples():
    rng = np.random.default_rng(3)
    normal = np.asarray([1.0, 0.0, 0.0]) + 0.03 * rng.normal(size=(100, 3))
    group = fit_prototype_group(normal, prototypes=2)
    normal_score = prototype_anomaly_score(normal, group)
    outlier_score = prototype_anomaly_score(np.asarray([[0.0, 1.0, 0.0]]), group)
    assert float(outlier_score[0]) > float(np.quantile(normal_score, 0.95))


def test_bank_prefers_step_group_and_falls_back_to_component():
    examples = {
        (0, 2): [np.asarray([1.0, 0.0])] * 25,
        (0, 3): [np.asarray([0.9, 0.1])] * 10,
    }
    bank = fit_prototype_bank(examples, minimum_step_samples=20)
    _, source = score_with_bank(bank, 0, 2, np.asarray([1.0, 0.0]))
    assert source == "component_step"
    _, source = score_with_bank(bank, 0, 3, np.asarray([0.9, 0.1]))
    assert source == "component_fallback"


def test_threshold_calibration_respects_false_alert_budget():
    scores = [0.1, 0.2, 0.3, 1.0, 1.2]
    labels = [0, 0, 0, 1, 1]
    calibrated = calibrate_threshold(
        scores, labels, duration_minutes=2.0, max_false_alerts_per_minute=0.0
    )
    metrics = threshold_metrics(
        scores, labels, calibrated["threshold"], duration_minutes=2.0
    )
    assert metrics["recall"] == 100.0
    assert metrics["false_alerts_per_minute"] == 0.0
    assert binary_average_precision(scores, labels) == 100.0


def test_prototype_bank_round_trip_without_pickle(tmp_path):
    bank = fit_prototype_bank(
        {(2, 4): [np.asarray([1.0, 0.1], dtype=np.float32)] * 25},
        minimum_step_samples=20,
    )
    path = tmp_path / "bank.npz"
    save_prototype_bank(path, bank, metadata={"split": "train"})
    restored, metadata = load_prototype_bank(path)
    assert metadata == {"split": "train"}
    assert restored.keys() == bank.keys()
    for key in bank:
        np.testing.assert_allclose(restored[key].centers, bank[key].centers)
        assert restored[key].samples == bank[key].samples
