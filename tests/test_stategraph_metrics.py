from __future__ import annotations

import unittest

from aiops.evaluation.temporal_metrics import edit_score, segmental_f1
import numpy as np

from aiops.training.train_stategraph_psr import (
    _match_event_counts,
    _precision_recall_f1,
    _predicted_events,
)


class StateGraphMetricsTest(unittest.TestCase):
    def test_metrics_reward_correct_segments_not_repeated_frames(self) -> None:
        truth = [0, 0, 1, 1, 2, 2]
        self.assertEqual(edit_score(truth, truth), 100.0)
        self.assertEqual(segmental_f1(truth, truth, 0.5), 100.0)
        oversegmented = [0, 1, 0, 1, 2, 2]
        self.assertLess(edit_score(oversegmented, truth), 100.0)
        self.assertLess(segmental_f1(oversegmented, truth, 0.5), 100.0)

    def test_ignored_labels_are_removed(self) -> None:
        truth = [-100, 0, 0, 1, -100]
        prediction = [2, 0, 0, 1, 2]
        self.assertEqual(edit_score(prediction, truth), 100.0)

    def test_fault_metrics_penalize_overprediction(self) -> None:
        metrics = _precision_recall_f1({"tp": 5, "predicted": 20, "true": 5})
        self.assertEqual(metrics["precision"], 25.0)
        self.assertEqual(metrics["recall"], 100.0)
        self.assertEqual(metrics["f1"], 40.0)

    def test_event_metric_uses_component_and_temporal_tolerance(self) -> None:
        truth = [(5, 0, 1), (5, 1, 1)]
        predicted = [(4, 0, 1), (6, 1, 1), (5, 2, 1)]
        self.assertEqual(
            _match_event_counts(truth, predicted, tolerance=1),
            {"true": 2, "predicted": 3, "tp": 2},
        )

    def test_event_predictions_are_peak_suppressed_per_component(self) -> None:
        scores = np.asarray([[0.1], [0.7], [0.9], [0.8], [0.1]], dtype=np.float32)
        outcomes = np.ones((5, 1), dtype=np.int64)
        self.assertEqual(_predicted_events(scores, outcomes), [(2, 0, 1)])


if __name__ == "__main__":
    unittest.main()
