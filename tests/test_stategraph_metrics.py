from __future__ import annotations

import unittest

from aiops.evaluation.temporal_metrics import edit_score, segmental_f1
import numpy as np

from aiops.training.train_stategraph_psr import (
    OutcomeAwareBatchSampler,
    _binary_average_precision,
    _calibrate_event_thresholds,
    _event_metrics_from_samples,
    _match_event_counts,
    _precision_recall_f1,
    _predicted_events,
    _predicted_events_from_scores,
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

    def test_joint_event_threshold_calibration_finds_rare_outcome(self) -> None:
        scores = np.zeros((5, 1, 3), dtype=np.float32)
        scores[2, 0, 1] = 0.12
        samples = [([(2, 0, 1)], scores)]
        thresholds, metrics, pr_auc = _calibrate_event_thresholds(samples, outcomes=3)
        self.assertLessEqual(thresholds[1], 0.12)
        self.assertEqual(metrics[1]["f1"], 100.0)
        self.assertGreaterEqual(pr_auc[1], 0.0)
        self.assertEqual(
            _event_metrics_from_samples(samples, thresholds)[1]["recall"], 100.0
        )

    def test_latency_metric_reports_delayed_but_component_correct_fault(self) -> None:
        scores = np.zeros((8, 1, 3), dtype=np.float32)
        scores[5, 0, 1] = 0.8
        samples = [([(2, 0, 1)], scores)]
        self.assertEqual(
            _event_metrics_from_samples(samples, [0.9, 0.5, 0.9], tolerance=1)[1]["f1"],
            0.0,
        )
        self.assertEqual(
            _event_metrics_from_samples(samples, [0.9, 0.5, 0.9], tolerance=4)[1]["f1"],
            100.0,
        )

    def test_outcome_aware_sampler_places_rare_window_in_each_batch(self) -> None:
        sampler = OutcomeAwareBatchSampler(
            dataset_size=8, batch_size=2, rare_indices=[3], rare_per_batch=1, seed=2
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 4)
        self.assertTrue(all(3 in batch for batch in batches))

    def test_joint_decoder_assigns_one_outcome_and_suppresses_nearby_peak(self) -> None:
        scores = np.zeros((8, 1, 3), dtype=np.float32)
        scores[2, 0] = [0.1, 0.7, 0.2]
        scores[4, 0] = [0.1, 0.6, 0.2]
        scores[7, 0] = [0.8, 0.1, 0.1]
        self.assertEqual(
            _predicted_events_from_scores(
                scores,
                [0.5, 0.5, 0.5],
                minimum_distance=4,
                install_minimum_distance=4,
            ),
            [(2, 0, 1), (7, 0, 0)],
        )

    def test_rare_outcome_peak_is_not_hidden_by_correct_outcome_sum(self) -> None:
        scores = np.zeros((7, 1, 3), dtype=np.float32)
        # Correct evidence keeps rising through row 4, so the summed curve has no
        # local maximum at the true incorrect event on row 2.
        scores[:, 0, 0] = [0.1, 0.2, 0.3, 0.5, 0.8, 0.2, 0.1]
        scores[:, 0, 1] = [0.0, 0.1, 0.6, 0.1, 0.0, 0.0, 0.0]
        predictions = _predicted_events_from_scores(
            scores,
            [0.5, 0.5, 0.5],
            minimum_distance=2,
            install_minimum_distance=2,
        )
        self.assertIn((2, 0, 1), predictions)

    def test_binary_average_precision_rewards_ranked_faults(self) -> None:
        perfect = _binary_average_precision([0.9, 0.8, 0.2], [1, 1, 0])
        reversed_score = _binary_average_precision([0.1, 0.2, 0.9], [1, 1, 0])
        self.assertEqual(perfect, 100.0)
        self.assertGreater(perfect, reversed_score)


if __name__ == "__main__":
    unittest.main()
