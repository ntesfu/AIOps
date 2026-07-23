from __future__ import annotations

import unittest
import random
import tempfile
from argparse import Namespace
from pathlib import Path

from aiops.evaluation.temporal_metrics import edit_score, segmental_f1
import numpy as np

from aiops.training.train_stategraph_psr import (
    OutcomeAwareBatchSampler,
    _effective_rare_window_boost,
    _event_branch_parameter_prefixes,
    _initialize_run_directory,
    _initialization_enabled,
    _checkpoint_argument_error,
    _validation_selection_result,
    _validation_selection_score,
    _binary_average_precision,
    _calibrate_event_thresholds,
    _capture_rng_state,
    _class_f1_from_confusion,
    _event_timing_diagnostics,
    _export_stateverify_emissions,
    _event_metrics_from_samples,
    _factorized_mistake_diagnostics,
    _macro_f1_from_confusion,
    _model_configs_compatible,
    _restore_rng_state,
    _stitch_recording_chunks,
    _calibrate_incorrect_state_threshold,
    _match_event_counts,
    _precision_recall_f1,
    _predicted_events,
    _predicted_events_from_scores,
    _project_event_scores_to_states,
)


class StateGraphMetricsTest(unittest.TestCase):
    def test_stateverify_export_uses_portable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            row = _export_stateverify_emissions(
                output_dir,
                "operator/recording 1",
                np.asarray(
                    [
                        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
                        [[0.1, 0.2, 0.7], [0.1, 0.8, 0.1]],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.1, 0.9], dtype=np.float32),
                np.asarray([[0.1, 0.2], [0.9, 0.1]], dtype=np.float32),
                np.asarray([[0.0, 0.0], [0.8, 0.0]], dtype=np.float32),
                component_outcome=np.asarray(
                    [[-100, -100], [1, -100]], dtype=np.int64
                ),
                seconds_per_step=0.5,
                component_names=["pin", "wheel"],
            )
            self.assertEqual(row["recording_id"], "operator/recording 1")
            self.assertEqual(row["rows"], 2)
            self.assertEqual(row["components"], 2)
            with np.load(output_dir / row["path"], allow_pickle=False) as archive:
                self.assertEqual(
                    archive["component_names"].tolist(), ["pin", "wheel"]
                )
                self.assertEqual(tuple(archive["state_emissions"].shape), (2, 2, 3))
                self.assertEqual(
                    archive["component_outcome"].tolist(),
                    [[-100, -100], [1, -100]],
                )
                self.assertEqual(float(archive["seconds_per_step"]), 0.5)

    def test_event_scores_project_to_distinct_state_axis(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")

        scores = torch.tensor([[0.2, 0.7, 0.5], [0.8, 0.1, 0.4]])
        mapping = torch.tensor([1, 3, 1])
        projected = _project_event_scores_to_states(scores, mapping, num_states=5)

        self.assertEqual(tuple(projected.shape), (2, 5))
        torch.testing.assert_close(
            projected,
            torch.tensor(
                [[0.0, 0.5, 0.0, 0.7, 0.0], [0.0, 0.8, 0.0, 0.1, 0.0]]
            ),
        )

    def test_event_branch_prefixes_cover_procedure_and_exclude_action_classifier(self) -> None:
        prefixes = _event_branch_parameter_prefixes()
        self.assertTrue("procedure_event_context.weight".startswith(prefixes))
        self.assertTrue("incorrect_onset_head.weight".startswith(prefixes))
        self.assertTrue("roi_visual_stem.1.weight".startswith(prefixes))
        self.assertTrue("component_roi_attention.in_proj_weight".startswith(prefixes))
        self.assertTrue("structured_roi_gate_raw".startswith(prefixes))
        self.assertFalse("step_classifier.weight".startswith(prefixes))

    def test_legacy_model_config_accepts_only_default_off_new_field(self) -> None:
        legacy = {"hidden_dim": 256}
        self.assertTrue(
            _model_configs_compatible(
                legacy, {"hidden_dim": 256, "factorized_mistake_detection": False}
            )
        )
        self.assertFalse(
            _model_configs_compatible(
                legacy, {"hidden_dim": 256, "factorized_mistake_detection": True}
            )
        )
        self.assertFalse(
            _model_configs_compatible(
                legacy, {"hidden_dim": 256, "component_evidence": True}
            )
        )
        self.assertFalse(
            _model_configs_compatible(
                legacy, {"hidden_dim": 256, "event_only_motion_aux": True}
            )
        )
        self.assertTrue(
            _model_configs_compatible(
                legacy,
                {
                    "hidden_dim": 256,
                    "structured_roi_tokens": False,
                    "roi_token_count": 4,
                    "roi_embedding_dim": 0,
                    "roi_global_dim": 3,
                    "roi_change_lag": 4,
                },
            )
        )

    def test_action_only_selection_uses_documented_action_score(self) -> None:
        metrics = self._selection_metrics()
        result = _validation_selection_result(
            metrics, strategy="action_only", epoch=4
        )
        expected = (
            0.40 * metrics["frame_accuracy"]
            + 0.30 * metrics["edit"]
            + 0.30 * metrics["f1@50"]
        )
        self.assertAlmostEqual(result["score"], expected)
        self.assertTrue(result["operationally_eligible"])

    def test_factorized_diagnostics_separate_timing_and_component_selection(self) -> None:
        outcomes = np.full((4, 3), -100, dtype=np.int64)
        outcomes[1, 2] = 1
        outcomes[3, 0] = 1
        diagnostics = _factorized_mistake_diagnostics(
            [
                {
                    "component_outcome": outcomes,
                    "any_mistake_onset_probability": np.asarray(
                        [0.1, 0.9, 0.2, 0.8], dtype=np.float32
                    ),
                    "incorrect_component_probabilities": np.asarray(
                        [
                            [0.4, 0.3, 0.3],
                            [0.1, 0.2, 0.7],
                            [0.2, 0.6, 0.2],
                            [0.1, 0.8, 0.1],
                        ],
                        dtype=np.float32,
                    ),
                }
            ]
        )
        self.assertEqual(diagnostics["any_mistake_onset_average_precision"], 100.0)
        self.assertEqual(diagnostics["mistake_component_top1_accuracy"], 50.0)
        self.assertEqual(diagnostics["mistake_component_onset_rows"], 2.0)

    def test_factorized_diagnostics_are_unavailable_for_legacy_outputs(self) -> None:
        diagnostics = _factorized_mistake_diagnostics(
            [{"component_outcome": np.full((3, 1), -100, dtype=np.int64)}]
        )
        self.assertTrue(np.isnan(diagnostics["any_mistake_onset_average_precision"]))
        self.assertTrue(np.isnan(diagnostics["mistake_component_top1_accuracy"]))
        self.assertEqual(diagnostics["mistake_component_onset_rows"], 0.0)

    @staticmethod
    def _selection_metrics(**overrides):
        metrics = {
            "frame_accuracy": 20.0,
            "edit": 10.0,
            "f1@50": 5.0,
            "state_macro_f1": 45.0,
            "state_incorrect_f1": 4.0,
            "incorrect_event_f1": 6.0,
            "incorrect_event_f1_tol4s": 8.0,
            "incorrect_event_recall": 5.0,
            "normality_incorrect_average_precision": 30.0,
            "normality_incorrect_prevalence": 20.0,
            "incorrect_false_alerts_per_minute": 1.0,
        }
        metrics.update(overrides)
        return metrics

    def test_legacy_selection_strategy_preserves_existing_score(self) -> None:
        metrics = self._selection_metrics()
        result = _validation_selection_result(
            metrics, strategy="legacy", epoch=3, mistake_weight=0.75
        )
        self.assertAlmostEqual(
            result["score"], _validation_selection_score(metrics, mistake_weight=0.75)
        )
        self.assertTrue(result["operationally_eligible"])

    def test_operational_selection_rejects_silent_or_unranked_mistakes(self) -> None:
        silent = _validation_selection_result(
            self._selection_metrics(
                incorrect_event_f1=0.0,
                incorrect_event_recall=0.0,
                incorrect_false_alerts_per_minute=0.0,
            ),
            strategy="operational_harmonic",
            epoch=8,
        )
        self.assertFalse(silent["operationally_eligible"])
        below_prevalence = _validation_selection_result(
            self._selection_metrics(normality_incorrect_average_precision=19.0),
            strategy="operational_harmonic",
            epoch=8,
        )
        self.assertFalse(below_prevalence["operationally_eligible"])

    def test_operational_selection_formula_and_tie_metadata(self) -> None:
        result = _validation_selection_result(
            self._selection_metrics(),
            strategy="operational_harmonic",
            epoch=7,
            max_false_alerts_per_minute=2.0,
        )
        action = 0.4 * 20.0 + 0.3 * 10.0 + 0.3 * 5.0
        mistake = 0.45 * 30.0 + 0.25 * 4.0 + 0.2 * 6.0 + 0.1 * 8.0
        expected = 2 * action * mistake / (action + mistake) - 0.5
        self.assertAlmostEqual(result["score"], expected)
        self.assertTrue(result["operationally_eligible"])
        self.assertEqual(result["tie_key"], [expected, 5.0, action, -7])
        over_limit = _validation_selection_result(
            self._selection_metrics(incorrect_false_alerts_per_minute=2.01),
            strategy="operational_harmonic",
            epoch=7,
        )
        self.assertFalse(over_limit["operationally_eligible"])

    def test_recording_metrics_are_invariant_to_sequence_windows(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2, 1, 1], dtype=np.int64)
        prediction = np.asarray([0, 0, 1, 1, 2, 2, 1, 1], dtype=np.int64)
        outcomes = np.full((8, 1), -100, dtype=np.int64)
        outcomes[3, 0] = 1
        scores = np.zeros((8, 1, 3), dtype=np.float32)
        scores[3, 0, 1] = 0.9
        seen = np.ones(3, dtype=np.bool_)

        def chunk(start: int, end: int) -> dict[str, np.ndarray | int]:
            return {
                "start": start,
                "step_target": truth[start:end],
                "step_prediction": prediction[start:end],
                "raw_step_prediction": prediction[start:end],
                "component_outcome": outcomes[start:end],
                "event_score": scores[start:end],
                "seen_action_mask": seen,
            }

        unsplit = _stitch_recording_chunks([chunk(0, 8)])
        split = _stitch_recording_chunks([chunk(0, 4), chunk(4, 8)])
        for reconstructed in (unsplit, split):
            self.assertEqual(
                edit_score(
                    reconstructed["step_prediction"].tolist(),
                    reconstructed["step_target"].tolist(),
                ),
                100.0,
            )
            self.assertEqual(
                segmental_f1(
                    reconstructed["step_prediction"].tolist(),
                    reconstructed["step_target"].tolist(),
                    0.5,
                ),
                100.0,
            )
            samples = [
                (
                    [(3, 0, 1)],
                    reconstructed["event_score"],
                )
            ]
            self.assertEqual(
                _event_metrics_from_samples(samples, [0.99, 0.5, 0.99], tolerance=0)[1],
                {"precision": 100.0, "recall": 100.0, "f1": 100.0},
            )
        np.testing.assert_array_equal(split["step_prediction"], unsplit["step_prediction"])

    def test_overlapping_windows_prefer_prediction_with_more_causal_context(self) -> None:
        seen = np.ones(3, dtype=np.bool_)
        first = {
            "start": 0,
            "step_target": np.asarray([0, 0, 1, 1]),
            "step_prediction": np.asarray([0, 0, 2, 2]),
            "seen_action_mask": seen,
        }
        second = {
            "start": 2,
            "step_target": np.asarray([1, 1, 2, 2]),
            "step_prediction": np.asarray([1, 1, 2, 2]),
            "seen_action_mask": seen,
        }
        stitched = _stitch_recording_chunks([first, second])
        # Global rows 2 and 3 come from local rows 2/3 of the first window,
        # not local rows 0/1 of the second window.
        np.testing.assert_array_equal(stitched["step_prediction"], [0, 0, 2, 2, 2, 2])

    def test_training_rng_state_round_trip(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        state = _capture_rng_state()
        expected = (random.random(), float(np.random.random()), float(torch.rand(())))
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        _restore_rng_state(state)
        actual = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(actual, expected)

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

    def test_state_macro_f1_uses_all_supported_classes(self) -> None:
        confusion = np.asarray([[2, 0, 0], [0, 1, 1], [0, 0, 2]])
        self.assertEqual(_class_f1_from_confusion(confusion, 0), 100.0)
        self.assertAlmostEqual(_macro_f1_from_confusion(confusion), (100 + 66.6666667 + 80) / 3)

    def test_rare_state_threshold_is_calibrated_below_argmax(self) -> None:
        scores = np.asarray([0.40, 0.35, 0.20, 0.10])
        truth = np.asarray([0, 0, 1, 2])
        other = np.asarray([1, 1, 1, 2])
        threshold, confusion = _calibrate_incorrect_state_threshold(scores, truth, other)
        self.assertLess(threshold, 0.5)
        self.assertEqual(_class_f1_from_confusion(confusion, 0), 100.0)

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

    def test_event_calibration_can_reject_high_confidence_false_peak(self) -> None:
        scores = np.zeros((16, 1, 3), dtype=np.float32)
        scores[2, 0, 1] = 0.98
        scores[13, 0, 1] = 0.95
        thresholds, metrics, _ = _calibrate_event_thresholds(
            [([(2, 0, 1)], scores)], outcomes=3
        )
        self.assertGreater(thresholds[1], 0.95)
        self.assertEqual(metrics[1]["f1"], 100.0)

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
        diagnostics = _event_timing_diagnostics(
            samples,
            [0.9, 0.5, 0.9],
            outcome=1,
            total_duration_seconds=4.0,
        )
        self.assertEqual(diagnostics["mean_delay_steps"], 3.0)
        self.assertEqual(diagnostics["matched_detections"], 1.0)
        self.assertEqual(diagnostics["false_alerts_per_minute"], 15.0)

    def test_outcome_aware_sampler_places_rare_window_in_each_batch(self) -> None:
        sampler = OutcomeAwareBatchSampler(
            dataset_size=8, batch_size=2, rare_indices=[3], rare_per_batch=1, seed=2
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 4)
        self.assertTrue(all(3 in batch for batch in batches))

    def test_outcome_aware_sampler_pairs_rare_and_matched_correct_windows(self) -> None:
        sampler = OutcomeAwareBatchSampler(
            dataset_size=8,
            batch_size=4,
            rare_indices=[2, 3],
            rare_per_batch=2,
            matched_correct_candidates={2: [5], 3: [6]},
            seed=2,
        )
        for batch in sampler:
            self.assertEqual(len(batch), 4)
            self.assertTrue(
                (2 in batch and 5 in batch) or (3 in batch and 6 in batch)
            )

    def test_outcome_aware_sampler_uses_configured_sampling_weights(self) -> None:
        sampler = OutcomeAwareBatchSampler(
            dataset_size=3,
            batch_size=30,
            rare_indices=[],
            rare_per_batch=0,
            sampling_weights=np.asarray([0.0, 0.0, 1.0]),
            seed=2,
        )
        self.assertEqual(list(sampler), [[2] * 3])

    def test_guaranteed_rare_slot_does_not_require_weight_boost(self) -> None:
        sampler = OutcomeAwareBatchSampler(
            dataset_size=4,
            batch_size=2,
            rare_indices=[3],
            rare_per_batch=1,
            sampling_weights=np.ones(4),
            seed=4,
        )
        self.assertTrue(all(3 in batch for batch in sampler))
        self.assertEqual(_effective_rare_window_boost(6.0, rare_per_batch=1), 1.0)
        self.assertEqual(_effective_rare_window_boost(6.0, rare_per_batch=0), 6.0)

    def test_event_calibration_can_enforce_false_alert_budget(self) -> None:
        scores = np.zeros((20, 1, 3), dtype=np.float32)
        scores[5, 0, 1] = 0.6  # true event
        scores[15, 0, 1] = 0.9  # unavoidable higher-scoring false event
        samples = [([(5, 0, 1)], scores)]
        unconstrained, _, _ = _calibrate_event_thresholds(
            samples, outcomes=3, tolerance=0, seconds_per_step=3.0
        )
        constrained, _, _ = _calibrate_event_thresholds(
            samples,
            outcomes=3,
            tolerance=0,
            seconds_per_step=3.0,
            max_incorrect_false_alerts_per_minute=0.0,
            total_duration_seconds=60.0,
        )
        self.assertLessEqual(unconstrained[1], 0.6)
        self.assertGreater(constrained[1], 0.9)

    def test_run_directory_manifest_prevents_colliding_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                output_dir=str(Path(directory) / "run"),
                cache_index=str(Path(directory) / "cache.json"),
                resume=None,
                seed=7,
            )
            output_dir, digest = _initialize_run_directory(args)
            manifest = output_dir / "run_manifest.json"
            self.assertTrue(manifest.exists())
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                _initialize_run_directory(args)
            args.resume = str(output_dir / "last_checkpoint.pt")
            self.assertEqual(_initialize_run_directory(args)[1], digest)
            args.seed = 8
            with self.assertRaises(ValueError):
                _initialize_run_directory(args)

    def test_staged_resume_keeps_init_checkpoint_as_manifest_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "event_run"
            init_checkpoint = str(Path(directory) / "action" / "best_checkpoint.pt")
            args = Namespace(
                output_dir=str(output_dir),
                cache_index=str(Path(directory) / "cache.json"),
                resume=None,
                init_checkpoint=init_checkpoint,
                event_init_checkpoint=None,
                freeze_action_backbone=True,
                seed=7,
            )
            _, digest = _initialize_run_directory(args)

            args.resume = str(output_dir / "last_checkpoint.pt")
            self.assertEqual(_initialize_run_directory(args)[1], digest)
            self.assertFalse(_initialization_enabled(args))
            self.assertIsNone(_checkpoint_argument_error(args))

            # Omitting the provenance path is incompatible with the immutable
            # launch contract and must not silently create a different run.
            args.init_checkpoint = None
            with self.assertRaises(ValueError):
                _initialize_run_directory(args)
            self.assertEqual(
                _checkpoint_argument_error(args),
                "--freeze-action-backbone requires --init-checkpoint",
            )

    def test_fresh_staged_run_initializes_but_resume_does_not(self) -> None:
        args = Namespace(
            resume=None,
            init_checkpoint="action.pt",
            event_init_checkpoint="event.pt",
            freeze_action_backbone=True,
        )
        self.assertTrue(_initialization_enabled(args))
        self.assertIsNone(_checkpoint_argument_error(args))
        args.resume = "last_checkpoint.pt"
        self.assertFalse(_initialization_enabled(args))

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

    def test_incorrect_attempt_and_nearby_correct_retry_are_both_retained(self) -> None:
        scores = np.zeros((12, 1, 3), dtype=np.float32)
        scores[3, 0, 1] = 0.8
        scores[7, 0, 0] = 0.9
        predictions = _predicted_events_from_scores(
            scores,
            [0.5, 0.5, 0.5],
            minimum_distance=8,
            install_minimum_distance=12,
        )
        self.assertEqual(predictions, [(3, 0, 1), (7, 0, 0)])

    def test_binary_average_precision_rewards_ranked_faults(self) -> None:
        perfect = _binary_average_precision([0.9, 0.8, 0.2], [1, 1, 0])
        reversed_score = _binary_average_precision([0.1, 0.2, 0.9], [1, 1, 0])
        self.assertEqual(perfect, 100.0)
        self.assertGreater(perfect, reversed_score)


if __name__ == "__main__":
    unittest.main()
