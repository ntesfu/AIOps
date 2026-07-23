from __future__ import annotations

import unittest
from dataclasses import fields

from aiops.models.stateverify_effect import (
    StateEffectObserverConfig,
    build_state_effect_observer,
)


class StateEffectObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")
        self.torch = torch

    def _config(self) -> StateEffectObserverConfig:
        return StateEffectObserverConfig(
            global_dim=12,
            roi_dim=8,
            num_components=3,
            roi_count=4,
            hand_dim=6,
            pose_dim=3,
            depth_dim=5,
            hidden_dim=24,
            temporal_blocks=3,
            num_heads=4,
            dropout=0.0,
            effect_lag=2,
        )

    def _inputs(self):
        torch = self.torch
        return {
            "global_features": torch.randn(2, 7, 12),
            "roi_features": torch.randn(2, 7, 4, 8),
            "roi_mask": torch.ones(2, 7, 4, dtype=torch.bool),
            "hands": torch.randn(2, 7, 6),
            "hand_mask": torch.ones(2, 7, dtype=torch.bool),
            "pose": torch.randn(2, 7, 3),
            "pose_mask": torch.ones(2, 7, dtype=torch.bool),
            "depth": torch.randn(2, 7, 5),
            "depth_mask": torch.ones(2, 7, dtype=torch.bool),
            "valid_mask": torch.ones(2, 7, dtype=torch.bool),
        }

    def test_contract_is_gaze_free_and_hand_explicit(self) -> None:
        names = {field.name for field in fields(StateEffectObserverConfig)}
        self.assertIn("hand_dim", names)
        self.assertIn("pose_dim", names)
        self.assertIn("depth_dim", names)
        self.assertNotIn("gaze_dim", names)

    def test_shapes_backward_and_effect_residual(self) -> None:
        torch = self.torch
        config = self._config()
        model = build_state_effect_observer(config)
        inputs = self._inputs()
        expected = torch.zeros(2, 7, 3, 4)
        expected[..., 1] = 1.0
        output = model(**inputs, expected_effect=expected)
        self.assertEqual(tuple(output["state_logits"].shape), (2, 7, 3, 3))
        self.assertEqual(tuple(output["effect_logits"].shape), (2, 7, 3, 4))
        self.assertEqual(tuple(output["effect_residual"].shape), (2, 7, 3))
        self.assertTrue(bool((output["effect_residual"] >= 0).all()))
        self.assertTrue(bool((output["effect_residual"] <= 1).all()))
        (
            output["state_logits"].mean()
            + output["effect_logits"].mean()
        ).backward()
        self.assertIsNotNone(model.component_queries.grad)

    def test_future_changes_do_not_affect_past_outputs(self) -> None:
        torch = self.torch
        model = build_state_effect_observer(self._config()).eval()
        first = self._inputs()
        second = {key: value.clone() for key, value in first.items()}
        second["global_features"][:, 5:] += 100.0
        second["roi_features"][:, 5:] -= 100.0
        second["hands"][:, 5:] += 50.0
        with torch.inference_mode():
            first_output = model(**first)
            second_output = model(**second)
        torch.testing.assert_close(
            first_output["state_logits"][:, :5],
            second_output["state_logits"][:, :5],
        )
        torch.testing.assert_close(
            first_output["effect_logits"][:, :5],
            second_output["effect_logits"][:, :5],
        )

    def test_masked_hand_pose_and_depth_values_are_ignored(self) -> None:
        torch = self.torch
        model = build_state_effect_observer(self._config()).eval()
        first = self._inputs()
        first["hand_mask"].zero_()
        first["pose_mask"].zero_()
        first["depth_mask"].zero_()
        second = {key: value.clone() for key, value in first.items()}
        second["hands"] += 1000.0
        second["pose"] -= 1000.0
        second["depth"] += 500.0
        with torch.inference_mode():
            first_output = model(**first)
            second_output = model(**second)
        torch.testing.assert_close(
            first_output["state_logits"], second_output["state_logits"]
        )
        torch.testing.assert_close(
            first_output["effect_logits"], second_output["effect_logits"]
        )

    def test_all_missing_rois_remain_finite(self) -> None:
        torch = self.torch
        model = build_state_effect_observer(self._config()).eval()
        inputs = self._inputs()
        inputs["roi_mask"].zero_()
        with torch.inference_mode():
            output = model(**inputs)
        self.assertTrue(bool(torch.isfinite(output["state_logits"]).all()))
        self.assertTrue(bool(torch.isfinite(output["effect_logits"]).all()))


if __name__ == "__main__":
    unittest.main()
