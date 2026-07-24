from __future__ import annotations

import unittest
from dataclasses import replace

from aiops.models.expected_effect import (
    EFFECT_ORDER,
    ExpectedEffectPredictorConfig,
    build_expected_effect_predictor,
    expected_effect_loss,
    expected_effect_residual,
)


class ExpectedEffectPredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")
        self.torch = torch

    def _config(self) -> ExpectedEffectPredictorConfig:
        return ExpectedEffectPredictorConfig(
            hidden_dim=16, num_components=3, num_steps=0, effect_lag=2, dropout=0.0
        )

    def test_shapes_and_probability_simplex(self) -> None:
        torch = self.torch
        model = build_expected_effect_predictor(self._config())
        features = torch.randn(2, 7, 3, 16)
        output = model(features)
        self.assertEqual(tuple(output["expected_effect"].shape), (2, 7, 3, len(EFFECT_ORDER)))
        totals = output["expected_effect"].sum(dim=-1)
        torch.testing.assert_close(totals, torch.ones_like(totals))

    def test_future_features_do_not_change_past_predictions(self) -> None:
        torch = self.torch
        model = build_expected_effect_predictor(self._config()).eval()
        first = torch.randn(2, 7, 3, 16)
        second = first.clone()
        second[:, 5:] += 100.0
        with torch.inference_mode():
            out_first = model(first)["expected_effect"]
            out_second = model(second)["expected_effect"]
        torch.testing.assert_close(out_first[:, :5], out_second[:, :5])

    def test_residual_rises_when_observed_diverges(self) -> None:
        torch = self.torch
        expected = torch.zeros(1, 1, 1, 4)
        expected[..., 1] = 1.0  # expected complete_correct
        matching = expected.clone()
        diverging = torch.zeros(1, 1, 1, 4)
        diverging[..., 2] = 1.0  # observed complete_incorrect
        low = expected_effect_residual(matching, expected)
        high = expected_effect_residual(diverging, expected)
        self.assertLess(float(low.max()), 1e-3)
        self.assertGreater(float(high.min()), 0.9)

    def test_loss_excludes_error_frames_and_is_differentiable(self) -> None:
        torch = self.torch
        model = build_expected_effect_predictor(self._config())
        features = torch.randn(2, 5, 3, 16, requires_grad=True)
        logits = model(features)["expected_effect_logits"]
        typed_effect = torch.zeros(2, 5, 3, dtype=torch.long)
        typed_effect[0, 1, 0] = 1  # complete_correct (kept)
        typed_effect[0, 2, 0] = 2  # complete_incorrect (excluded by default)
        effect_mask = torch.ones(2, 5, 3, dtype=torch.bool)
        loss = expected_effect_loss(logits, typed_effect, effect_mask)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(model.component_embedding.grad)

    def test_step_conditioning_path(self) -> None:
        torch = self.torch
        config = replace(self._config(), num_steps=4)
        model = build_expected_effect_predictor(config)
        features = torch.randn(2, 6, 3, 16)
        step_probabilities = torch.softmax(torch.randn(2, 6, 4), dim=-1)
        output = model(features, step_probabilities=step_probabilities)
        self.assertEqual(tuple(output["expected_effect"].shape), (2, 6, 3, 4))
        output["expected_effect_logits"].mean().backward()
        self.assertIsNotNone(model.step_embedding.grad)


if __name__ == "__main__":
    unittest.main()
