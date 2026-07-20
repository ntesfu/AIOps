from __future__ import annotations

import importlib.util
import unittest

from aiops.models.stategraph_psr import (
    StateGraphLossConfig,
    StateGraphPSRConfig,
    build_stategraph_loss,
    build_stategraph_psr,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
class StateGraphModelTest(unittest.TestCase):
    def test_shapes_causality_and_backward(self) -> None:
        import torch

        torch.manual_seed(3)
        config = StateGraphPSRConfig(
            motion_dim=8,
            appearance_dim=6,
            sensor_dim=4,
            num_steps=3,
            num_action_verbs=2,
            num_action_objects=2,
            action_verb_indices=(0, 0, 1),
            action_object_indices=(0, 1, 1),
            seen_action_mask=(True, True, False),
            num_completion_components=2,
            num_components=2,
            hidden_dim=16,
            num_temporal_blocks=2,
            attention_every=1,
            num_heads=2,
            dropout=0.0,
            max_dilation=2,
        )
        transition = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        model = build_stategraph_psr(config, transition).eval()
        motion = torch.randn(2, 7, 8)
        appearance = torch.randn(2, 7, 6)
        sensor = torch.randn(2, 7, 4)
        valid = torch.ones(2, 7, dtype=torch.bool)
        modalities = torch.ones(2, 7, 3, dtype=torch.bool)
        output = model(motion, appearance, sensor, valid, modalities)
        self.assertEqual(tuple(output["step_logits"].shape), (2, 7, 3))
        self.assertEqual(tuple(output["state_logits"].shape), (2, 7, 2, 3))
        self.assertEqual(tuple(output["completion_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["component_outcome_logits"].shape), (2, 7, 2, 3))
        self.assertTrue(bool((output["atomic_step_logits"][..., 2] == 0).all()))
        self.assertTrue(bool((output["raw_step_logits"][..., 2] != 0).any()))
        self.assertTrue(bool(((output["uncertainty"] >= 0) & (output["uncertainty"] <= 1.0001)).all()))

        changed = motion.clone()
        changed[:, 4:] += 100.0
        changed_output = model(changed, appearance, sensor, valid, modalities)
        torch.testing.assert_close(output["step_logits"][:, :4], changed_output["step_logits"][:, :4])

        targets = {
            "valid_mask": valid,
            "step": torch.randint(0, 3, (2, 7)),
            "completion": torch.randint(0, 2, (2, 7, 2)).float(),
            "component_outcome": torch.randint(0, 3, (2, 7, 2)),
            "state": torch.randint(0, 3, (2, 7, 2)),
            "state_mask": torch.ones(2, 7, 2, dtype=torch.bool),
            "boundary": torch.randint(0, 2, (2, 7)).float(),
            "next_step": torch.randint(0, 3, (2, 7)),
        }
        loss = build_stategraph_loss(StateGraphLossConfig())(output, targets, transition)
        self.assertTrue(bool(torch.isfinite(loss["total"])))
        loss["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
