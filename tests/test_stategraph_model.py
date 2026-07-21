from __future__ import annotations

import importlib.util
import unittest

from aiops.models.stategraph_psr import (
    StateGraphLossConfig,
    StateGraphPSRConfig,
    build_stategraph_loss,
    build_stategraph_psr,
    _expand_causal_event_targets,
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
            motion_aux_dim=5,
            num_action_verbs=2,
            num_action_objects=2,
            action_verb_indices=(0, 0, 1),
            action_object_indices=(0, 1, 1),
            seen_action_mask=(True, True, False),
            mistake_action_mask=(False, True, False),
            action_event_component_indices=(0, 1, -1),
            event_state_indices=(0, 1),
            num_completion_components=2,
            num_components=2,
            hidden_dim=16,
            num_temporal_blocks=2,
            attention_every=1,
            num_heads=2,
            num_action_refinement_stages=2,
            num_refinement_blocks=2,
            num_event_blocks=2,
            dropout=0.0,
            max_dilation=2,
            procedural_event_context=True,
            learned_event_fusion=True,
        )
        transition = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        model = build_stategraph_psr(config, transition).eval()
        motion = torch.randn(2, 7, 8)
        appearance = torch.randn(2, 7, 6)
        sensor = torch.randn(2, 7, 4)
        motion_aux = torch.randn(2, 7, 5)
        valid = torch.ones(2, 7, dtype=torch.bool)
        modalities = torch.ones(2, 7, 3, dtype=torch.bool)
        output = model(
            motion,
            appearance,
            sensor,
            valid,
            modalities,
            motion_aux=motion_aux,
        )
        self.assertEqual(tuple(output["step_logits"].shape), (2, 7, 3))
        self.assertEqual(tuple(output["state_logits"].shape), (2, 7, 2, 3))
        self.assertEqual(tuple(output["completion_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["component_outcome_logits"].shape), (2, 7, 2, 3))
        self.assertEqual(tuple(output["incorrect_onset_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["fused_incorrect_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["procedure_violation_score"].shape), (2, 7))
        self.assertEqual(tuple(output["mistake_action_probability"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["mistake_action_onset_score"].shape), (2, 7, 2))
        self.assertTrue(bool((output["mistake_action_probability"] >= 0).all()))
        self.assertEqual(tuple(output["normality_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["state_outcome_probabilities"].shape), (2, 7, 2, 3))
        self.assertEqual(output["event_state_indices"].tolist(), [0, 1])
        self.assertEqual(tuple(output["progress_logits"].shape), (2, 7))
        self.assertEqual(tuple(output["modality_gates"].shape), (2, 7, 4))
        self.assertEqual(tuple(output["modality_disagreement"].shape), (2, 7, 16))
        self.assertEqual(len(output["refinement_step_logits"]), 2)
        self.assertEqual(tuple(output["refinement_step_logits"][-1].shape), (2, 7, 3))
        self.assertTrue(bool((output["atomic_step_logits"][..., 2] == 0).all()))
        self.assertTrue(bool((output["raw_step_logits"][..., 2] != 0).any()))
        self.assertTrue(bool(((output["uncertainty"] >= 0) & (output["uncertainty"] <= 1.0001)).all()))

        changed = motion.clone()
        changed[:, 4:] += 100.0
        changed_output = model(
            changed,
            appearance,
            sensor,
            valid,
            modalities,
            motion_aux=motion_aux,
        )
        torch.testing.assert_close(output["step_logits"][:, :4], changed_output["step_logits"][:, :4])
        torch.testing.assert_close(
            output["component_outcome_logits"][:, :4],
            changed_output["component_outcome_logits"][:, :4],
        )
        torch.testing.assert_close(
            output["fused_incorrect_logits"][:, :4],
            changed_output["fused_incorrect_logits"][:, :4],
        )

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
        self.assertIn("progress", loss)
        self.assertIn("normality", loss)
        self.assertIn("incorrect_onset", loss)
        self.assertIn("refinement", loss)
        self.assertGreater(float(loss["refinement"]), 0.0)
        loss["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_dense_incorrect_states_supervise_normality_without_events(self) -> None:
        import torch

        config = StateGraphPSRConfig(
            motion_dim=4,
            appearance_dim=3,
            sensor_dim=2,
            num_steps=2,
            event_state_indices=(1,),
            num_completion_components=1,
            num_components=2,
            hidden_dim=8,
            num_temporal_blocks=1,
            attention_every=0,
            num_heads=2,
            dropout=0.0,
        )
        model = build_stategraph_psr(config)
        valid = torch.ones(1, 4, dtype=torch.bool)
        output = model(
            torch.randn(1, 4, 4),
            torch.randn(1, 4, 3),
            torch.randn(1, 4, 2),
            valid,
            torch.ones(1, 4, 3, dtype=torch.bool),
        )
        targets = {
            "valid_mask": valid,
            "step": torch.zeros(1, 4, dtype=torch.long),
            "completion": torch.zeros(1, 4, 1),
            "component_outcome": torch.full((1, 4, 1), -100, dtype=torch.long),
            "state": torch.ones(1, 4, 2, dtype=torch.long),
            "state_mask": torch.zeros(1, 4, 2, dtype=torch.bool),
            "boundary": torch.zeros(1, 4),
            "next_step": torch.full((1, 4), -100, dtype=torch.long),
        }
        targets["state"][:, :, 1] = 0
        targets["state_mask"][:, :, 1] = True
        losses = build_stategraph_loss(StateGraphLossConfig())(output, targets)
        self.assertTrue(bool(torch.isfinite(losses["normality"])))
        self.assertGreater(float(losses["normality"]), 0.0)

    def test_event_targets_expand_forward_without_overwriting_events(self) -> None:
        import torch

        completion = torch.zeros(1, 6, 1)
        completion[0, 1, 0] = 1
        completion[0, 3, 0] = 1
        outcomes = torch.full((1, 6, 1), -100, dtype=torch.long)
        outcomes[0, 1, 0] = 1
        outcomes[0, 3, 0] = 2
        valid = torch.tensor([[True, True, True, True, True, False]])
        expanded_completion, expanded_outcomes = _expand_causal_event_targets(
            completion, outcomes, valid, horizon=2
        )
        self.assertEqual(expanded_outcomes[0, :, 0].tolist(), [-100, 1, 1, 2, 2, -100])
        self.assertEqual(expanded_completion[0, :, 0].tolist(), [0, 1, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
