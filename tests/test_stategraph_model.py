from __future__ import annotations

import importlib.util
import unittest

from aiops.models.stategraph_psr import (
    StateGraphLossConfig,
    StateGraphPSRConfig,
    build_dual_expert_stategraph_psr,
    build_stategraph_loss,
    build_stategraph_psr,
    _expand_causal_event_targets,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
class StateGraphModelTest(unittest.TestCase):
    def test_dual_expert_routes_action_and_event_outputs(self) -> None:
        import torch

        config = StateGraphPSRConfig(
            motion_dim=4,
            appearance_dim=3,
            sensor_dim=2,
            num_steps=2,
            num_completion_components=2,
            num_components=2,
            hidden_dim=8,
            num_temporal_blocks=1,
            attention_every=0,
            num_heads=2,
            dropout=0.0,
        )
        model = build_dual_expert_stategraph_psr(config, config).eval()
        inputs = (
            torch.randn(1, 4, 4),
            torch.randn(1, 4, 3),
            torch.randn(1, 4, 2),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 4, 3, dtype=torch.bool),
        )
        action = model.action_expert(*inputs)
        event = model.event_expert(*inputs)
        merged = model(*inputs)
        torch.testing.assert_close(merged["step_logits"], action["step_logits"])
        torch.testing.assert_close(merged["state_logits"], event["state_logits"])
        torch.testing.assert_close(
            merged["incorrect_onset_logits"], event["incorrect_onset_logits"]
        )

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
            factorized_mistake_detection=True,
            component_evidence=True,
            event_only_motion_aux=True,
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
        self.assertEqual(tuple(output["any_mistake_onset_logits"].shape), (2, 7))
        self.assertEqual(tuple(output["incorrect_component_logits"].shape), (2, 7, 2))
        self.assertEqual(
            tuple(output["factorized_incorrect_probabilities"].shape), (2, 7, 2)
        )
        torch.testing.assert_close(
            output["factorized_incorrect_probabilities"].sum(dim=-1),
            torch.sigmoid(output["any_mistake_onset_logits"]),
        )
        torch.testing.assert_close(
            torch.sigmoid(output["factorized_incorrect_logits"]),
            output["factorized_incorrect_probabilities"],
        )
        self.assertEqual(tuple(output["procedure_violation_score"].shape), (2, 7))
        self.assertEqual(tuple(output["mistake_action_probability"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["mistake_action_onset_score"].shape), (2, 7, 2))
        self.assertTrue(bool((output["mistake_action_probability"] >= 0).all()))
        self.assertEqual(tuple(output["normality_logits"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["component_evidence_features"].shape), (2, 7, 2, 16))
        self.assertEqual(tuple(output["state_outcome_probabilities"].shape), (2, 7, 2, 3))
        self.assertEqual(output["event_state_indices"].tolist(), [0, 1])
        self.assertEqual(tuple(output["progress_logits"].shape), (2, 7))
        self.assertEqual(tuple(output["modality_gates"].shape), (2, 7, 3))
        self.assertEqual(tuple(output["modality_disagreement"].shape), (2, 7, 16))
        self.assertEqual(len(output["refinement_step_logits"]), 2)
        self.assertEqual(tuple(output["refinement_step_logits"][-1].shape), (2, 7, 3))
        self.assertTrue(bool((output["atomic_step_logits"][..., 2] == 0).all()))
        self.assertTrue(bool((output["raw_step_logits"][..., 2] != 0).any()))
        self.assertTrue(bool(((output["uncertainty"] >= 0) & (output["uncertainty"] <= 1.0001)).all()))

        changed_aux = motion_aux.clone()
        changed_aux[..., 0] += 50.0
        changed_aux_output = model(
            motion,
            appearance,
            sensor,
            valid,
            modalities,
            motion_aux=changed_aux,
        )
        torch.testing.assert_close(
            output["step_logits"], changed_aux_output["step_logits"]
        )
        self.assertFalse(
            bool(
                torch.allclose(
                    output["component_outcome_logits"],
                    changed_aux_output["component_outcome_logits"],
                )
            )
        )

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
        torch.testing.assert_close(
            output["factorized_incorrect_logits"][:, :4],
            changed_output["factorized_incorrect_logits"][:, :4],
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
        targets["component_outcome"][0, 0, 0] = 1
        loss = build_stategraph_loss(StateGraphLossConfig())(output, targets, transition)
        weighted_loss = build_stategraph_loss(
            StateGraphLossConfig(any_mistake_pos_weight=5.0)
        )(output, targets, transition)
        self.assertGreater(
            float(weighted_loss["any_mistake_onset"]),
            float(loss["any_mistake_onset"]),
        )
        self.assertTrue(bool(torch.isfinite(loss["total"])))
        self.assertIn("progress", loss)
        self.assertIn("normality", loss)
        self.assertIn("incorrect_onset", loss)
        self.assertIn("any_mistake_onset", loss)
        self.assertIn("mistake_component", loss)
        self.assertIn("refinement", loss)
        self.assertGreater(float(loss["refinement"]), 0.0)
        loss["total"].backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertIsNotNone(model.any_mistake_onset_head.weight.grad)

    def test_factorized_loss_is_safe_without_positive_mistakes(self) -> None:
        import torch

        config = StateGraphPSRConfig(
            motion_dim=4,
            appearance_dim=3,
            sensor_dim=2,
            num_steps=2,
            num_completion_components=2,
            num_components=2,
            hidden_dim=8,
            num_temporal_blocks=1,
            attention_every=0,
            num_heads=2,
            dropout=0.0,
            factorized_mistake_detection=True,
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
            "completion": torch.zeros(1, 4, 2),
            "component_outcome": torch.full((1, 4, 2), -100, dtype=torch.long),
            "state": torch.ones(1, 4, 2, dtype=torch.long),
            "state_mask": torch.ones(1, 4, 2, dtype=torch.bool),
            "boundary": torch.zeros(1, 4),
            "next_step": torch.full((1, 4), -100, dtype=torch.long),
        }
        losses = build_stategraph_loss(StateGraphLossConfig())(output, targets)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        self.assertGreater(float(losses["any_mistake_onset"]), 0.0)
        self.assertEqual(float(losses["mistake_component"]), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(model.any_mistake_onset_head.weight.grad)

    def test_legacy_model_has_no_factorized_parameters(self) -> None:
        config = StateGraphPSRConfig(
            motion_dim=4,
            appearance_dim=3,
            sensor_dim=2,
            num_steps=2,
            num_completion_components=1,
            num_components=1,
            hidden_dim=8,
            num_temporal_blocks=1,
            attention_every=0,
            num_heads=2,
        )
        model = build_stategraph_psr(config)
        self.assertFalse(hasattr(model, "any_mistake_onset_head"))

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
        targets["state"][:, :2, 1] = 0
        targets["state"][:, 2:, 1] = 2
        targets["state_mask"][:, :, 1] = True
        losses = build_stategraph_loss(
            StateGraphLossConfig(component_rank_weight=0.25)
        )(output, targets)
        self.assertTrue(bool(torch.isfinite(losses["normality"])))
        self.assertGreater(float(losses["normality"]), 0.0)
        self.assertTrue(bool(torch.isfinite(losses["component_rank"])))
        self.assertGreater(float(losses["component_rank"]), 0.0)

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

    def test_hard_negative_mask_keeps_all_positives_and_top_negatives(self) -> None:
        import torch

        criterion = build_stategraph_loss(StateGraphLossConfig())
        logits = torch.tensor([[[0.0, 4.0], [3.0, 2.0], [1.0, -1.0]]])
        targets = torch.zeros_like(logits)
        targets[0, 0, 0] = 1.0
        mask = criterion._hard_negative_mask(
            logits, targets, torch.ones(1, 3, dtype=torch.bool), ratio=2.0
        )
        self.assertTrue(bool(mask[0, 0, 0]))
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(bool(mask[0, 0, 1]))
        self.assertTrue(bool(mask[0, 1, 0]))


if __name__ == "__main__":
    unittest.main()
