from __future__ import annotations

import json
import unittest

import numpy as np

from aiops.validation.selection_verifier import (
    SelectionVerifierConfig,
    StepExpectation,
    active_object_categories_from_motion_aux,
    load_selection_schema,
    selection_residual,
    selection_residual_from_motion_aux,
)


def _config() -> SelectionVerifierConfig:
    # 2 ROIs x 3 dims visual, active object = ROI index 1, 3 state components.
    return SelectionVerifierConfig(
        roi_count=2,
        roi_dim=3,
        active_object_roi_index=1,
        num_state_components=3,
        event_state_indices=(0, 2),
    )


def _motion_aux(categories, present):
    config = _config()
    width = config.visual_width + config.roi_count + 4 * config.roi_count + 3
    length = len(categories)
    aux = np.zeros((length, width), dtype=np.float32)
    aux[:, config.active_object_present_index()] = np.asarray(present, dtype=np.float32)
    aux[:, config.active_object_category_index()] = (
        np.asarray(categories, dtype=np.float32) / config.category_scale
    )
    return aux


class SelectionVerifierTest(unittest.TestCase):
    def test_category_recovery_and_presence(self) -> None:
        config = _config()
        aux = _motion_aux([7, 12, 4], [True, False, True])
        categories, present = active_object_categories_from_motion_aux(aux, config)
        self.assertEqual(categories.tolist(), [7, -1, 4])
        self.assertEqual(present.tolist(), [True, False, True])

    def test_mismatch_fires_on_mapped_component(self) -> None:
        config = _config()
        # Step 0 expects category {5}; affects event component 0 -> state 0.
        schema = {0: StepExpectation(frozenset({5}), (0,))}
        categories = np.array([5, 9])
        present = np.array([True, True])
        steps = np.array([0, 0])
        residual, mask = selection_residual(categories, present, steps, schema, config)
        # Row 0 correct part -> zero residual but observed (mask True).
        self.assertTrue(mask[0, 0])
        self.assertEqual(residual[0, 0], 0.0)
        # Row 1 wrong part -> residual 1.0 on state component 0.
        self.assertTrue(mask[1, 0])
        self.assertEqual(residual[1, 0], 1.0)
        # Unaffected components stay masked-neutral.
        self.assertFalse(mask[:, 1].any())
        self.assertFalse(mask[:, 2].any())

    def test_absent_object_and_unknown_step_are_masked(self) -> None:
        config = _config()
        schema = {0: StepExpectation(frozenset({5}), (0,))}
        # Row 0: object absent -> masked. Row 1: step 3 has no expectation -> masked.
        categories = np.array([9, 9])
        present = np.array([False, True])
        steps = np.array([0, 3])
        residual, mask = selection_residual(categories, present, steps, schema, config)
        self.assertFalse(mask.any())
        self.assertEqual(residual.sum(), 0.0)

    def test_near_zero_false_positive_on_correct_data(self) -> None:
        config = _config()
        schema = {
            0: StepExpectation(frozenset({5}), (0,)),
            1: StepExpectation(frozenset({8}), (1,)),
        }
        # Correct execution: right part for each step, object present throughout.
        steps = np.array([0, 0, 1, 1])
        categories = np.array([5, 5, 8, 8])
        present = np.array([True, True, True, True])
        residual, mask = selection_residual(categories, present, steps, schema, config)
        self.assertEqual(residual.sum(), 0.0)  # no false selection alerts
        self.assertTrue(mask.any())  # but the verifier did observe the frames

    def test_from_motion_aux_matches_manual(self) -> None:
        config = _config()
        schema = {0: StepExpectation(frozenset({5}), (0,))}
        aux = _motion_aux([5, 9], [True, True])
        steps = np.array([0, 0])
        residual, mask = selection_residual_from_motion_aux(aux, steps, schema, config)
        self.assertEqual(residual[1, 0], 1.0)
        self.assertEqual(residual[0, 0], 0.0)

    def test_schema_loads_from_dict_and_json(self) -> None:
        parsed = load_selection_schema(
            {"2": {"expected_categories": [1, 3], "components": [4]}}
        )
        self.assertIn(2, parsed)
        self.assertEqual(parsed[2].expected_categories, frozenset({1, 3}))
        self.assertEqual(parsed[2].components, (4,))
        # A full schema-shaped JSON string routes through the "step_expectations" key.
        payload = json.dumps(
            {"step_expectations": {"5": {"expected_categories": [7], "components": [0]}}}
        )
        # load_selection_schema accepts a mapping; emulate the nested field.
        nested = load_selection_schema(json.loads(payload)["step_expectations"])
        self.assertEqual(nested[5].expected_categories, frozenset({7}))

    def test_empty_schema_is_noop(self) -> None:
        config = _config()
        residual, mask = selection_residual(
            np.array([1, 2]), np.array([True, True]), np.array([0, 1]), {}, config
        )
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
