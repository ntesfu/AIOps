from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from aiops.data.counterfactuals import (
    CounterfactualConfig,
    CounterfactualWindowDataset,
    assert_train_only,
    collect_wrong_part_donors,
    donor_pool_excluding,
    extract_evidence_visual,
    synthesize_incorrect_event,
)
from aiops.data.stategraph_cache import StateGraphCacheRecord


def _config() -> CounterfactualConfig:
    # Small ROI dims keep the synthetic motion_aux readable: 2 ROIs x 3 dims
    # visual block, then 2 presence flags, then geometry/extras we do not touch.
    return CounterfactualConfig(
        roi_count=2,
        roi_dim=3,
        evidence_roi_index=1,
        event_state_indices=(0, 2),
        evidence_radius=1,
    )


def _sample(outcome_rows, num_components=2, num_state=3, length=6):
    config = _config()
    motion_aux_dim = config.visual_width + config.roi_count + 5
    motion_aux = np.zeros((length, motion_aux_dim), dtype=np.float32)
    # Active-object present everywhere; give the active-object visual a value
    # that identifies the (unmodified) original evidence.
    motion_aux[:, config.evidence_present_index()] = 1.0
    motion_aux[:, config.evidence_slice()] = 0.11
    component_outcome = np.full((length, num_components), -100, dtype=np.int64)
    for row, component, value in outcome_rows:
        component_outcome[row, component] = value
    state = np.full((length, num_state), 1, dtype=np.int64)  # raw pending
    state_mask = np.ones((length, num_state), dtype=bool)
    return {
        "motion_aux": motion_aux,
        "component_outcome": component_outcome,
        "state": state,
        "state_mask": state_mask,
    }


class CounterfactualTest(unittest.TestCase):
    def test_leakage_guard_rejects_non_train_records(self) -> None:
        records = [
            StateGraphCacheRecord("r0", "train", Path("r0.npz"), 1, 1, 1, 1, 1),
            StateGraphCacheRecord("r1", "val", Path("r1.npz"), 1, 1, 1, 1, 1),
        ]
        with self.assertRaises(ValueError):
            assert_train_only(records)
        # Train-only passes silently.
        assert_train_only([records[0]])

    def test_donor_pool_excludes_same_component(self) -> None:
        config = _config()
        # Component 0 correct install at row 2; component 1 correct at row 4.
        donor_source = _sample([(2, 0, 0), (4, 1, 0)])
        # Make each component's active-object evidence distinguishable.
        donor_source["motion_aux"][2, config.evidence_slice()] = 0.5
        donor_source["motion_aux"][4, config.evidence_slice()] = 0.9
        donors = collect_wrong_part_donors([donor_source], config)
        self.assertEqual(set(donors), {0, 1})
        pool_for_0 = donor_pool_excluding(donors, 0)
        # Only component 1's embedding (0.9) is a valid wrong part for component 0.
        self.assertEqual(pool_for_0.shape, (1, config.roi_dim))
        self.assertTrue(np.allclose(pool_for_0[0], 0.9))

    def test_synthesis_flips_outcome_state_and_evidence(self) -> None:
        config = _config()
        rng = np.random.default_rng(0)
        # A window with a single correct install of component 0 at row 3.
        sample = _sample([(3, 0, 0)])
        # After a correct install the persistent state becomes raw-correct (2).
        sample["state"][3:, 0] = 2
        donors = {1: np.full((1, config.roi_dim), 0.77, dtype=np.float32)}
        result = synthesize_incorrect_event(sample, rng, donors, config)
        self.assertIsNotNone(result)
        assert result is not None
        # Outcome flipped correct(0) -> incorrect(1).
        self.assertEqual(int(result["component_outcome"][3, 0]), config.incorrect_outcome)
        # Persistent state relabelled correct(2) -> incorrect(0) from the event on.
        self.assertTrue(bool((result["state"][3:, 0] == config.raw_state_incorrect).all()))
        # Active-object evidence replaced by the donor around the event.
        visual = extract_evidence_visual(result["motion_aux"], config)
        self.assertTrue(np.allclose(visual[3], 0.77))
        # Provenance recorded.
        self.assertTrue(result["is_counterfactual"])
        self.assertEqual(result["counterfactual_events"], [(3, 0)])

    def test_synthesis_does_not_mutate_source(self) -> None:
        config = _config()
        sample = _sample([(2, 0, 0)])
        sample["state"][2:, 0] = 2
        original_outcome = sample["component_outcome"].copy()
        original_state = sample["state"].copy()
        original_aux = sample["motion_aux"].copy()
        donors = {1: np.full((1, config.roi_dim), 0.42, dtype=np.float32)}
        synthesize_incorrect_event(sample, np.random.default_rng(1), donors, config)
        np.testing.assert_array_equal(sample["component_outcome"], original_outcome)
        np.testing.assert_array_equal(sample["state"], original_state)
        np.testing.assert_array_equal(sample["motion_aux"], original_aux)

    def test_synthesis_returns_none_without_donor(self) -> None:
        config = _config()
        sample = _sample([(3, 0, 0)])
        # Only component 0 has a donor, which is an invalid wrong part for itself.
        donors = {0: np.full((1, config.roi_dim), 0.5, dtype=np.float32)}
        self.assertIsNone(
            synthesize_incorrect_event(sample, np.random.default_rng(2), donors, config)
        )

    def test_wrapper_rate_zero_is_identity(self) -> None:
        config = _config()
        sample = _sample([(3, 0, 0)])
        wrapper = CounterfactualWindowDataset(
            [sample], donors={1: np.ones((1, config.roi_dim), np.float32)},
            config=config, rate=0.0,
        )
        out = wrapper[0]
        self.assertNotIn("is_counterfactual", out)

    def test_wrapper_rate_one_synthesizes(self) -> None:
        config = _config()
        sample = _sample([(3, 0, 0)])
        sample["state"][3:, 0] = 2
        wrapper = CounterfactualWindowDataset(
            [sample], donors={1: np.full((1, config.roi_dim), 0.3, np.float32)},
            config=config, rate=1.0,
        )
        out = wrapper[0]
        self.assertTrue(out.get("is_counterfactual", False))


if __name__ == "__main__":
    unittest.main()
