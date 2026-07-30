from __future__ import annotations

import unittest
from types import SimpleNamespace

from aiops.models.stateverify_effect import StateEffectObserverConfig
from aiops.training.train_stateverify_observer import (
    ROI_ORDER,
    _observer_inputs,
    _validate_contract,
)


class StateVerifyTrainingTest(unittest.TestCase):
    def _record(self):
        return SimpleNamespace(
            motion_dim=12,
            appearance_dim=8,
            sensor_dim=113,
            motion_aux_dim=4 * 6 + 4 + 16 + 3,
            num_components=3,
            num_completion_components=2,
        )

    def test_contract_requires_gaze_free_v2_inputs(self) -> None:
        metadata = {
            "sensor_contract": {
                "sources": ["hands", "headset_pose"],
                "gaze_excluded": True,
            },
            "motion_aux_provenance": {
                "format_version": 2,
                "roi_names": list(ROI_ORDER),
            },
        }
        self.assertEqual(_validate_contract(metadata, [self._record()]), 6)
        metadata["motion_aux_provenance"]["format_version"] = 1
        with self.assertRaisesRegex(ValueError, "format_version=2"):
            _validate_contract(metadata, [self._record()])

    def test_tensor_adapter_preserves_hands_and_excludes_extra_sensor_values(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")
        config = StateEffectObserverConfig(
            global_dim=20,
            roi_dim=6,
            num_components=3,
            hand_dim=104,
            pose_dim=9,
            hidden_dim=24,
            num_heads=4,
        )
        auxiliary = torch.zeros(2, 5, 47)
        auxiliary[..., 24:28] = 1.0
        sensor = torch.randn(2, 5, 120)
        batch = {
            "motion": torch.randn(2, 5, 12),
            "appearance": torch.randn(2, 5, 8),
            "motion_aux": auxiliary,
            "sensor": sensor,
            "modality_mask": torch.ones(2, 5, 3, dtype=torch.bool),
            "valid_mask": torch.ones(2, 5, dtype=torch.bool),
        }
        adapted = _observer_inputs(batch, config)
        self.assertEqual(tuple(adapted["hands"].shape), (2, 5, 104))
        self.assertEqual(tuple(adapted["pose"].shape), (2, 5, 9))
        torch.testing.assert_close(adapted["hands"], sensor[..., :104])
        torch.testing.assert_close(adapted["pose"], sensor[..., 104:113])
        self.assertEqual(tuple(adapted["roi_features"].shape), (2, 5, 4, 6))


if __name__ == "__main__":
    unittest.main()
