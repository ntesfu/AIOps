from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from aiops.models.stategraph_psr import StateGraphPSRConfig, build_stategraph_psr


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extend_stategraph_hybrid_roi_checkpoint",
    ROOT / "scripts" / "extend_stategraph_hybrid_roi_checkpoint.py",
)
assert SPEC and SPEC.loader
EXTENSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTENSION)


class HybridCheckpointExtensionTest(unittest.TestCase):
    def test_extension_preserves_flat_outputs_at_zero_gate(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")

        torch.manual_seed(31)
        config = StateGraphPSRConfig(
            motion_dim=4,
            motion_aux_dim=18,
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
            component_evidence=True,
            event_only_motion_aux=True,
            roi_token_count=2,
            roi_embedding_dim=0,
            roi_global_dim=2,
            roi_change_lag=2,
        )
        source_model = build_stategraph_psr(config).eval()
        motion = torch.randn(1, 5, 4)
        appearance = torch.randn(1, 5, 3)
        sensor = torch.randn(1, 5, 2)
        auxiliary = torch.randn(1, 5, 18)
        auxiliary[..., 6:8] = 1.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "flat.pt"
            target_path = root / "hybrid.pt"
            torch.save(
                {
                    "model_config": config.to_dict(),
                    "model_state": source_model.state_dict(),
                    "transition_matrix": None,
                },
                source_path,
            )
            report = EXTENSION.extend_checkpoint(
                source_path,
                target_path,
                roi_token_count=2,
                roi_embedding_dim=3,
                roi_global_dim=2,
                roi_change_lag=2,
            )
            target = torch.load(target_path, map_location="cpu", weights_only=False)
            target_model = build_stategraph_psr(
                StateGraphPSRConfig(**target["model_config"])
            ).eval()
            target_model.load_state_dict(target["model_state"])

        with torch.no_grad():
            source_output = source_model(
                motion, appearance, sensor, motion_aux=auxiliary
            )
            target_output = target_model(
                motion, appearance, sensor, motion_aux=auxiliary
            )
        torch.testing.assert_close(
            source_output["step_logits"], target_output["step_logits"]
        )
        torch.testing.assert_close(
            source_output["component_outcome_logits"],
            target_output["component_outcome_logits"],
        )
        torch.testing.assert_close(
            target_output["structured_roi_gate"], torch.zeros(2)
        )
        self.assertGreater(len(report["new_parameter_tensors"]), 0)


if __name__ == "__main__":
    unittest.main()
