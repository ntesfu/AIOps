from __future__ import annotations

import importlib.util
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / ".test-tmp"
TEMP_ROOT.mkdir(exist_ok=True)


@contextmanager
def _workspace_temp():
    path = TEMP_ROOT / f"backbone-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _module(
    "run_industreal_backbone_ablation",
    ROOT / "scripts" / "run_industreal_backbone_ablation.py",
)
comparison = _module(
    "compare_industreal_backbones",
    ROOT / "scripts" / "compare_industreal_backbones.py",
)


class BackboneAblationTest(unittest.TestCase):
    def _cache(self, root: Path, name: str, motion_dim: int, *, offset: int = 0) -> Path:
        cache = root / name
        cache.mkdir()
        arrays = {
            "motion": np.zeros((3, motion_dim), np.float32),
            "step": np.asarray([0, 1 + offset, 1], np.int64),
            "completion": np.zeros((3, 2), np.float32),
            "component_outcome": np.zeros((3, 2), np.int64),
            "state": np.zeros((3, 2), np.int64),
            "state_mask": np.ones((3, 2), bool),
            "boundary": np.asarray([1, 1, 0], np.float32),
            "timestamps": np.asarray([0.0, 0.5, 1.0]),
            "prediction_frame_indices": np.asarray([0, 5, 10]),
        }
        np.savez_compressed(cache / "rec.npz", **arrays)
        payload = {
            "metadata": {"causal_features": True},
            "records": [
                {
                    "recording_id": "rec",
                    "split": "train",
                    "path": "rec.npz",
                    "num_steps": 2,
                    "motion_dim": motion_dim,
                    "appearance_dim": 4,
                    "sensor_dim": 3,
                    "num_components": 2,
                    "num_completion_components": 2,
                    "num_frames": 3,
                    "motion_aux_dim": 8,
                }
            ],
        }
        index = cache / "index.json"
        index.write_text(json.dumps(payload), encoding="utf-8")
        return index

    def test_cache_pair_allows_only_motion_dimension_to_change(self):
        with _workspace_temp() as root:
            result = runner.validate_cache_pair(
                self._cache(root, "swin", 768),
                self._cache(root, "vmae", 768 + 1),
            )
            self.assertEqual(result["recordings"], 1)
            self.assertTrue(result["array_labels_checked"])
            self.assertEqual(len(result["paired_label_sha256"]), 64)

    def test_cache_pair_rejects_label_drift(self):
        with _workspace_temp() as root:
            with self.assertRaisesRegex(ValueError, "step"):
                runner.validate_cache_pair(
                    self._cache(root, "swin", 768),
                    self._cache(root, "vmae", 768, offset=1),
                )

    def test_quick_comparison_applies_paired_promotion_gate(self):
        with _workspace_temp() as repo:
            config = {
                "name": "paired",
                "backbones": {"swin3d_s": {}, "videomaev2_b": {}},
                "stages": {"quick": {"seeds": [7]}},
                "promotion_gates": {
                    "minimum_paired_f1_at_50_gain_points": 2.0,
                    "maximum_frame_accuracy_regression_points": 2.0,
                    "maximum_incorrect_false_alerts_per_minute": 2.0,
                    "require_positive_incorrect_event_recall": True,
                    "maximum_peak_training_vram_gib": 23.0,
                },
            }
            for backbone, f1 in (("swin3d_s", 17.0), ("videomaev2_b", 19.5)):
                path = (
                    repo / "runs" / f"paired_quick_{backbone}_s7_action" / "metrics.json"
                )
                path.parent.mkdir(parents=True)
                metrics = {key: 0.0 for key in comparison.METRICS}
                metrics.update({"f1@50": f1, "frame_accuracy": 70.0, "edit": 40.0})
                path.write_text(
                    json.dumps(
                        {
                            "best_validation": metrics,
                            "test": None,
                            "peak_training_vram_gib": 8.0,
                        }
                    ),
                    encoding="utf-8",
                )
            report = comparison.compare(config, repo, "quick")
            self.assertTrue(report["promote_videomaev2_b"])
            self.assertAlmostEqual(
                report["paired_candidate_minus_reference"][0]["f1@50_delta"], 2.5
            )


if __name__ == "__main__":
    unittest.main()
