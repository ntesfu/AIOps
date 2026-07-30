from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiops.features.videomaev2_features import (
    _assert_resume_contract,
    centered_clip_indices,
    clip_indices,
    shard_items,
    trailing_clip_indices,
)


class VideoMAEv2FeatureTest(unittest.TestCase):
    def test_centered_clip_uses_ssv2_sampling_interval(self) -> None:
        indices = centered_clip_indices(50, 100, num_frames=16, sampling_rate=2)
        np.testing.assert_array_equal(indices, np.arange(35, 66, 2))
        self.assertEqual(len(indices), 16)

    def test_centered_clip_clamps_video_boundaries(self) -> None:
        start = centered_clip_indices(0, 20, num_frames=4, sampling_rate=2)
        end = centered_clip_indices(19, 20, num_frames=4, sampling_rate=2)
        np.testing.assert_array_equal(start, [0, 0, 1, 3])
        np.testing.assert_array_equal(end, [16, 18, 19, 19])

    def test_centered_clip_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            centered_clip_indices(0, 0)

    def test_trailing_clip_never_reads_after_prediction_frame(self) -> None:
        indices = trailing_clip_indices(50, 100, num_frames=16, sampling_rate=2)
        np.testing.assert_array_equal(indices, np.arange(20, 51, 2))
        self.assertEqual(indices[-1], 50)
        self.assertLessEqual(int(indices.max()), 50)

    def test_trailing_clip_left_pads_without_future_leakage(self) -> None:
        indices = trailing_clip_indices(3, 20, num_frames=4, sampling_rate=2)
        np.testing.assert_array_equal(indices, [0, 0, 1, 3])
        self.assertLessEqual(int(indices.max()), 3)

    def test_clip_anchor_is_explicit(self) -> None:
        np.testing.assert_array_equal(clip_indices(8, 20, 4, 2, "trailing"), [2, 4, 6, 8])
        with self.assertRaises(ValueError):
            clip_indices(8, 20, 4, 2, "unknown")

    def test_resume_rejects_legacy_centered_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "videomaev2_manifest.json").write_text(
                json.dumps(
                    {
                        "backend": "test",
                        "model_id": "model",
                        "revision": "rev",
                        "checkpoint": None,
                        "stride_frames": 5,
                        "num_frames": 16,
                        "sampling_rate": 2,
                        "pooling": "model",
                    }
                ),
                encoding="utf-8",
            )
            expected = {
                "backend": "test",
                "model_id": "model",
                "revision": "rev",
                "checkpoint": None,
                "stride_frames": 5,
                "num_frames": 16,
                "sampling_rate": 2,
                "pooling": "model",
                "clip_anchor": "trailing",
            }
            with self.assertRaisesRegex(RuntimeError, "incompatible feature contract"):
                _assert_resume_contract(root, expected)

    def test_recording_shards_are_disjoint_and_complete(self) -> None:
        items = list(range(11))
        shards = [shard_items(items, index, 3) for index in range(3)]
        self.assertEqual(sorted(item for shard in shards for item in shard), items)
        self.assertTrue(set(shards[0]).isdisjoint(shards[1]))
        with self.assertRaises(ValueError):
            shard_items(items, 3, 3)


if __name__ == "__main__":
    unittest.main()
