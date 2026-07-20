from __future__ import annotations

import unittest

import numpy as np

from aiops.features.videomaev2_features import centered_clip_indices


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


if __name__ == "__main__":
    unittest.main()
