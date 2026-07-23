from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiops.features.industreal_roi_features import (
    _read_gaze,
    _read_hand_points,
    _read_object_detections,
    _select_active_object,
)


class IndustRealROIFeatureTest(unittest.TestCase):
    def test_release_hand_and_gaze_rows_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hands = root / "hands.csv"
            gaze = root / "gaze.csv"
            hands.write_text(
                "000007.jpg,10,20,30,40,100,110,120,130\n", encoding="utf-8"
            )
            gaze.write_text("000007.jpg,35,45\n", encoding="utf-8")
            parsed_hands = _read_hand_points(hands)
            parsed_gaze = _read_gaze(gaze)
            self.assertEqual(parsed_hands[7][0].shape, (2, 2))
            self.assertEqual(parsed_hands[7][1].shape, (2, 2))
            self.assertEqual(parsed_gaze[7], (35.0, 45.0))

    def test_coco_object_nearest_to_gaze_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "OD_labels.json"
            path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 4, "file_name": "000007.jpg"}],
                        "annotations": [
                            {"image_id": 4, "category_id": 1, "bbox": [0, 0, 20, 20]},
                            {"image_id": 4, "category_id": 2, "bbox": [80, 80, 20, 20]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            detections = _read_object_detections(path)
            selected = _select_active_object(detections[7], (85.0, 85.0), None, None)
            self.assertIsNotNone(selected)
            self.assertEqual(selected["category_id"], 2)

    def test_hand_anchor_selects_object_without_gaze(self) -> None:
        detections = [
            {"category_id": 1, "bbox": (0.0, 0.0, 20.0, 20.0)},
            {"category_id": 2, "bbox": (80.0, 80.0, 100.0, 100.0)},
        ]
        hand = np.asarray([[88.0, 91.0], [92.0, 89.0]], dtype=np.float32)
        selected = _select_active_object(detections, None, hand, None)
        self.assertEqual(selected["category_id"], 2)
