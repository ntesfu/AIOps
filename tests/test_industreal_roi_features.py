from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiops.features.industreal_roi_features import (
    ROI_NAMES,
    _interaction_context_box,
    _normalized_hand_object_distance,
    _read_hand_points,
    _read_object_detections,
    _select_active_object,
)


class IndustRealROIFeatureTest(unittest.TestCase):
    def test_release_hand_rows_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hands = root / "hands.csv"
            hands.write_text(
                "000007.jpg,10,20,30,40,100,110,120,130\n", encoding="utf-8"
            )
            parsed_hands = _read_hand_points(hands)
            self.assertEqual(parsed_hands[7][0].shape, (2, 2))
            self.assertEqual(parsed_hands[7][1].shape, (2, 2))

    def test_roi_contract_is_gaze_free(self) -> None:
        self.assertEqual(
            ROI_NAMES,
            ("left_hand", "right_hand", "active_object", "interaction_context"),
        )

    def test_largest_object_is_selected_without_hand_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "OD_labels.json"
            path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 4, "file_name": "000007.jpg"}],
                        "annotations": [
                            {"image_id": 4, "category_id": 1, "bbox": [0, 0, 40, 40]},
                            {"image_id": 4, "category_id": 2, "bbox": [80, 80, 20, 20]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            detections = _read_object_detections(path)
            selected = _select_active_object(detections[7], None, None)
            self.assertIsNotNone(selected)
            self.assertEqual(selected["category_id"], 1)

    def test_hand_anchor_selects_object_without_gaze(self) -> None:
        detections = [
            {"category_id": 1, "bbox": (0.0, 0.0, 20.0, 20.0)},
            {"category_id": 2, "bbox": (80.0, 80.0, 100.0, 100.0)},
        ]
        hand = np.asarray([[88.0, 91.0], [92.0, 89.0]], dtype=np.float32)
        selected = _select_active_object(detections, hand, None)
        self.assertEqual(selected["category_id"], 2)

    def test_interaction_context_covers_hands_and_object(self) -> None:
        context = _interaction_context_box(
            ((10.0, 20.0, 30.0, 40.0), None, (60.0, 50.0, 80.0, 70.0)),
            width=100,
            height=100,
            scale=1.0,
        )
        self.assertIsNotNone(context)
        assert context is not None
        self.assertLessEqual(context[0], 10.0)
        self.assertLessEqual(context[1], 20.0)
        self.assertGreaterEqual(context[2], 80.0)
        self.assertGreaterEqual(context[3], 70.0)

    def test_hand_object_distance_is_normalized(self) -> None:
        hand = np.asarray([[10.0, 10.0], [12.0, 12.0]], dtype=np.float32)
        distance = _normalized_hand_object_distance(
            (8.0, 8.0, 14.0, 14.0), hand, None, width=100, height=100
        )
        self.assertGreaterEqual(distance, 0.0)
        self.assertLess(distance, 0.05)
