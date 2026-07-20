from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiops.data.industreal import (
    audit_industreal_root,
    dense_labels_from_events,
    discover_industreal_recordings,
    read_completion_events,
    read_raw_states,
)
from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    StateGraphCacheRecord,
    build_transition_matrix,
    read_cache_index,
    save_cache_record,
    write_cache_index,
)


class IndustRealAdapterTest(unittest.TestCase):
    def test_official_layout_annotations_and_dense_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recordings" / "train" / "rec_01"
            (recording / "rgb").mkdir(parents=True)
            for frame in (0, 10, 20, 30):
                (recording / "rgb" / f"frame_{frame:06d}.jpg").touch()
            self._write_csv(
                recording / "PSR_labels_with_errors.csv",
                ["image_name", "completed_step_id", "description"],
                [
                    ["frame_000010.jpg", "step_1", "install part correctly"],
                    ["frame_000030.jpg", "step_2", "wrong connection"],
                ],
            )
            self._write_csv(
                recording / "PSR_labels_raw.csv",
                ["image_name", *[f"component_{i}" for i in range(11)]],
                [["frame_000010.jpg", *([1] * 11)]],
            )

            records = discover_industreal_recordings(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].split, "train")
            audit = audit_industreal_root(root)
            self.assertTrue(audit.official_layout)
            self.assertEqual(audit.labeled_recordings, 1)

            events = read_completion_events(records[0].psr_labels)
            self.assertEqual([event.outcome for event in events], ["correct", "incorrect"])
            step, outcome, boundary = dense_labels_from_events(
                [0, 10, 20, 30], events, {"step_1": 0, "step_2": 1}
            )
            np.testing.assert_array_equal(step, [0, 0, 1, 1])
            np.testing.assert_array_equal(outcome, [1, 1, 2, 2])
            np.testing.assert_array_equal(boundary, [0, 1, 0, 1])
            raw = read_raw_states(records[0].psr_raw_labels)
            np.testing.assert_array_equal(raw[10], np.full(11, 2))

    def test_cache_round_trip_windows_and_sparse_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, labels in enumerate(([0, 0, 1, 1], [0, 2, 2])):
                path = root / f"rec_{index}.npz"
                length = len(labels)
                save_cache_record(
                    path,
                    motion=np.ones((length, 4), dtype=np.float32),
                    appearance=np.ones((length, 3), dtype=np.float32),
                    sensor=np.ones((length, 2), dtype=np.float32),
                    modality_mask=np.ones((length, 3), dtype=np.bool_),
                    step=np.asarray(labels),
                    outcome=np.ones(length, dtype=np.int64),
                    state=np.ones((length, 2), dtype=np.int64),
                    state_mask=np.ones((length, 2), dtype=np.bool_),
                    boundary=np.zeros(length, dtype=np.float32),
                    timestamps=np.arange(length, dtype=np.float32),
                )
                records.append(StateGraphCacheRecord(f"rec_{index}", "train", path, 3, 4, 3, 2, 2))
            index_path = root / "index.json"
            write_cache_index(index_path, records, {"dataset": "synthetic"})
            metadata, restored = read_cache_index(index_path)
            self.assertEqual(metadata["dataset"], "synthetic")
            self.assertEqual(len(StateGraphCacheDataset(restored, sequence_length=2, sequence_stride=1)), 5)
            graph = build_transition_matrix(restored, 3)
            self.assertEqual(graph.shape, (3, 3))
            self.assertEqual(float(graph[1, 2]), 0.0)
            self.assertGreater(float(graph[0, 1]), 0.0)
            self.assertGreater(float(graph[0, 2]), 0.0)
            np.testing.assert_allclose(graph.sum(axis=1), 1.0)

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
