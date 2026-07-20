from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from aiops.data.industreal import (
    ActionSegment,
    CompletionEvent,
    audit_industreal_root,
    dense_action_labels,
    dense_labels_from_events,
    discover_industreal_recordings,
    multi_component_targets_from_events,
    read_action_segments,
    read_completion_events,
    read_raw_states,
)
from aiops.data.procedure_schema import ProcedureSchema
from aiops.data.audit_stategraph_cache import audit_stategraph_cache
from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    StateGraphCacheRecord,
    build_transition_matrix,
    read_cache_index,
    save_cache_record,
    write_cache_index,
)
from aiops.features.industreal_cache import build_industreal_cache


class IndustRealAdapterTest(unittest.TestCase):
    def test_headerless_release_csv_and_root_level_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recordings" / "train" / "27_main_0_1"
            recording.mkdir(parents=True)
            (root / "27_main_0_1.mp4").touch()
            self._write_csv(
                recording / "PSR_labels_with_errors.csv",
                [],
                [
                    ["000094.jpg", 17, "Remove front rear chassis pin"],
                    ["000173.jpg", 32, "Wrong rear wheel connection"],
                ],
            )
            self._write_csv(
                recording / "PSR_labels_raw.csv",
                [],
                [["000000.jpg", *([1] * 11)], ["000094.jpg", *([0] * 11)]],
            )

            records = discover_industreal_recordings(root)
            self.assertEqual(records[0].video_path, root / "27_main_0_1.mp4")
            self.assertIsNone(records[0].rgb_dir)
            events = read_completion_events(records[0].psr_labels)
            self.assertEqual([event.frame_index for event in events], [94, 173])
            self.assertEqual([event.step_id for event in events], ["17", "32"])
            self.assertEqual([event.outcome for event in events], ["remove", "incorrect"])
            raw = read_raw_states(records[0].psr_raw_labels)
            np.testing.assert_array_equal(raw[0], np.full(11, 2))
            np.testing.assert_array_equal(raw[94], np.full(11, 1))
            self.assertTrue(audit_industreal_root(root).official_layout)

            self._write_csv(
                recording / "AR_labels.csv",
                [],
                [["27_main_0_1", 5, "attach bracket", "000020.jpg", "000040.jpg"]],
            )
            actions = read_action_segments(recording / "AR_labels.csv")
            self.assertEqual((actions[0].start_frame, actions[0].end_frame), (20, 40))

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

    def test_action_timeline_and_simultaneous_component_events_are_separate(self) -> None:
        actions, boundary = dense_action_labels(
            [0, 5, 10, 15],
            [ActionSegment(0, 0, "idle", "idle"), ActionSegment(10, 10, "attach", "attach")],
            {"idle": 0, "attach": 1},
        )
        np.testing.assert_array_equal(actions, [0, 0, 1, 1])
        np.testing.assert_array_equal(boundary, [1, 0, 1, 0])
        events = [
            CompletionEvent(10, "3", "front chassis", "correct"),
            CompletionEvent(10, "6", "front chassis pin", "correct"),
            CompletionEvent(15, "4", "incorrect front chassis", "incorrect"),
        ]
        schema = ProcedureSchema.from_dict(
            {
                "schema_version": 2,
                "name": "test",
                "completion_components": ["chassis", "pin"],
                "raw_completion_map": {"3": "chassis", "4": "chassis", "6": "pin"},
            }
        )
        completion, component_outcome = multi_component_targets_from_events(
            [0, 5, 10, 15], events, schema.resolve_component, 2
        )
        np.testing.assert_array_equal(completion[2], [1, 1])
        np.testing.assert_array_equal(component_outcome[2], [0, 0])
        np.testing.assert_array_equal(component_outcome[3], [1, -100])

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
                    completion=np.zeros((length, 2), dtype=np.float32),
                    component_outcome=np.full((length, 2), -100, dtype=np.int64),
                    state=np.ones((length, 2), dtype=np.int64),
                    state_mask=np.ones((length, 2), dtype=np.bool_),
                    boundary=np.zeros(length, dtype=np.float32),
                    timestamps=np.arange(length, dtype=np.float32),
                )
                records.append(StateGraphCacheRecord(f"rec_{index}", "train", path, 3, 4, 3, 2, 2, 2))
            index_path = root / "index.json"
            write_cache_index(index_path, records, {"dataset": "synthetic"})
            metadata, restored = read_cache_index(index_path)
            self.assertEqual(metadata["dataset"], "synthetic")
            dataset = StateGraphCacheDataset(restored, sequence_length=2, sequence_stride=1)
            self.assertEqual(len(dataset), 5)
            weights = dataset.window_sampling_weights(incorrect_outcome_index=1, rare_window_boost=4.0)
            np.testing.assert_array_equal(weights, np.ones(5))
            with self.assertRaises(ValueError):
                dataset.window_sampling_weights(rare_window_boost=0.5)
            graph = build_transition_matrix(restored, 3)
            self.assertEqual(graph.shape, (3, 3))
            self.assertEqual(float(graph[1, 2]), 0.0)
            self.assertGreater(float(graph[0, 1]), 0.0)
            self.assertGreater(float(graph[0, 2]), 0.0)
            np.testing.assert_allclose(graph.sum(axis=1), 1.0)

    def test_cache_builder_writes_v2_action_and_multilabel_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "data" / "recordings" / "train" / "rec_01"
            rgb = recording / "rgb"
            rgb.mkdir(parents=True)
            for frame in range(4):
                (rgb / f"frame_{frame:06d}.jpg").touch()
            self._write_csv(
                recording / "AR_labels.csv",
                ["frame", "action_id", "description"],
                [[0, "idle", "idle"], [2, "attach", "attach"]],
            )
            self._write_csv(
                recording / "PSR_labels_with_errors.csv",
                ["frame", "step_id", "description"],
                [[2, "a", "part a"], [2, "b", "part b"]],
            )
            schema_path = root / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "name": "synthetic_v2",
                        "completion_components": ["part_a", "part_b"],
                        "state_components": ["state_a", "state_b"],
                        "raw_completion_map": {"a": "part_a", "b": "part_b"},
                    }
                ),
                encoding="utf-8",
            )
            old_cache = root / "old_cache" / "train"
            old_cache.mkdir(parents=True)
            np.savez_compressed(
                old_cache / "rec_01.npz",
                motion=np.ones((4, 3), dtype=np.float32),
                appearance=np.ones((4, 2), dtype=np.float32),
            )
            output = root / "cache"
            result = build_industreal_cache(
                Namespace(
                    data_root=str(root / "data"),
                    output_dir=str(output),
                    motion_features_dir=str(root / "old_cache"),
                    appearance_features_dir=str(root / "old_cache"),
                    recording_id=None,
                    max_recordings=None,
                    device="cpu",
                    mixed_precision="bf16",
                    fps=10.0,
                    stride_frames=1,
                    clip_frames=2,
                    clip_span_frames=2,
                    sensor_dim=2,
                    num_components=2,
                    procedure_schema=str(schema_path),
                )
            )
            metadata, records = read_cache_index(result["index"])
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["action_ids"], ["attach", "idle"])
            self.assertEqual(metadata["action_descriptions"], ["attach", "idle"])
            self.assertEqual(records[0].num_steps, 2)
            self.assertEqual(records[0].num_frames, 4)
            self.assertEqual(records[0].num_completion_components, 2)
            with np.load(records[0].path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["step"], [1, 1, 0, 0])
                np.testing.assert_array_equal(arrays["completion"][2], [1, 1])
            audit = audit_stategraph_cache(result["index"])
            self.assertEqual(audit["schema_version"], 2)
            self.assertEqual(audit["multi_component_rows"]["train"], 1)
            self.assertEqual(audit["warnings"], [])

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if header:
                writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
