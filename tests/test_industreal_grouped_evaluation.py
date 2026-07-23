from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiops.data.stategraph_cache import StateGraphCacheRecord, write_cache_index
from aiops.evaluation.industreal_grouped import (
    materialize_grouped_fold_indexes,
    parse_industreal_operator_id,
    prepare_grouped_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_industreal_grouped_reference",
    ROOT / "scripts" / "summarize_industreal_grouped_reference.py",
)
assert SUMMARY_SPEC and SUMMARY_SPEC.loader
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


class IndustRealGroupedEvaluationTest(unittest.TestCase):
    def test_operator_parser_is_strict_and_supports_override(self) -> None:
        self.assertEqual(parse_industreal_operator_id("01_assy_0_1"), "01")
        self.assertEqual(
            parse_industreal_operator_id("custom-session", {"custom-session": "operator-a"}),
            "operator-a",
        )
        with self.assertRaises(ValueError):
            parse_industreal_operator_id("custom-session")

    def test_folds_are_operator_disjoint_and_test_arrays_remain_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for operator, split in (("01", "train"), ("02", "train"), ("03", "val"), ("99", "test")):
                path = root / split / f"{operator}_assy_0_1.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if split == "test":
                    # Not a valid NPZ: construction succeeds only if test is never opened.
                    path.write_text("sealed", encoding="utf-8")
                else:
                    self._write_record(path, incorrect=(operator != "02"))
                records.append(self._record(operator, split, path))
            index = root / "index.json"
            write_cache_index(
                index,
                records,
                {"event_outcomes": ["correct", "incorrect", "remove"]},
            )

            payload = prepare_grouped_evaluation(
                index,
                folds=3,
                episode_radius=2,
                matches_per_incorrect=1,
            )

            self.assertTrue(payload["policy"]["official_test_sealed"])
            self.assertEqual(payload["development_recordings"], 3)
            self.assertNotIn("99", payload["development_operators"])
            seen_validation = []
            for fold in payload["folds"]:
                train = set(fold["train_operator_ids"])
                validation = set(fold["validation_operator_ids"])
                self.assertTrue(train.isdisjoint(validation))
                seen_validation.extend(validation)
                self.assertFalse(
                    any(recording.startswith("99_") for recording in fold["train_recording_ids"])
                )
            self.assertEqual(sorted(seen_validation), ["01", "02", "03"])

            fold_indexes = materialize_grouped_fold_indexes(
                index, payload, root / "fold-indexes"
            )
            self.assertEqual(len(fold_indexes), 3)
            for fold_index in fold_indexes:
                fold_payload = json.loads(fold_index.read_text(encoding="utf-8"))
                ids = {
                    record["recording_id"] for record in fold_payload["records"]
                }
                self.assertEqual(
                    ids, {"01_assy_0_1", "02_assy_0_1", "03_assy_0_1"}
                )
                self.assertFalse(any(recording.startswith("99_") for recording in ids))
                self.assertEqual(
                    sum(record["split"] == "val" for record in fold_payload["records"]),
                    1,
                )
                self.assertTrue(
                    fold_payload["metadata"]["grouped_evaluation"][
                        "official_test_sealed"
                    ]
                )

    def test_zero_error_operators_balance_recording_load(self) -> None:
        from aiops.evaluation.industreal_grouped import _balanced_operator_assignment

        records = []
        operators = {}
        events = {}
        for ordinal in range(10):
            operator = f"{ordinal:02d}"
            record = self._record(operator, "train", Path(f"/unused/{operator}.npz"))
            records.append(record)
            operators[record.recording_id] = operator
            events[record.recording_id] = []
        assignment = _balanced_operator_assignment(
            records, operators, events, folds=5, seed=7
        )
        fold_counts = [
            sum(fold == fold_index for fold in assignment.values())
            for fold_index in range(5)
        ]
        self.assertEqual(fold_counts, [2, 2, 2, 2, 2])

    def test_matching_is_causal_component_aligned_and_partition_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for operator in ("01", "02", "03"):
                path = root / "train" / f"{operator}_assy_0_1.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                self._write_record(path, incorrect=True)
                records.append(self._record(operator, "train", path))
            index = root / "index.json"
            write_cache_index(
                index,
                records,
                {"event_outcomes": ["correct", "incorrect", "remove"]},
            )

            payload = prepare_grouped_evaluation(
                index,
                folds=3,
                episode_radius=3,
                matches_per_incorrect=2,
            )
            for fold in payload["folds"]:
                for partition_name in ("train", "validation"):
                    partition = fold[partition_name]
                    episodes = {
                        episode["episode_id"]: episode for episode in partition["episodes"]
                    }
                    allowed_recordings = set(fold[f"{partition_name}_recording_ids"])
                    for episode in episodes.values():
                        self.assertEqual(
                            episode["end_row_exclusive"], episode["event_row"] + 1
                        )
                        self.assertLessEqual(
                            episode["event_row"] - episode["start_row"], 3
                        )
                        self.assertIn(episode["recording_id"], allowed_recordings)
                    for pair in partition["pairs"]:
                        error = episodes[pair["incorrect_episode_id"]]
                        correct = episodes[pair["correct_episode_id"]]
                        self.assertEqual(error["component_index"], correct["component_index"])
                        self.assertIn(error["recording_id"], allowed_recordings)
                        self.assertIn(correct["recording_id"], allowed_recordings)

    def test_grouped_summary_counts_operational_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = {name: 1.0 for name in SUMMARY.METRICS}
            validation = (
                root
                / "runs"
                / "grouped_fold0_s7_dual_expert"
                / "validation.json"
            )
            validation.parent.mkdir(parents=True)
            validation.write_text(
                json.dumps({"metrics": metrics}), encoding="utf-8"
            )
            failed = root / "runs" / "grouped_fold1_s7_event" / "history.jsonl"
            failed.parent.mkdir(parents=True)
            failed.write_text("{}\n", encoding="utf-8")

            report = SUMMARY.summarize(
                root, run_prefix_base="grouped", seed=7, folds=2
            )

            self.assertEqual(report["operational_checkpoint_folds"], 1)
            self.assertEqual(report["operational_pass_rate"], 0.5)
            self.assertEqual(
                report["rows"][1]["status"], "no_operational_checkpoint"
            )
            self.assertTrue(report["test_remained_sealed"])

    @staticmethod
    def _write_record(path: Path, *, incorrect: bool) -> None:
        length = 8
        outcomes = np.full((length, 2), -100, dtype=np.int64)
        outcomes[2, 0] = 0
        outcomes[4, 1] = 0
        if incorrect:
            outcomes[6, 0] = 1
        np.savez_compressed(
            path,
            component_outcome=outcomes,
            step=np.asarray([0, 0, 1, 1, 2, 2, 1, 1], dtype=np.int64),
            timestamps=np.arange(length, dtype=np.float32) / 2,
            prediction_frame_indices=np.arange(length, dtype=np.int64) * 3,
        )

    @staticmethod
    def _record(operator: str, split: str, path: Path) -> StateGraphCacheRecord:
        return StateGraphCacheRecord(
            recording_id=f"{operator}_assy_0_1",
            split=split,
            path=path,
            num_steps=3,
            motion_dim=4,
            appearance_dim=4,
            sensor_dim=0,
            num_components=2,
            num_frames=8,
        )


if __name__ == "__main__":
    unittest.main()
