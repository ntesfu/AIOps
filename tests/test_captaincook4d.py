import json
from pathlib import Path

from aiops.data.captaincook4d import (
    ERROR_CATEGORIES,
    filter_segments,
    load_recording_split,
    load_segments,
    video_path,
)


def _write_fixture(root: Path) -> None:
    json_dir = root / "annotation_json"
    json_dir.mkdir(parents=True)
    complete = {
        "1_1": {
            "recording_id": "1_1",
            "activity_id": 1,
            "activity_name": "Recipe A",
            "person_id": 3,
            "environment": 8,
            "steps": [
                {"step_id": 3, "start_time": 1.0, "end_time": 5.0,
                 "description": "coat", "has_errors": False},
                {"step_id": 4, "start_time": -1.0, "end_time": -1.0,
                 "description": "microwave", "has_errors": True},
            ],
        },
        "2_1": {
            "recording_id": "2_1",
            "activity_id": 2,
            "activity_name": "Recipe B",
            "person_id": 5,
            "environment": 2,
            "steps": [
                {"step_id": 1, "start_time": 0.0, "end_time": 3.0,
                 "description": "pour", "has_errors": True},
            ],
        },
    }
    errors = [
        {"recording_id": "1_1", "is_error": True, "step_annotations": [
            {"step_id": 3, "start_time": 1.0, "end_time": 5.0, "errors": []},
            {"step_id": 4, "start_time": -1.0, "end_time": -1.0,
             "errors": [{"tag": "Missing Step", "description": "skipped"}]},
        ]},
        {"recording_id": "2_1", "is_error": True, "step_annotations": [
            {"step_id": 1, "start_time": 0.0, "end_time": 3.0,
             "errors": [{"tag": "Order Error", "description": "wrong order"}]},
        ]},
    ]
    (json_dir / "complete_step_annotations.json").write_text(json.dumps(complete))
    (json_dir / "error_annotations.json").write_text(json.dumps(errors))
    splits = root / "data_splits"
    splits.mkdir()
    (splits / "person_data_split_combined.json").write_text(
        json.dumps({"train": ["1_1"], "val": [], "test": ["2_1"]})
    )


def test_load_segments_excludes_missing_and_joins_tags(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    segments = load_segments(tmp_path)
    # Missing-step (start == -1) dropped by default: 2 observable of 3 total.
    assert len(segments) == 2
    by_rec = {(s.recording_id, s.step_index): s for s in segments}
    normal = by_rec[("1_1", 0)]
    assert normal.label == 0 and normal.error_tags == ()
    assert normal.person_id == "3" and normal.duration == 4.0
    err = by_rec[("2_1", 0)]
    assert err.label == 1 and err.error_tags == ("Order Error",)


def test_include_missing_flag(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    segments = load_segments(tmp_path, include_missing=True)
    assert len(segments) == 3
    missing = [s for s in segments if s.is_missing]
    assert len(missing) == 1
    assert missing[0].error_tags == ("Missing Step",)
    assert missing[0].label == 1


def test_load_recording_split_and_filter(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    split = load_recording_split(tmp_path, "person")
    assert split["train"] == {"1_1"} and split["test"] == {"2_1"}
    segments = load_segments(tmp_path)
    train = filter_segments(segments, split["train"])
    assert {s.recording_id for s in train} == {"1_1"}


def test_error_categories_and_video_path() -> None:
    assert len(ERROR_CATEGORIES) == 8
    assert ERROR_CATEGORIES[2] == "Order Error"
    p = video_path("/data", "1_7")
    assert p.name == "1_7_360p.mp4"
    assert p.parent.name == "resolution_360p"
