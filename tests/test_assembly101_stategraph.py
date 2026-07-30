from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from aiops.data.assembly101 import (
    CoarseActionSegment,
    dense_assembly_targets,
    parse_recording_identity,
    read_coarse_action_segments,
    read_mistake_segments,
)
from aiops.features.assembly101_cache import _archive_frames, _sample_positions
from aiops.training.train_stategraph_psr import _infer_event_state_mapping


def test_reads_headerless_official_mistake_csv_and_repairs_known_correction_typo(tmp_path: Path):
    path = tmp_path / "nusar-2021_action_both_9033-c02a_9033_user_id_2021.csv"
    path.write_text(
        "100,130,attach,cabin,interior,mistake,wrong order\n"
        "130,160,interior,cabin,interior,correction,\n",
        encoding="utf-8",
    )

    segments = read_mistake_segments(path)

    assert segments[0].action_id == "attach:cabin"
    assert segments[0].outcome_index == 1
    assert segments[1].verb == "detach"
    assert segments[1].outcome_index == 2


def test_dense_targets_are_causal_and_preserve_fault_then_recovery(tmp_path: Path):
    path = tmp_path / "nusar-2021_action_both_9033-c02a_9033_user_id_2021.csv"
    path.write_text(
        "30,90,attach,cabin,interior,mistake,wrong order\n"
        "90,150,detach,cabin,interior,correction,\n",
        encoding="utf-8",
    )
    segments = read_mistake_segments(path)
    targets = dense_assembly_targets(
        np.asarray([30, 60, 90, 120, 150]),
        segments,
        {"__background__": 0, "attach:cabin": 1, "detach:cabin": 2},
        {"cabin": 0},
        background_index=0,
    )

    assert targets["step"].tolist() == [1, 1, 2, 2, 0]
    assert targets["component_outcome"][:, 0].tolist() == [-100, 1, -100, 2, -100]
    assert targets["state"][:, 0].tolist() == [1, 0, 0, 1, 1]


def test_components_absent_from_recording_are_not_supervised_as_pending(tmp_path: Path):
    path = tmp_path / "nusar-2021_action_both_9033-c02a_9033_user_id_2021.csv"
    path.write_text("30,60,attach,wheel,chassis,mistake,\n", encoding="utf-8")
    targets = dense_assembly_targets(
        np.asarray([0, 30, 60]),
        read_mistake_segments(path),
        {"__background__": 0, "attach:wheel": 1},
        {"wheel": 0, "bucket": 1},
        background_index=0,
    )
    assert targets["state_mask"][:, 0].all()
    assert not targets["state_mask"][:, 1].any()


def test_complete_coarse_actions_replace_sparse_mistake_action_labels(tmp_path: Path):
    mistake = tmp_path / "nusar-2021_action_both_9033-c02a_9033_user_id_2021.csv"
    mistake.write_text("60,90,attach,wheel,chassis,mistake,\n", encoding="utf-8")
    coarse = tmp_path / "assembly.txt"
    coarse.write_text("000000000\t000000060\tinspect toy\t\n60\t120\tattach wheel\t\n")
    action_segments = read_coarse_action_segments([coarse])
    assert action_segments[0] == CoarseActionSegment(0, 60, "inspect toy")
    targets = dense_assembly_targets(
        np.asarray([0, 30, 60, 90, 120]),
        read_mistake_segments(mistake),
        {"__background__": 0, "inspect toy": 1, "attach wheel": 2},
        {"wheel": 0},
        background_index=0,
        action_segments=action_segments,
    )
    assert targets["step"].tolist() == [1, 1, 2, 2, 0]
    assert targets["component_outcome"][:, 0].tolist() == [-100, -100, 1, -100, -100]


def test_public_mirror_frame_clock_is_converted_to_official_30fps(tmp_path: Path):
    archive = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for index in (3, 1, 2):
            handle.writestr(f"C10095_rgb/{index}.png", b"not decoded in this test")

    names, frames = _archive_frames(archive, annotation_fps=30.0, mirror_fps=1.0)

    assert names == ["C10095_rgb/1.png", "C10095_rgb/2.png", "C10095_rgb/3.png"]
    assert frames.tolist() == [30, 60, 90]
    assert _sample_positions(frames, 60).tolist() == [0, 2]


def test_recording_identity_is_parsed_without_leaking_actor_across_splits():
    actor, toy = parse_recording_identity(
        "nusar-2021_action_both_9086-c14a_9086_user_id_2021-02-16_153910"
    )
    assert actor == "9086"
    assert toy == "c14a"


def test_declared_component_names_prevent_correlated_state_mapping_errors():
    mapping = _infer_event_state_mapping(
        [],
        event_components=2,
        state_components=3,
        metadata={
            "completion_components": ["wheel", "cabin"],
            "state_components": ["cabin", "unused", "wheel"],
        },
    )
    assert mapping["indices"] == [2, 0]
    assert mapping["source"] == "metadata_component_names"
