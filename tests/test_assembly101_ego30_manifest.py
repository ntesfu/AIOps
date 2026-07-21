import json
from pathlib import Path

from scripts.prepare_assembly101_ego30_subset import E1_CAMERAS, build_manifest
from aiops.features.assembly101_video_cache import merge_shard_indexes


def test_committed_ego30_manifest_is_actor_disjoint_and_one_view(tmp_path: Path):
    output = tmp_path / "manifest.json"
    payload = build_manifest(
        Path("data/raw/assembly101/subset_manifest.json"),
        Path("data/raw/assembly101/official_hf/annotations"),
        output,
    )
    assert json.loads(output.read_text()) == payload
    records = payload["records"]
    actors = {
        split: {row["actor_id"] for row in records if row["split"] == split}
        for split in ("train", "val", "test")
    }
    assert actors["train"].isdisjoint(actors["val"] | actors["test"])
    assert actors["val"].isdisjoint(actors["test"])
    assert all(row["camera_file"] in E1_CAMERAS for row in records)
    assert len({row["recording_id"] for row in records}) == len(records)
    assert payload["sampling"]["annotation_and_decode_fps"] == 30


def test_cache_shards_merge_only_when_manifest_is_complete(tmp_path: Path):
    metadata = {
        "schema_version": 1,
        "dataset": "Assembly101",
        "manifest_selection": "test",
        "label_contract": "test",
        "event_outcomes": ["correct", "incorrect", "remove"],
        "completion_components": ["wheel"],
        "state_components": ["wheel"],
        "action_ids": ["background", "attach wheel"],
        "fps": 30,
        "stride_frames": 8,
        "clip_frames": 32,
        "camera": "e1",
        "feature_backends": {"motion": "test"},
    }
    records = []
    for index, split in enumerate(("train", "test")):
        path = tmp_path / split / f"r{index}.npz"
        path.parent.mkdir()
        path.write_bytes(b"cache")
        row = {"recording_id": f"r{index}", "split": split, "path": f"{split}/r{index}.npz"}
        records.append(row)
        shard_metadata = {**metadata, "peak_feature_vram_gib": 10 + index, "shard": {"index": index}}
        (tmp_path / f"index.shard-0{index}-of-02.json").write_text(
            json.dumps({"metadata": shard_metadata, "records": [row]})
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": records}))
    result = merge_shard_indexes(tmp_path, manifest)
    assert result["recordings"] == 2
    assert result["peak_vram_gib"] == 11
    assert json.loads((tmp_path / "index.json").read_text())["metadata"][
        "extraction_shards"
    ] == 2
