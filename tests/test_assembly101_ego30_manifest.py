import json
from pathlib import Path

from scripts.prepare_assembly101_ego30_subset import E1_CAMERAS, build_manifest


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
