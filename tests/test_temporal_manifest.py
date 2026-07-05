from pathlib import Path

from aiops.data.temporal_manifest import TemporalSegment, TemporalVideo, read_temporal_manifest, write_temporal_manifest
from aiops.training.train_mstcn import label_for_time


def test_temporal_manifest_round_trip(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        TemporalVideo(
            video_path=Path("video.mp4"),
            dataset="demo",
            procedure="assembly",
            segments=(
                TemporalSegment("pick", "Pick", 0.0, 1.0),
                TemporalSegment("place", "Place", 1.0, 2.0),
            ),
        )
    ]

    write_temporal_manifest(rows, manifest)
    loaded = read_temporal_manifest(manifest)

    assert loaded[0].procedure == "assembly"
    assert loaded[0].segments[1].step_id == "place"


def test_label_for_time_uses_segment_boundaries() -> None:
    video = TemporalVideo(
        video_path=Path("video.mp4"),
        segments=(
            TemporalSegment("pick", "Pick", 0.0, 1.0),
            TemporalSegment("place", "Place", 1.0, 2.0),
        ),
    )

    assert label_for_time(video, 0.5, {"pick": 0, "place": 1}) == 0
    assert label_for_time(video, 1.5, {"pick": 0, "place": 1}) == 1
