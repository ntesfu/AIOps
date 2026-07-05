from pathlib import Path

import numpy as np

from aiops.architecture import AIOpsArchitectureConfig
from aiops.decoding import decode_segments
from aiops.features import ClipFeature
from aiops.inference.pipeline import run_temporal_inference
from aiops.procedure import Procedure, ProcedureStep


def make_procedure() -> Procedure:
    return Procedure(
        name="demo",
        steps=(
            ProcedureStep("pick", "Pick"),
            ProcedureStep("attach", "Attach"),
            ProcedureStep("inspect", "Inspect"),
        ),
    )


def test_decode_segments_merges_contiguous_clip_predictions() -> None:
    clips = [
        ClipFeature(0.0, 0.5, np.zeros(14, dtype=np.float32)),
        ClipFeature(0.5, 1.0, np.zeros(14, dtype=np.float32)),
        ClipFeature(1.0, 1.5, np.zeros(14, dtype=np.float32)),
    ]
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    segments = decode_segments(clips, probabilities, make_procedure(), smoothing_window=1)

    assert [segment.step_id for segment in segments] == ["pick", "attach"]
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 1.0


def test_temporal_pipeline_runs_on_tiny_video(tmp_path: Path) -> None:
    import cv2

    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for index in range(20):
        frame = np.full((64, 64, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    architecture = AIOpsArchitectureConfig(
        name="test_temporal",
        clip_stride_s=0.5,
        clip_duration_s=1.0,
    )

    payload = run_temporal_inference(video_path, make_procedure(), architecture)

    assert payload["num_clips"] > 0
    assert payload["architecture"] == "test_temporal"
    assert payload["predictions"]
