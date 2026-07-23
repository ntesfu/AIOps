from pathlib import Path

import cv2
import numpy as np
import pytest

from aiops.features.assembly101_video import iter_causal_clips


def _video(path: Path, fps: float, frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    for index in range(frames):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()


def test_60fps_source_is_sampled_on_30fps_clock(tmp_path: Path):
    path = tmp_path / "ego.mp4"
    _video(path, 60.0, 60)
    clips = list(iter_causal_clips(path, clip_frames=4, stride_frames=8))
    assert [clip.end_frame for clip in clips] == [0, 8, 16, 24]
    assert all(clip.frames.shape == (4, 24, 32, 3) for clip in clips)


def test_one_fps_mirror_is_rejected(tmp_path: Path):
    path = tmp_path / "mirror.mp4"
    _video(path, 1.0, 2)
    with pytest.raises(ValueError, match="genuine 30-fps"):
        list(iter_causal_clips(path))
