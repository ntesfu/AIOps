from pathlib import Path

import cv2
import numpy as np
import pytest

from aiops.features.assembly101_video import iter_causal_clips
from aiops.features.assembly101_video_cache import _manifest_video_path


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


def test_external_video_root_uses_recording_and_selected_camera():
    row = {
        "recording_id": "recording-a",
        "camera_file": "recording-a_e1_rgb.mp4",
        "video_relative_path": "legacy/missing.mp4",
    }
    path = _manifest_video_path(row, Path("/annotations"), Path("/videos"))
    assert path == Path("/videos/recording-a/recording-a_e1_rgb.mp4")
