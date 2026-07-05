from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClipFeature:
    start_s: float
    end_s: float
    vector: np.ndarray


class StatisticalClipFeatureExtractor:
    """CPU feature extractor used until VideoMAE/EgoVLP features are wired in.

    It preserves the same sequence shape that a frozen video backbone would
    produce: one feature vector per temporal clip.
    """

    output_dim = 14

    def __init__(self, clip_duration_s: float = 1.0, stride_s: float = 0.5, frame_size: int = 64) -> None:
        self.clip_duration_s = clip_duration_s
        self.stride_s = stride_s
        self.frame_size = frame_size

    def extract(self, video_path: str | Path) -> list[ClipFeature]:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return []

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            capture.release()
            return []

        duration_s = frame_count / fps
        starts = np.arange(0.0, max(duration_s, self.stride_s), self.stride_s)
        clips: list[ClipFeature] = []
        for start_s in starts:
            end_s = min(float(start_s + self.clip_duration_s), duration_s)
            if end_s <= start_s:
                continue
            vector = self._extract_clip(capture, fps=fps, start_s=float(start_s), end_s=end_s)
            clips.append(ClipFeature(start_s=float(start_s), end_s=end_s, vector=vector))
        capture.release()
        return clips

    def _extract_clip(self, capture: object, fps: float, start_s: float, end_s: float) -> np.ndarray:
        import cv2

        start_frame = int(round(start_s * fps))
        end_frame = max(start_frame + 1, int(round(end_s * fps)))
        sample_frames = np.linspace(start_frame, end_frame - 1, num=min(8, end_frame - start_frame), dtype=int)

        frames: list[np.ndarray] = []
        last_gray: np.ndarray | None = None
        motion_scores: list[float] = []
        for frame_index in sample_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            resized = cv2.resize(frame, (self.frame_size, self.frame_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if last_gray is not None:
                motion_scores.append(float(np.mean(np.abs(gray - last_gray))))
            last_gray = gray
            frames.append(rgb)

        if not frames:
            return np.zeros(self.output_dim, dtype=np.float32)

        stack = np.stack(frames)
        channel_mean = stack.mean(axis=(0, 1, 2))
        channel_std = stack.std(axis=(0, 1, 2))
        gray_stack = stack.mean(axis=3)
        brightness = gray_stack.mean(axis=(1, 2))
        return np.array(
            [
                *channel_mean.tolist(),
                *channel_std.tolist(),
                float(brightness.mean()),
                float(brightness.std()),
                float(brightness.min()),
                float(brightness.max()),
                float(np.mean(motion_scores)) if motion_scores else 0.0,
                float(np.std(motion_scores)) if motion_scores else 0.0,
                float(len(frames) / len(sample_frames)),
                fps,
            ],
            dtype=np.float32,
        )

