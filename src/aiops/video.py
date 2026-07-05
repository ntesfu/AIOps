from __future__ import annotations

from pathlib import Path


def probe_video_duration_s(path: str | Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if not fps or fps <= 0:
        return None
    return float(frame_count / fps)

