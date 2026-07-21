from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoClip:
    """A causal clip decoded on the Assembly101 30-fps annotation clock."""

    end_frame: int
    frames: np.ndarray  # [T, H, W, 3], uint8 RGB


def iter_causal_clips(
    path: str | Path,
    *,
    target_fps: float = 30.0,
    clip_frames: int = 32,
    stride_frames: int = 8,
) -> Iterator[VideoClip]:
    """Decode a video at ``target_fps`` and yield past-only temporal clips.

    Assembly101 raw ego videos are normally 60 fps while all released labels
    use a 30-fps frame clock. Decoding by timestamp (rather than assuming a
    fixed source rate) keeps the two clocks aligned and also handles unusual
    files safely. The first clip is left-padded with the first decoded frame.
    """
    import cv2

    if target_fps <= 0 or clip_frames <= 0 or stride_frames <= 0:
        raise ValueError("target_fps, clip_frames, and stride_frames must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps + 1e-6 < target_fps:
        capture.release()
        raise ValueError(
            f"Source is only {source_fps:.3f} fps; genuine {target_fps:g}-fps input is impossible"
        )

    history: deque[np.ndarray] = deque(maxlen=clip_frames)
    source_index = 0
    annotation_index = 0
    next_source_position = 0.0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            # Select the source frame nearest the next point on the 30-fps clock.
            if source_index + 0.5 >= next_source_position:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if not history:
                    history.extend([rgb] * clip_frames)
                else:
                    history.append(rgb)
                if annotation_index % stride_frames == 0:
                    yield VideoClip(annotation_index, np.stack(tuple(history)))
                annotation_index += 1
                next_source_position = annotation_index * source_fps / target_fps
            source_index += 1
    finally:
        capture.release()


class Swin3DFeatureExtractor:
    """Frozen Kinetics-400 temporal features for causal 30-fps clips."""

    output_dim = 768

    def __init__(self, device: str | None = None, precision: str = "bf16") -> None:
        import torch
        import torch.nn as nn
        from torchvision.models.video import Swin3D_S_Weights, swin3d_s

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.model = swin3d_s(weights=Swin3D_S_Weights.KINETICS400_V1)
        self.model.head = nn.Identity()
        self.model.eval().to(self.device)

    def extract(self, clips: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        tensors = []
        for clip in clips:
            # Swin3D weights use 224x224, Kinetics mean/std, and [B,C,T,H,W].
            x = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
            x = x.permute(1, 0, 2, 3)
            mean = x.new_tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
            std = x.new_tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)
            tensors.append((x - mean) / std)
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        enabled = self.device.type == "cuda" and self.precision in {"bf16", "fp16"}
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=dtype, enabled=enabled
        ):
            output = self.model(batch)
        if self.device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
            if peak_gib >= 23.0:
                raise RuntimeError(f"Swin3D extraction exceeded the 23-GiB limit: {peak_gib:.2f} GiB")
        return output.float().cpu().numpy().astype(np.float32)
