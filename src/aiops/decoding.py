from __future__ import annotations

import numpy as np

from aiops.features import ClipFeature
from aiops.procedure import Procedure
from aiops.types import StepPrediction


def smooth_probabilities(probabilities: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(probabilities) == 0:
        return probabilities

    radius = window // 2
    smoothed = np.zeros_like(probabilities)
    for index in range(len(probabilities)):
        start = max(0, index - radius)
        end = min(len(probabilities), index + radius + 1)
        smoothed[index] = probabilities[start:end].mean(axis=0)
    return smoothed


def decode_segments(
    clips: list[ClipFeature],
    probabilities: np.ndarray,
    procedure: Procedure,
    min_segment_s: float = 0.5,
    smoothing_window: int = 3,
) -> list[StepPrediction]:
    if not clips or probabilities.size == 0:
        return []

    boundaries = _segment_boundaries(clips)
    probabilities = smooth_probabilities(probabilities, smoothing_window)
    label_indices = np.argmax(probabilities, axis=1)
    labels_by_id = procedure.labels_by_id
    step_ids = procedure.step_ids
    segments: list[StepPrediction] = []

    start_index = 0
    current_label = int(label_indices[0])
    for index in range(1, len(label_indices) + 1):
        is_boundary = index == len(label_indices) or int(label_indices[index]) != current_label
        if not is_boundary:
            continue

        start_s = boundaries[start_index]
        end_s = boundaries[index]
        if end_s - start_s >= min_segment_s:
            step_id = step_ids[current_label]
            confidence = float(probabilities[start_index:index, current_label].mean())
            segments.append(
                StepPrediction(
                    step_id=step_id,
                    label=labels_by_id.get(step_id, step_id),
                    start_s=round(start_s, 3),
                    end_s=round(end_s, 3),
                    confidence=round(confidence, 3),
                )
            )

        if index < len(label_indices):
            start_index = index
            current_label = int(label_indices[index])

    return segments


def _segment_boundaries(clips: list[ClipFeature]) -> list[float]:
    if not clips:
        return []

    centers = [(clip.start_s + clip.end_s) / 2.0 for clip in clips]
    boundaries = [clips[0].start_s]
    for left, right in zip(centers, centers[1:]):
        boundaries.append((left + right) / 2.0)
    boundaries.append(clips[-1].end_s)
    return boundaries
