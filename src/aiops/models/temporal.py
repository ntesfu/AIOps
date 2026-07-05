from __future__ import annotations

from typing import Protocol

import numpy as np

from aiops.features import ClipFeature
from aiops.procedure import Procedure


class TemporalProbabilityHead(Protocol):
    def predict_proba(self, clips: list[ClipFeature], procedure: Procedure) -> np.ndarray:
        """Return class probabilities with shape [num_clips, num_steps]."""


class TemporalPriorHead:
    """Deterministic temporal head used before a trained MS-TCN checkpoint exists.

    This is deliberately shaped like the learned head output so the rest of the
    pipeline can be tested with real video clip features.
    """

    def __init__(self, temperature: float = 0.12) -> None:
        self.temperature = temperature

    def predict_proba(self, clips: list[ClipFeature], procedure: Procedure) -> np.ndarray:
        if not clips:
            return np.zeros((0, len(procedure.steps)), dtype=np.float32)

        num_steps = max(len(procedure.steps), 1)
        timeline = np.linspace(0.0, 1.0, len(clips), dtype=np.float32)
        centers = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
        logits = -np.square(timeline[:, None] - centers[None, :]) / max(self.temperature, 1e-6)

        feature_matrix = np.stack([clip.vector for clip in clips])
        motion = feature_matrix[:, 10] if feature_matrix.shape[1] > 10 else np.zeros(len(clips), dtype=np.float32)
        if motion.std() > 1e-6:
            motion = (motion - motion.mean()) / (motion.std() + 1e-6)
            logits += 0.03 * motion[:, None]

        return _softmax(logits.astype(np.float32))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

