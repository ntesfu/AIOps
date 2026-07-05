from __future__ import annotations

from typing import Protocol

import numpy as np

from aiops.features import ClipFeature
from aiops.models.mstcn import build_ms_tcn_plus_plus
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


class MSTCNCheckpointHead:
    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes: list[str] = list(checkpoint["classes"])
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.model_config = dict(checkpoint["model_config"])
        self.model = build_ms_tcn_plus_plus(
            input_dim=int(self.model_config["input_dim"]),
            num_classes=len(self.classes),
            hidden_dim=int(self.model_config["hidden_dim"]),
            num_stages=int(self.model_config["num_stages"]),
            num_layers=int(self.model_config["num_layers"]),
            dropout=float(self.model_config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict_proba(self, clips: list[ClipFeature], procedure: Procedure) -> np.ndarray:
        if not clips:
            return np.zeros((0, len(self.classes)), dtype=np.float32)

        feature_matrix = np.stack([clip.vector for clip in clips]).astype(np.float32)
        feature_matrix = (feature_matrix - self.feature_mean) / self.feature_std
        tensor = self.torch.from_numpy(feature_matrix.T[None, :, :]).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)[-1][0].transpose(0, 1)
            probabilities = self.torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probabilities.astype(np.float32)
