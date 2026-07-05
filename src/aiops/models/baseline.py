from __future__ import annotations

from aiops.procedure import Procedure
from aiops.types import StepPrediction


class UniformProcedureBaseline:
    """Deterministic stand-in until trained temporal models are available."""

    def __init__(self, procedure: Procedure, default_duration_s: float = 20.0) -> None:
        self.procedure = procedure
        self.default_duration_s = default_duration_s

    def predict(self, duration_s: float | None = None) -> list[StepPrediction]:
        duration = duration_s or self.default_duration_s
        step_count = max(len(self.procedure.steps), 1)
        segment_s = duration / step_count

        predictions: list[StepPrediction] = []
        for index, step in enumerate(self.procedure.steps):
            start_s = round(index * segment_s, 3)
            end_s = round(duration if index == step_count - 1 else (index + 1) * segment_s, 3)
            confidence = round(max(0.6, 0.9 - index * 0.04), 3)
            predictions.append(
                StepPrediction(
                    step_id=step.id,
                    label=step.label,
                    start_s=start_s,
                    end_s=end_s,
                    confidence=confidence,
                )
            )
        return predictions

