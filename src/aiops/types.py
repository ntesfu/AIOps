from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StepPrediction:
    step_id: str
    label: str
    start_s: float
    end_s: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationEvent:
    kind: str
    message: str
    step_id: str | None = None
    time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

