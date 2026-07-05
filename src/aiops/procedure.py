from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcedureStep:
    id: str
    label: str


@dataclass(frozen=True)
class Procedure:
    name: str
    steps: tuple[ProcedureStep, ...]
    min_confidence: float = 0.5

    @property
    def step_ids(self) -> list[str]:
        return [step.id for step in self.steps]

    @property
    def labels_by_id(self) -> dict[str, str]:
        return {step.id: step.label for step in self.steps}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Procedure":
        steps = tuple(ProcedureStep(id=item["id"], label=item["label"]) for item in payload["steps"])
        return cls(
            name=payload.get("name", "procedure"),
            steps=steps,
            min_confidence=float(payload.get("min_confidence", 0.5)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Procedure":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

