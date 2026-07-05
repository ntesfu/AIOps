from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiops.models import UniformProcedureBaseline
from aiops.procedure import Procedure
from aiops.validation import ProcedureValidator
from aiops.video import probe_video_duration_s


def run_baseline_inference(video_path: str | Path, procedure: Procedure) -> dict[str, Any]:
    duration_s = probe_video_duration_s(video_path)
    predictions = UniformProcedureBaseline(procedure).predict(duration_s=duration_s)
    validation_events = ProcedureValidator(procedure).validate(predictions)

    return {
        "video_path": str(video_path),
        "procedure": procedure.name,
        "duration_s": duration_s,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "validation_events": [event.to_dict() for event in validation_events],
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

