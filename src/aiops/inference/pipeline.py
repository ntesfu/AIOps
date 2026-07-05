from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiops.architecture import AIOpsArchitectureConfig
from aiops.decoding import decode_segments
from aiops.features import StatisticalClipFeatureExtractor
from aiops.models import UniformProcedureBaseline
from aiops.models.temporal import MSTCNCheckpointHead, TemporalPriorHead
from aiops.procedure import Procedure, ProcedureStep
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


def run_temporal_inference(
    video_path: str | Path,
    procedure: Procedure,
    architecture: AIOpsArchitectureConfig,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    duration_s = probe_video_duration_s(video_path)
    extractor = StatisticalClipFeatureExtractor(
        clip_duration_s=architecture.clip_duration_s,
        stride_s=architecture.clip_stride_s,
    )
    clips = extractor.extract(video_path)
    if checkpoint_path:
        head = MSTCNCheckpointHead(str(checkpoint_path))
        active_procedure = Procedure(
            name=Path(checkpoint_path).stem,
            steps=tuple(ProcedureStep(step_id, step_id) for step_id in head.classes),
            min_confidence=procedure.min_confidence,
        )
        temporal_head_name = "ms_tcn_checkpoint"
        should_validate = False
    else:
        head = TemporalPriorHead()
        active_procedure = procedure
        temporal_head_name = "temporal_prior_placeholder"
        should_validate = architecture.validator_enabled

    probabilities = head.predict_proba(clips, active_procedure)
    predictions = decode_segments(
        clips=clips,
        probabilities=probabilities,
        procedure=active_procedure,
        min_segment_s=architecture.decoder.min_segment_s,
        smoothing_window=architecture.decoder.smoothing_window,
    )
    validation_events = ProcedureValidator(active_procedure).validate(predictions) if should_validate else []
    clip_predictions = _clip_predictions(clips, probabilities, active_procedure)

    return {
        "video_path": str(video_path),
        "procedure": active_procedure.name,
        "architecture": architecture.name,
        "feature_backend": architecture.feature_backend,
        "temporal_head": temporal_head_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "duration_s": duration_s,
        "num_clips": len(clips),
        "clip_stride_s": architecture.clip_stride_s,
        "clip_duration_s": architecture.clip_duration_s,
        "clip_predictions": clip_predictions,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "validation_events": [event.to_dict() for event in validation_events],
    }


def _clip_predictions(clips: list[Any], probabilities: Any, procedure: Procedure) -> list[dict[str, Any]]:
    if not clips or getattr(probabilities, "size", 0) == 0:
        return []
    step_ids = procedure.step_ids
    labels_by_id = procedure.labels_by_id
    rows: list[dict[str, Any]] = []
    for clip, probability in zip(clips, probabilities):
        index = int(probability.argmax())
        step_id = step_ids[index]
        rows.append(
            {
                "start_s": round(float(clip.start_s), 3),
                "end_s": round(float(clip.end_s), 3),
                "step_id": step_id,
                "label": labels_by_id.get(step_id, step_id),
                "confidence": round(float(probability[index]), 3),
            }
        )
    return rows


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
