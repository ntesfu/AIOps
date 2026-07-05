from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiops.architecture import AIOpsArchitectureConfig
from aiops.decoding import decode_segments
from aiops.features import StatisticalClipFeatureExtractor
from aiops.models import UniformProcedureBaseline
from aiops.models.temporal import TemporalPriorHead
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


def run_temporal_inference(
    video_path: str | Path,
    procedure: Procedure,
    architecture: AIOpsArchitectureConfig,
) -> dict[str, Any]:
    duration_s = probe_video_duration_s(video_path)
    extractor = StatisticalClipFeatureExtractor(
        clip_duration_s=architecture.clip_duration_s,
        stride_s=architecture.clip_stride_s,
    )
    clips = extractor.extract(video_path)
    probabilities = TemporalPriorHead().predict_proba(clips, procedure)
    predictions = decode_segments(
        clips=clips,
        probabilities=probabilities,
        procedure=procedure,
        min_segment_s=architecture.decoder.min_segment_s,
        smoothing_window=architecture.decoder.smoothing_window,
    )
    validation_events = ProcedureValidator(procedure).validate(predictions) if architecture.validator_enabled else []

    return {
        "video_path": str(video_path),
        "procedure": procedure.name,
        "architecture": architecture.name,
        "feature_backend": architecture.feature_backend,
        "temporal_head": "temporal_prior_placeholder",
        "duration_s": duration_s,
        "num_clips": len(clips),
        "clip_stride_s": architecture.clip_stride_s,
        "clip_duration_s": architecture.clip_duration_s,
        "predictions": [prediction.to_dict() for prediction in predictions],
        "validation_events": [event.to_dict() for event in validation_events],
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
