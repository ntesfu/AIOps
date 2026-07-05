from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemporalHeadConfig:
    type: str
    input_dim: int = 14
    hidden_dim: int = 128
    num_stages: int = 4
    num_layers: int = 8
    dropout: float = 0.3


@dataclass(frozen=True)
class DecoderConfig:
    min_segment_s: float = 0.5
    smoothing_window: int = 3


@dataclass(frozen=True)
class AIOpsArchitectureConfig:
    name: str
    clip_stride_s: float = 0.5
    clip_duration_s: float = 1.0
    feature_backend: str = "statistical_clip"
    temporal_head: TemporalHeadConfig = TemporalHeadConfig(type="ms_tcn_plus_plus")
    decoder: DecoderConfig = DecoderConfig()
    validator_enabled: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AIOpsArchitectureConfig":
        return cls(
            name=payload.get("name", "aiops_temporal"),
            clip_stride_s=float(payload.get("clip_stride_s", 0.5)),
            clip_duration_s=float(payload.get("clip_duration_s", 1.0)),
            feature_backend=payload.get("feature_backend", "statistical_clip"),
            temporal_head=TemporalHeadConfig(**payload.get("temporal_head", {"type": "ms_tcn_plus_plus"})),
            decoder=DecoderConfig(**payload.get("decoder", {})),
            validator_enabled=bool(payload.get("validator", {}).get("enabled", True)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "AIOpsArchitectureConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

