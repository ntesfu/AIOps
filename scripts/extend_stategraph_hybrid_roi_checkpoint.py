#!/usr/bin/env python3
"""Extend a flat-ROI StateGraph checkpoint with zero-gated structured ROI modules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aiops.models.stategraph_psr import StateGraphPSRConfig, build_stategraph_psr


_ALLOWED_NEW_PREFIXES = (
    "roi_",
    "component_roi_attention.",
    "component_roi_attention_norm.",
    "structured_roi_gate_raw",
)


def extend_checkpoint(
    source_path: Path,
    output_path: Path,
    *,
    roi_token_count: int = 4,
    roi_embedding_dim: int = 768,
    roi_global_dim: int = 3,
    roi_change_lag: int = 4,
) -> dict[str, Any]:
    import torch

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_config = StateGraphPSRConfig(**source["model_config"])
    if not source_config.component_evidence or not source_config.event_only_motion_aux:
        raise ValueError(
            "Hybrid extension requires a component-evidence checkpoint with event-only motion_aux."
        )
    target_values = source_config.to_dict()
    target_values.update(
        {
            "structured_roi_tokens": True,
            "hybrid_roi_residual": True,
            "roi_token_count": roi_token_count,
            "roi_embedding_dim": roi_embedding_dim,
            "roi_global_dim": roi_global_dim,
            "roi_change_lag": roi_change_lag,
        }
    )
    target_config = StateGraphPSRConfig(**target_values)
    model = build_stategraph_psr(target_config, source.get("transition_matrix"))
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(_ALLOWED_NEW_PREFIXES)
    ]
    if unexpected or disallowed_missing:
        raise ValueError(
            "Checkpoint extension changed non-ROI parameters: "
            f"unexpected={unexpected}, missing={disallowed_missing}"
        )
    if not incompatible.missing_keys:
        raise ValueError("Source checkpoint already contains the hybrid ROI extension.")
    if not torch.equal(
        torch.tanh(model.structured_roi_gate_raw.detach()),
        torch.zeros_like(model.structured_roi_gate_raw),
    ):
        raise RuntimeError("Hybrid residual gate did not initialize to exactly zero.")

    payload = dict(source)
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "rng_state"):
        payload.pop(key, None)
    payload["model_config"] = target_config.to_dict()
    payload["model_state"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload["architecture_extension"] = {
        "type": "hybrid_structured_roi_v1",
        "source_checkpoint": str(source_path.resolve()),
        "copied_parameter_tensors": len(source["model_state"]),
        "new_parameter_tensors": list(incompatible.missing_keys),
        "gate_initialization": 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload["architecture_extension"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--roi-token-count", type=int, default=4)
    parser.add_argument("--roi-embedding-dim", type=int, default=768)
    parser.add_argument("--roi-global-dim", type=int, default=3)
    parser.add_argument("--roi-change-lag", type=int, default=4)
    args = parser.parse_args()
    report = extend_checkpoint(
        args.source_checkpoint,
        args.output_checkpoint,
        roi_token_count=args.roi_token_count,
        roi_embedding_dim=args.roi_embedding_dim,
        roi_global_dim=args.roi_global_dim,
        roi_change_lag=args.roi_change_lag,
    )
    print(report)


if __name__ == "__main__":
    main()
