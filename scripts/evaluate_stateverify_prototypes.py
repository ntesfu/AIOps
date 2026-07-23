#!/usr/bin/env python3
"""Evaluate step/component-conditioned normal prototypes on IndustReal."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.stategraph_cache import read_cache_index
from aiops.evaluation.stateverify_prototypes import (
    binary_average_precision,
    calibrate_threshold,
    fit_prototype_bank,
    score_with_bank,
    threshold_metrics,
)
from aiops.models.stateverify_effect import (
    StateEffectObserverConfig,
    build_state_effect_observer,
)
from aiops.training.train_stateverify_observer import (
    _observer_inputs,
    _validate_contract,
)


def _record_outputs(record, model, config, device):
    import torch

    with np.load(record.path, allow_pickle=False) as arrays:
        copied = {name: arrays[name].copy() for name in arrays.files}
    length = len(copied["step"])
    batch = {
        name: torch.from_numpy(copied[name]).unsqueeze(0).to(device)
        for name in ("motion", "appearance", "motion_aux", "sensor", "modality_mask")
    }
    batch["valid_mask"] = torch.ones(1, length, dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(**_observer_inputs(batch, config))
    return copied, {
        "component_features": output["component_features"][0].float().cpu().numpy(),
        "effect_probabilities": output["effect_probabilities"][0]
        .float()
        .cpu()
        .numpy(),
    }


def _duration_minutes(arrays: dict[str, np.ndarray]) -> float:
    timestamps = arrays.get("timestamps")
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    return max(float(timestamps[-1] - timestamps[0]) / 60.0, 0.0)


def _collect(
    records,
    model,
    config,
    event_state_indices,
    device,
    *,
    collect_normals: bool,
):
    normal_examples: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    duration = 0.0
    for ordinal, record in enumerate(records, start=1):
        arrays, output = _record_outputs(record, model, config, device)
        duration += _duration_minutes(arrays)
        features = output["component_features"]
        if collect_normals:
            state = arrays["state"]
            state_mask = arrays["state_mask"].astype(bool)
            steps = arrays["step"]
            for component in range(state.shape[1]):
                rows = np.where(state_mask[:, component] & (state[:, component] == 2))[0]
                for row in rows:
                    step = int(steps[row])
                    if step >= 0:
                        normal_examples[(component, step)].append(
                            features[row, component]
                        )
        outcomes = arrays["component_outcome"]
        rows, event_components = np.where((outcomes == 0) | (outcomes == 1))
        for row, event_component in zip(rows, event_components):
            state_component = int(event_state_indices[event_component])
            events.append(
                {
                    "recording_id": record.recording_id,
                    "row": int(row),
                    "event_component": int(event_component),
                    "state_component": state_component,
                    "step": int(arrays["step"][row]),
                    "label": int(outcomes[row, event_component] == 1),
                    "feature": features[row, state_component],
                    "supervised_score": float(
                        output["effect_probabilities"][row, state_component, 2]
                    ),
                }
            )
        print(
            f"[{ordinal}/{len(records)}] {record.split}/{record.recording_id}",
            flush=True,
        )
    return normal_examples, events, duration


def _score_events(bank, events):
    scores = []
    sources: dict[str, int] = defaultdict(int)
    retained = []
    for event in events:
        score, source = score_with_bank(
            bank,
            event["state_component"],
            event["step"],
            event["feature"],
        )
        sources[source] += 1
        if np.isfinite(score):
            scores.append(score)
            retained.append(event)
    return np.asarray(scores), retained, dict(sources)


def _per_component(scores, events):
    report = {}
    components = sorted({event["event_component"] for event in events})
    for component in components:
        indices = [
            index
            for index, event in enumerate(events)
            if event["event_component"] == component
        ]
        labels = [events[index]["label"] for index in indices]
        report[str(component)] = {
            "samples": len(indices),
            "incorrect": int(sum(labels)),
            "average_precision": binary_average_precision(
                [scores[index] for index in indices], labels
            ),
        }
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    metadata, records = read_cache_index(args.cache_index)
    _validate_contract(metadata, records)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = StateEffectObserverConfig(**checkpoint["model_config"])
    model = build_state_effect_observer(config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    event_state_indices = tuple(checkpoint["event_state_mapping"]["indices"])
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    normal_examples, train_events, train_duration = _collect(
        train_records,
        model,
        config,
        event_state_indices,
        device,
        collect_normals=True,
    )
    _, val_events, val_duration = _collect(
        val_records,
        model,
        config,
        event_state_indices,
        device,
        collect_normals=False,
    )
    result: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache_index": str(Path(args.cache_index).resolve()),
        "train_duration_minutes": train_duration,
        "validation_duration_minutes": val_duration,
        "train_events": len(train_events),
        "validation_events": len(val_events),
        "train_incorrect_events": sum(event["label"] for event in train_events),
        "validation_incorrect_events": sum(event["label"] for event in val_events),
        "supervised_effect_incorrect_average_precision": binary_average_precision(
            [event["supervised_score"] for event in val_events],
            [event["label"] for event in val_events],
        ),
        "prototype_candidates": {},
    }
    for prototype_count in args.prototype_counts:
        bank = fit_prototype_bank(
            normal_examples,
            prototypes=prototype_count,
            minimum_step_samples=args.minimum_step_samples,
            maximum_samples=args.maximum_samples_per_group,
            seed=args.seed,
        )
        train_scores, retained_train, train_sources = _score_events(bank, train_events)
        val_scores, retained_val, val_sources = _score_events(bank, val_events)
        train_labels = [event["label"] for event in retained_train]
        val_labels = [event["label"] for event in retained_val]
        calibration = calibrate_threshold(
            train_scores,
            train_labels,
            duration_minutes=train_duration,
            max_false_alerts_per_minute=args.max_false_alerts_per_minute,
        )
        validation_metrics = threshold_metrics(
            val_scores,
            val_labels,
            calibration["threshold"],
            duration_minutes=val_duration,
        )
        key = str(prototype_count)
        result["prototype_candidates"][key] = {
            "prototype_groups": len(bank),
            "train_score_sources": train_sources,
            "validation_score_sources": val_sources,
            "train_average_precision": binary_average_precision(
                train_scores, train_labels
            ),
            "validation_average_precision": binary_average_precision(
                val_scores, val_labels
            ),
            "calibration": calibration,
            "validation_threshold_metrics": validation_metrics,
            "validation_score_mean_correct": float(
                np.mean(
                    [
                        score
                        for score, label in zip(val_scores, val_labels)
                        if label == 0
                    ]
                )
            ),
            "validation_score_mean_incorrect": float(
                np.mean(
                    [
                        score
                        for score, label in zip(val_scores, val_labels)
                        if label == 1
                    ]
                )
            ),
            "per_event_component": _per_component(val_scores, retained_val),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prototype-counts", type=int, nargs="+", default=[1, 3, 5]
    )
    parser.add_argument("--minimum-step-samples", type=int, default=20)
    parser.add_argument("--maximum-samples-per-group", type=int, default=4000)
    parser.add_argument("--max-false-alerts-per-minute", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
