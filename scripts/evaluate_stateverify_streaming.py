#!/usr/bin/env python3
"""Calibrate and evaluate StateVerify on complete causal IndustReal timelines."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.stategraph_cache import read_cache_index
from aiops.evaluation.stateverify_prototypes import (
    fit_prototype_bank,
    save_prototype_bank,
)
from aiops.evaluation.stateverify_streaming import (
    StreamingConfig,
    component_anomaly_scores,
    match_streaming_alerts,
    predicted_step_anomaly_scores,
    run_streaming_tracker,
)
from aiops.models.stategraph_psr import StateGraphPSRConfig, build_stategraph_psr
from aiops.models.stateverify_effect import (
    StateEffectObserverConfig,
    build_state_effect_observer,
)
from aiops.training.train_stateverify_observer import (
    _observer_inputs,
    _validate_contract,
)


def _infer_record(
    record,
    model,
    config,
    device,
    action_model=None,
    action_sensor_dim=None,
    hand_dim=0,
    pose_dim=0,
) -> dict[str, Any]:
    import torch

    with np.load(record.path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    length = len(arrays["step"])
    batch = {
        name: torch.from_numpy(arrays[name]).unsqueeze(0).to(device)
        for name in ("motion", "appearance", "motion_aux", "sensor", "modality_mask")
    }
    batch["valid_mask"] = torch.ones(1, length, dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model(**_observer_inputs(batch, config))
        action_sensor = (
            _adapt_gaze_free_sensor(
                batch["sensor"], action_sensor_dim, hand_dim, pose_dim
            )
            if action_model is not None
            else None
        )
        action_output = (
            action_model(
                batch["motion"],
                batch["appearance"],
                action_sensor,
                valid_mask=batch["valid_mask"],
                modality_mask=batch["modality_mask"],
                motion_aux=batch["motion_aux"],
            )
            if action_model is not None
            else None
        )
    timestamps = arrays.get("timestamps")
    if timestamps is None:
        timestamps = np.arange(length, dtype=np.float64) * 0.5
    result = {
        "recording_id": record.recording_id,
        "split": record.split,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "step": arrays["step"],
        "state": arrays["state"],
        "state_mask": arrays["state_mask"].astype(bool),
        "component_outcome": arrays["component_outcome"],
        "component_features": output["component_features"][0]
        .float()
        .cpu()
        .numpy(),
        "state_probabilities": output["state_probabilities"][0]
        .float()
        .cpu()
        .numpy(),
        "effect_probabilities": output["effect_probabilities"][0]
        .float()
        .cpu()
        .numpy(),
    }
    if action_output is not None:
        result["predicted_step"] = (
            action_output["step_probabilities"][0].argmax(dim=-1).cpu().numpy()
        )
    elif output["step_probabilities"] is not None:
        result["predicted_step"] = (
            output["step_probabilities"][0].argmax(dim=-1).cpu().numpy()
        )
    return result


def _adapt_gaze_free_sensor(sensor, target_dim, hand_dim, pose_dim):
    """Map hands+pose into a legacy-width tensor while keeping gaze zeroed."""
    if target_dim is None or sensor.shape[-1] == target_dim:
        return sensor
    if hand_dim + pose_dim != sensor.shape[-1] or target_dim < sensor.shape[-1]:
        raise ValueError(
            "Cannot safely adapt the gaze-free sensor contract to the action model."
        )
    adapted = sensor.new_zeros(*sensor.shape[:-1], target_dim)
    adapted[..., :hand_dim] = sensor[..., :hand_dim]
    if pose_dim:
        adapted[..., -pose_dim:] = sensor[..., hand_dim : hand_dim + pose_dim]
    return adapted


def _duration_minutes(record: dict[str, Any]) -> float:
    timestamps = record["timestamps"]
    if len(timestamps) < 2:
        return 0.0
    interval = float(np.median(np.diff(timestamps)))
    return max(float(timestamps[-1] - timestamps[0] + interval) / 60.0, 0.0)


def _normal_examples(records):
    examples: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for record in records:
        for component in range(record["state"].shape[1]):
            rows = np.where(
                record["state_mask"][:, component]
                & (record["state"][:, component] == 2)
            )[0]
            for row in rows:
                step = int(record["step"][row])
                if step >= 0:
                    examples[(component, step)].append(
                        record["component_features"][row, component]
                    )
    return examples


def _attach_anomaly_scores(records, bank) -> None:
    for record in records:
        if "predicted_step" in record:
            score, mask, sources = predicted_step_anomaly_scores(
                record["component_features"], record["predicted_step"], bank
            )
            record["prototype_score_sources"] = sources
        else:
            score, mask = component_anomaly_scores(record["component_features"], bank)
            record["prototype_score_sources"] = {
                "component_step": 0,
                "component_fallback": int(mask.sum()),
                "missing": int((~mask).sum()),
            }
        record["anomaly_scores"] = score
        record["anomaly_available"] = mask


def _targets(records, event_state_indices):
    targets = []
    seen = set()
    for record in records:
        rows, event_components = np.where(record["component_outcome"] == 1)
        for row, event_component in zip(rows, event_components):
            state_component = int(event_state_indices[event_component])
            key = (record["recording_id"], int(row), state_component)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                (
                    record["recording_id"],
                    int(row),
                    state_component,
                    float(record["timestamps"][row]),
                )
            )
    return targets


def _target_signal_diagnostics(records, event_state_indices):
    rows = []
    for record in records:
        event_rows, event_components = np.where(record["component_outcome"] == 1)
        for row, event_component in zip(event_rows, event_components):
            state_component = int(event_state_indices[event_component])
            effect = record["effect_probabilities"][row, state_component]
            item = {
                "recording_id": record["recording_id"],
                "row": int(row),
                "timestamp": float(record["timestamps"][row]),
                "event_component": int(event_component),
                "state_component": state_component,
                "true_step": int(record["step"][row]),
                "completion_probability": float(effect[1] + effect[2]),
                "incorrect_effect_probability": float(effect[2]),
                "incorrect_state_probability": float(
                    record["state_probabilities"][row, state_component, 2]
                ),
                "component_anomaly_score": float(
                    record["anomaly_scores"][row, state_component]
                ),
            }
            if "predicted_step" in record:
                item["predicted_step"] = int(record["predicted_step"][row])
                item["predicted_step_correct"] = bool(
                    item["predicted_step"] == item["true_step"]
                )
            rows.append(item)
    completion = np.asarray(
        [row["completion_probability"] for row in rows], dtype=np.float64
    )
    anomaly = np.asarray(
        [row["component_anomaly_score"] for row in rows], dtype=np.float64
    )
    return {
        "events": rows,
        "completion_probability_mean": (
            float(completion.mean()) if len(completion) else None
        ),
        "completion_probability_max": (
            float(completion.max()) if len(completion) else None
        ),
        "completion_candidate_recall": {
            str(threshold): (
                100.0 * float((completion >= threshold).mean())
                if len(completion)
                else 0.0
            )
            for threshold in (0.1, 0.2, 0.35, 0.5)
        },
        "component_anomaly_mean": (
            float(anomaly.mean()) if len(anomaly) else None
        ),
        "component_anomaly_max": (
            float(anomaly.max()) if len(anomaly) else None
        ),
    }


def _evaluate_config(records, targets, config, tolerance_seconds):
    alerts = []
    duration = 0.0
    peak_incorrect = 0.0
    for record in records:
        trace = run_streaming_tracker(
            record["state_probabilities"],
            record["anomaly_scores"],
            record["anomaly_available"],
            record["effect_probabilities"],
            config,
        )
        duration += _duration_minutes(record)
        peak_incorrect = max(
            peak_incorrect, float(trace.posterior[..., 2].max(initial=0.0))
        )
        for event in trace.events:
            if event.kind == "alert":
                alerts.append(
                    (
                        record["recording_id"],
                        event.step,
                        event.component,
                        float(record["timestamps"][event.step]),
                    )
                )
    metrics = match_streaming_alerts(
        alerts,
        targets,
        duration_minutes=duration,
        tolerance_seconds=tolerance_seconds,
    )
    metrics["duration_minutes"] = duration
    metrics["peak_incorrect_posterior"] = peak_incorrect
    return metrics


def _calibrate(args, records, targets):
    trials = []
    best = None
    eligible_trials = 0
    for values in itertools.product(
        args.completion_thresholds,
        args.anomaly_thresholds,
        args.alert_thresholds,
        args.confirmation_steps,
    ):
        config = StreamingConfig(
            completion_threshold=values[0],
            anomaly_threshold=values[1],
            anomaly_temperature=args.anomaly_temperature,
            alert_threshold=values[2],
            confirmation_steps=values[3],
        )
        metrics = _evaluate_config(
            records, targets, config, args.tolerance_seconds
        )
        row = {"config": config.__dict__, "metrics": metrics}
        trials.append(row)
        eligible = (
            metrics["false_alerts_per_minute"]
            <= args.max_false_alerts_per_minute + 1e-12
        )
        eligible_trials += int(eligible)
        rank = (
            int(eligible),
            metrics["recall"] if eligible else -metrics["false_alerts_per_minute"],
            metrics["precision"],
            -metrics["false_alerts_per_minute"],
            -abs(metrics["mean_signed_delay_seconds"] or 0.0),
        )
        if best is None or rank > best[0]:
            best = (rank, row)
    assert best is not None
    selection = dict(best[1])
    selection["constraint_satisfied"] = eligible_trials > 0
    selection["eligible_trials"] = eligible_trials
    selection["total_trials"] = len(trials)
    return selection, trials


def _select_existing_trials(trials, max_false_alerts_per_minute):
    best = None
    eligible_trials = 0
    for row in trials:
        metrics = row["metrics"]
        eligible = (
            metrics["false_alerts_per_minute"]
            <= max_false_alerts_per_minute + 1e-12
        )
        eligible_trials += int(eligible)
        rank = (
            int(eligible),
            metrics["recall"] if eligible else -metrics["false_alerts_per_minute"],
            metrics["precision"],
            -metrics["false_alerts_per_minute"],
            -abs(metrics["mean_signed_delay_seconds"] or 0.0),
        )
        if best is None or rank > best[0]:
            best = (rank, row)
    assert best is not None
    selection = dict(best[1])
    selection["constraint_satisfied"] = eligible_trials > 0
    selection["eligible_trials"] = eligible_trials
    selection["total_trials"] = len(trials)
    selection["max_false_alerts_per_minute"] = max_false_alerts_per_minute
    return selection


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    metadata, indexed_records = read_cache_index(args.cache_index)
    _validate_contract(metadata, indexed_records)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = StateEffectObserverConfig(**checkpoint["model_config"])
    model = build_state_effect_observer(model_config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    action_model = None
    action_config = None
    if args.action_checkpoint:
        action_checkpoint = torch.load(
            args.action_checkpoint, map_location="cpu", weights_only=False
        )
        action_config = StateGraphPSRConfig(**action_checkpoint["model_config"])
        action_model = build_stategraph_psr(
            action_config, action_checkpoint["transition_matrix"]
        )
        action_model.load_state_dict(action_checkpoint["model_state"])
        action_model.to(device).eval()
    records = []
    sensor_contract = metadata["sensor_contract"]
    for index, record in enumerate(indexed_records, start=1):
        records.append(
            _infer_record(
                record,
                model,
                model_config,
                device,
                action_model=action_model,
                action_sensor_dim=(
                    action_config.sensor_dim if action_config is not None else None
                ),
                hand_dim=int(sensor_contract["hand_dim"]),
                pose_dim=int(sensor_contract["pose_dim"]),
            )
        )
        print(f"[{index}/{len(indexed_records)}] inferred {record.recording_id}", flush=True)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "val"]
    bank = fit_prototype_bank(
        _normal_examples(train_records),
        prototypes=args.prototypes,
        minimum_step_samples=args.minimum_step_samples,
        maximum_samples=args.maximum_samples_per_group,
        seed=args.seed,
    )
    save_prototype_bank(
        args.prototype_bank,
        bank,
        metadata={
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "cache_index": str(Path(args.cache_index).resolve()),
            "prototypes": args.prototypes,
            "fit_split": "train_correct_states_only",
            "online_scoring": (
                "causal_predicted_step_with_component_fallback"
                if action_model is not None
                else "component_fallback_only"
            ),
        },
    )
    _attach_anomaly_scores(records, bank)
    event_state_indices = tuple(checkpoint["event_state_mapping"]["indices"])
    train_targets = _targets(train_records, event_state_indices)
    validation_targets = _targets(validation_records, event_state_indices)
    selection, trials = _calibrate(args, train_records, train_targets)
    selected_config = StreamingConfig(**selection["config"])
    validation_metrics = _evaluate_config(
        validation_records,
        validation_targets,
        selected_config,
        args.tolerance_seconds,
    )
    operating_curve = {}
    budgets = list(
        dict.fromkeys(
            [args.max_false_alerts_per_minute, *args.false_alert_budgets]
        )
    )
    for budget in budgets:
        budget_selection = _select_existing_trials(trials, budget)
        budget_config = StreamingConfig(**budget_selection["config"])
        operating_curve[str(budget)] = {
            "selection": budget_selection,
            "validation": _evaluate_config(
                validation_records,
                validation_targets,
                budget_config,
                args.tolerance_seconds,
            ),
        }
    result = {
        "protocol": {
            "causal": True,
            "prototype_fit_split": "train",
            "calibration_split": "train",
            "evaluation_split": "val",
            "ground_truth_step_used_online": False,
            "prototype_online_grouping": (
                "predicted_step_with_component_fallback"
                if "predicted_step" in records[0]
                else "component_only"
            ),
            "candidate_source": "observer P(complete_correct)+P(complete_incorrect)",
            "action_sensor_contract": (
                "hands + zeroed legacy gaze slot + headset pose"
                if action_model is not None
                and action_config.sensor_dim != int(sensor_contract["sensor_dim"])
                else "native gaze-free hands + headset pose"
            ),
            "max_training_false_alerts_per_minute": args.max_false_alerts_per_minute,
            "tolerance_seconds": args.tolerance_seconds,
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "action_checkpoint": (
            str(Path(args.action_checkpoint).resolve())
            if args.action_checkpoint
            else None
        ),
        "cache_index": str(Path(args.cache_index).resolve()),
        "prototype_bank": str(Path(args.prototype_bank).resolve()),
        "prototype_groups": len(bank),
        "validation_action_frame_accuracy": (
            _action_frame_accuracy(validation_records)
            if "predicted_step" in validation_records[0]
            else None
        ),
        "prototype_score_sources": _score_source_totals(records),
        "train_incorrect_events": len(train_targets),
        "validation_incorrect_events": len(validation_targets),
        "train_target_signal_diagnostics": _target_signal_diagnostics(
            train_records, event_state_indices
        ),
        "validation_target_signal_diagnostics": _target_signal_diagnostics(
            validation_records, event_state_indices
        ),
        "selected": selection,
        "validation": validation_metrics,
        "train_selected_operating_curve": operating_curve,
        "calibration_trials": trials,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _action_frame_accuracy(records) -> float:
    correct = 0
    total = 0
    for record in records:
        mask = record["step"] >= 0
        correct += int((record["predicted_step"][mask] == record["step"][mask]).sum())
        total += int(mask.sum())
    return 100.0 * correct / max(total, 1)


def _score_source_totals(records) -> dict[str, int]:
    totals = {"component_step": 0, "component_fallback": 0, "missing": 0}
    for record in records:
        for key in totals:
            totals[key] += int(record["prototype_score_sources"][key])
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--action-checkpoint",
        help="Optional causal StateGraph checkpoint for predicted-step prototype routing.",
    )
    parser.add_argument("--prototype-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prototypes", type=int, default=5)
    parser.add_argument("--minimum-step-samples", type=int, default=20)
    parser.add_argument("--maximum-samples-per-group", type=int, default=20000)
    parser.add_argument("--completion-thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.35, 0.5])
    parser.add_argument("--anomaly-thresholds", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--alert-thresholds", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    parser.add_argument("--confirmation-steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--anomaly-temperature", type=float, default=0.75)
    parser.add_argument("--max-false-alerts-per-minute", type=float, default=0.1)
    parser.add_argument(
        "--false-alert-budgets",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0],
        help="Additional train-only operating points evaluated unchanged on validation.",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"selected": result["selected"], "validation": result["validation"]}, indent=2))


if __name__ == "__main__":
    main()
