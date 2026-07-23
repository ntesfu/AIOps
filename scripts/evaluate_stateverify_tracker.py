from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aiops.stateverify import (
    BeliefTrackerConfig,
    TripleResidualEvidence,
    TypedComponentBeliefTracker,
    canonicalize_stategraph_emissions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the StateVerify typed component belief tracker over saved causal "
            "emissions. The input NPZ must contain state_emissions [T,C,3] and "
            "procedure_residual, execution_residual, effect_residual arrays."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stategraph-order",
        action="store_true",
        help="Input emissions use historical incorrect,pending,correct class order.",
    )
    parser.add_argument("--alert-threshold", type=float, default=0.65)
    parser.add_argument("--recovery-threshold", type=float, default=0.70)
    parser.add_argument("--confirmation-steps", type=int, default=2)
    parser.add_argument("--recovery-confirmation-steps", type=int, default=2)
    parser.add_argument(
        "--seconds-per-step",
        type=float,
        help="Override the cache update interval stored in the archive.",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=1.0)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as payload:
        state_emissions = payload["state_emissions"]
        if args.stategraph_order:
            state_emissions = canonicalize_stategraph_emissions(state_emissions)
        residuals = TripleResidualEvidence.from_arrays(
            payload["procedure_residual"],
            payload["execution_residual"],
            payload["effect_residual"],
            num_components=state_emissions.shape[1],
            procedure_mask=payload["procedure_mask"]
            if "procedure_mask" in payload
            else None,
            execution_mask=payload["execution_mask"]
            if "execution_mask" in payload
            else None,
            effect_mask=payload["effect_mask"] if "effect_mask" in payload else None,
        )
        component_names = (
            [str(value) for value in payload["component_names"].tolist()]
            if "component_names" in payload
            else None
        )
        component_outcome = (
            payload["component_outcome"].astype(np.int64)
            if "component_outcome" in payload
            else None
        )
        stored_seconds_per_step = (
            float(payload["seconds_per_step"])
            if "seconds_per_step" in payload
            else 0.5
        )

    config = BeliefTrackerConfig(
        alert_threshold=args.alert_threshold,
        recovery_threshold=args.recovery_threshold,
        confirmation_steps=args.confirmation_steps,
        recovery_confirmation_steps=args.recovery_confirmation_steps,
    )
    tracker = TypedComponentBeliefTracker(state_emissions.shape[1], config)
    trace = tracker.run(state_emissions, residuals)
    report = trace.to_dict(component_names)
    report["config"] = {
        "alert_threshold": args.alert_threshold,
        "recovery_threshold": args.recovery_threshold,
        "confirmation_steps": args.confirmation_steps,
        "recovery_confirmation_steps": args.recovery_confirmation_steps,
        "input": str(args.input),
    }
    report["summary"] = {
        "alerts": sum(event.kind == "alert" for event in trace.events),
        "recoveries": sum(event.kind == "recovery" for event in trace.events),
        "peak_incorrect_posterior": float(
            trace.posterior[..., 2].max(initial=0.0)
        ),
    }
    seconds_per_step = (
        args.seconds_per_step
        if args.seconds_per_step is not None
        else stored_seconds_per_step
    )
    if seconds_per_step <= 0 or args.tolerance_seconds < 0:
        parser.error("Time intervals must be positive and tolerance cannot be negative.")
    if component_outcome is not None:
        report["incorrect_alert_metrics"] = _incorrect_alert_metrics(
            component_outcome,
            trace.events,
            seconds_per_step=seconds_per_step,
            tolerance_seconds=args.tolerance_seconds,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


def _incorrect_alert_metrics(
    component_outcome: np.ndarray,
    events,
    *,
    seconds_per_step: float,
    tolerance_seconds: float,
) -> dict[str, float | int | None]:
    outcomes = np.asarray(component_outcome, dtype=np.int64)
    if outcomes.ndim != 2:
        raise ValueError("component_outcome must have shape [time, component].")
    targets = [
        (int(row), int(component))
        for row, component in zip(*np.where(outcomes == 1))
    ]
    predictions = [
        (int(event.step), int(event.component))
        for event in events
        if event.kind == "alert"
    ]
    tolerance_steps = int(round(tolerance_seconds / seconds_per_step))
    unmatched = set(range(len(targets)))
    delays: list[float] = []
    matched = 0
    for prediction_step, component in predictions:
        candidates = [
            index
            for index in unmatched
            if targets[index][1] == component
            and abs(prediction_step - targets[index][0]) <= tolerance_steps
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda index: abs(prediction_step - targets[index][0]),
        )
        unmatched.remove(selected)
        matched += 1
        delays.append((prediction_step - targets[selected][0]) * seconds_per_step)
    precision = matched / len(predictions) if predictions else 0.0
    recall = matched / len(targets) if targets else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    duration_minutes = outcomes.shape[0] * seconds_per_step / 60.0
    false_alerts = len(predictions) - matched
    return {
        "true_events": len(targets),
        "predicted_alerts": len(predictions),
        "matched_alerts": matched,
        "precision_percent": 100.0 * precision,
        "recall_percent": 100.0 * recall,
        "f1_percent": 100.0 * f1,
        "false_alerts_per_minute": (
            false_alerts / duration_minutes if duration_minutes > 0 else 0.0
        ),
        "mean_signed_delay_seconds": (
            float(np.mean(delays)) if delays else None
        ),
        "tolerance_seconds": tolerance_seconds,
    }


if __name__ == "__main__":
    main()
