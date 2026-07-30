#!/usr/bin/env python3
"""Summarize operator-disjoint StateGraph runs without touching sealed test data."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "frame_accuracy",
    "edit",
    "f1@50",
    "incorrect_event_precision",
    "incorrect_event_recall",
    "incorrect_event_f1",
    "normality_incorrect_average_precision",
    "state_incorrect_f1",
    "incorrect_false_alerts_per_minute",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_diagnostics(path: Path) -> dict[str, Any]:
    observations = [
        json.loads(line)["validation"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    within_budget = [
        row
        for row in observations
        if float(row.get("incorrect_false_alerts_per_minute", float("inf"))) <= 2.0
    ]
    positive_recall = [
        row for row in observations if float(row.get("incorrect_event_recall", 0.0)) > 0
    ]
    return {
        "evaluated_epochs": len(observations),
        "best_incorrect_recall_within_2_false_alerts_per_minute": max(
            (float(row.get("incorrect_event_recall", 0.0)) for row in within_budget),
            default=0.0,
        ),
        "best_incorrect_f1_within_2_false_alerts_per_minute": max(
            (float(row.get("incorrect_event_f1", 0.0)) for row in within_budget),
            default=0.0,
        ),
        "minimum_false_alerts_per_minute_with_positive_recall": min(
            (
                float(row.get("incorrect_false_alerts_per_minute", float("inf")))
                for row in positive_recall
            ),
            default=None,
        ),
        "best_normality_incorrect_average_precision": max(
            (
                float(row.get("normality_incorrect_average_precision", 0.0))
                for row in observations
            ),
            default=0.0,
        ),
    }


def summarize(
    repo_root: Path,
    *,
    run_prefix_base: str,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in range(folds):
        prefix = f"{run_prefix_base}_fold{fold}_s{seed}"
        action_path = repo_root / "runs" / f"{prefix}_action" / "metrics.json"
        event_path = repo_root / "runs" / f"{prefix}_event" / "metrics.json"
        event_history_path = (
            repo_root / "runs" / f"{prefix}_event" / "history.jsonl"
        )
        validation_path = (
            repo_root / "runs" / f"{prefix}_dual_expert" / "validation.json"
        )
        row: dict[str, Any] = {
            "fold": fold,
            "prefix": prefix,
            "status": "missing",
        }
        if action_path.exists():
            action = _json(action_path)
            row["action_best_epoch"] = action.get("best_epoch")
            row["action_peak_training_vram_gib"] = action.get(
                "peak_training_vram_gib"
            )
        if event_path.exists():
            event = _json(event_path)
            row["event_best_epoch"] = event.get("best_epoch")
            row["event_peak_training_vram_gib"] = event.get(
                "peak_training_vram_gib"
            )
        if event_history_path.exists():
            row["event_history"] = _history_diagnostics(event_history_path)
        if validation_path.exists():
            metrics = _json(validation_path)["metrics"]
            row.update({metric: metrics.get(metric) for metric in METRICS})
            row["status"] = "operational_checkpoint"
        elif event_history_path.exists():
            row["status"] = "no_operational_checkpoint"
        elif action_path.exists():
            row["status"] = "action_only"
        rows.append(row)

    completed = [row for row in rows if row["status"] == "operational_checkpoint"]
    aggregates: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in completed
            if row.get(metric) is not None
        ]
        aggregates[metric] = {
            "mean": statistics.fmean(values) if values else None,
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    return {
        "run_prefix_base": run_prefix_base,
        "seed": seed,
        "folds": folds,
        "test_remained_sealed": True,
        "operational_checkpoint_folds": len(completed),
        "operational_pass_rate": len(completed) / folds,
        "rows": rows,
        "aggregates_over_operational_folds": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-prefix-base", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        args.repo_root.resolve(),
        run_prefix_base=args.run_prefix_base,
        seed=args.seed,
        folds=args.folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
