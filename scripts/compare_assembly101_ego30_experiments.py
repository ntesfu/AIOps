#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiops.training.train_stategraph_psr import _validation_selection_score


METRICS = (
    "frame_accuracy",
    "edit",
    "f1@50",
    "state_macro_f1",
    "state_incorrect_f1",
    "incorrect_event_f1",
    "normality_incorrect_average_precision",
    "incorrect_false_alerts_per_minute",
)


def compare(paths: list[Path], require_no_test: bool = True) -> dict:
    rows = []
    for path in paths:
        metrics_path = path / "metrics.json" if path.is_dir() else path
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if require_no_test and payload.get("test") is not None:
            raise ValueError(f"Pilot unexpectedly evaluated held-out test: {metrics_path}")
        validation = payload.get("best_validation") or payload.get("final_validation")
        if not validation:
            raise ValueError(f"No validation metrics in {metrics_path}")
        rows.append(
            {
                "run": metrics_path.parent.name,
                "best_epoch": payload.get("best_epoch"),
                "parameters": payload.get("parameters"),
                "peak_vram_gib": payload.get("peak_training_vram_gib"),
                "selection_score": _validation_selection_score(validation, mistake_weight=1.0),
                **{name: validation[name] for name in METRICS},
            }
        )
    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    return {"winner": rows[0]["run"] if rows else None, "runs": rows}


def markdown(report: dict) -> str:
    columns = ("run", "selection_score", "f1@50", "state_incorrect_f1", "incorrect_event_f1", "incorrect_false_alerts_per_minute", "peak_vram_gib")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in report["runs"]:
        values = []
        for name in columns:
            value = row.get(name)
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank validation-only Assembly101 ego30 pilots.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = compare(args.runs, require_no_test=not args.allow_test)
    result = json.dumps(report, indent=2) + "\n" + markdown(report)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
