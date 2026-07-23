#!/usr/bin/env python3
"""Produce a validation-only paired comparison for the backbone ablation."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "frame_accuracy",
    "edit",
    "f1@10",
    "f1@25",
    "f1@50",
    "incorrect_event_precision",
    "incorrect_event_recall",
    "incorrect_event_f1",
    "normality_incorrect_average_precision",
    "incorrect_false_alerts_per_minute",
)


def _validation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test") is not None:
        raise ValueError(f"Sealed test data found in development result: {path}")
    metrics = payload.get("best_validation") or payload.get("metrics")
    if not metrics:
        raise ValueError(f"No validation metrics found: {path}")
    return payload, metrics


def _result_path(repo: Path, name: str, stage: str, backbone: str, seed: int) -> Path:
    prefix = f"{name}_{stage}_{backbone}_s{seed}"
    if stage == "quick":
        return repo / "runs" / f"{prefix}_action" / "metrics.json"
    return repo / "runs" / f"{prefix}_dual_expert" / "validation.json"


def compare(
    config: dict[str, Any], repo: Path, stage: str, resource_root: Path | None = None
) -> dict[str, Any]:
    rows = []
    seeds = [int(seed) for seed in config["stages"][stage]["seeds"]]
    for backbone in config["backbones"]:
        for seed in seeds:
            path = _result_path(repo, config["name"], stage, backbone, seed)
            payload, metrics = _validation(path)
            resource_path = (
                (resource_root or repo / "experiments/backbones")
                / "resources"
                / stage
                / backbone
                / f"seed-{seed}.summary.json"
            )
            resource = (
                json.loads(resource_path.read_text(encoding="utf-8"))
                if resource_path.exists()
                else {}
            )
            rows.append(
                {
                    "backbone": backbone,
                    "seed": seed,
                    "result_path": str(path),
                    "best_epoch": payload.get("best_epoch"),
                    "peak_training_vram_gib": payload.get("peak_training_vram_gib"),
                    "wall_seconds": resource.get("wall_seconds"),
                    **{key: metrics.get(key) for key in METRICS},
                }
            )
    aggregates = {}
    for backbone in config["backbones"]:
        selected = [row for row in rows if row["backbone"] == backbone]
        aggregates[backbone] = {}
        for key in (*METRICS, "peak_training_vram_gib", "wall_seconds"):
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            aggregates[backbone][key] = {
                "mean": statistics.fmean(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            }
    paired = []
    for seed in seeds:
        reference = next(row for row in rows if row["backbone"] == "swin3d_s" and row["seed"] == seed)
        candidate = next(row for row in rows if row["backbone"] == "videomaev2_b" and row["seed"] == seed)
        paired.append(
            {
                "seed": seed,
                **{
                    f"{key}_delta": (
                        float(candidate[key]) - float(reference[key])
                        if candidate.get(key) is not None and reference.get(key) is not None
                        else None
                    )
                    for key in METRICS
                },
            }
        )
    gates = config["promotion_gates"]
    f1_delta = statistics.fmean(row["f1@50_delta"] for row in paired)
    frame_delta = statistics.fmean(row["frame_accuracy_delta"] for row in paired)
    candidate = aggregates["videomaev2_b"]
    decisions = {
        "f1_at_50_gain": f1_delta >= gates["minimum_paired_f1_at_50_gain_points"],
        "frame_accuracy_noninferior": frame_delta >= -gates["maximum_frame_accuracy_regression_points"],
        "vram_within_budget": (
            candidate["peak_training_vram_gib"]["mean"] is None
            or candidate["peak_training_vram_gib"]["mean"]
            <= gates["maximum_peak_training_vram_gib"]
        ),
    }
    if stage == "promotion":
        decisions["false_alert_budget"] = (
            candidate["incorrect_false_alerts_per_minute"]["mean"]
            <= gates["maximum_incorrect_false_alerts_per_minute"]
        )
        decisions["positive_mistake_recall"] = (
            not gates["require_positive_incorrect_event_recall"]
            or candidate["incorrect_event_recall"]["mean"] > 0
        )
    return {
        "protocol": config["name"],
        "stage": stage,
        "test_remained_sealed": True,
        "rows": rows,
        "aggregates": aggregates,
        "paired_candidate_minus_reference": paired,
        "gates": decisions,
        "promote_videomaev2_b": all(decisions.values()),
    }


def markdown(report: dict[str, Any]) -> str:
    columns = (
        "backbone", "seed", "f1@50", "edit", "frame_accuracy",
        "incorrect_event_recall", "incorrect_false_alerts_per_minute",
        "peak_training_vram_gib", "wall_seconds",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in report["rows"]:
        values = []
        for key in columns:
            value = row.get(key)
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", f"Promote VideoMAE V2-B: **{report['promote_videomaev2_b']}**", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/backbones"),
        help="Resource-monitor root, relative to repo-root unless absolute.",
    )
    parser.add_argument("--stage", choices=("quick", "promotion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resource_root = args.experiment_root
    if not resource_root.is_absolute():
        resource_root = args.repo_root.resolve() / resource_root
    report = compare(config, args.repo_root.resolve(), args.stage, resource_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "promote": report["promote_videomaev2_b"]}, indent=2))


if __name__ == "__main__":
    main()
