from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("frame_accuracy", "Frame accuracy"),
    ("edit", "Edit"),
    ("f1@50", "F1@50"),
    ("state_macro_f1", "State macro-F1"),
    ("state_incorrect_f1", "Incorrect-state F1"),
    ("normality_incorrect_average_precision", "Mistake AP"),
    ("incorrect_event_pr_auc", "Incorrect-event PR-AUC"),
    ("incorrect_event_f1", "Incorrect-event F1"),
    ("incorrect_event_recall", "Incorrect-event recall"),
    ("incorrect_event_f1_tol4s", "Incorrect-event F1 ±4s"),
    ("incorrect_false_alerts_per_minute", "False alerts/min"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(config_path: Path, runs_root: Path) -> str:
    config = _load(config_path)
    experiments = config["experiments"]
    by_id: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        metrics_path = runs_root / experiment["run"] / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing completed run metrics: {metrics_path}")
        summary = _load(metrics_path)
        by_id[experiment["id"]] = {
            "experiment": experiment,
            "summary": summary,
            "metrics": summary["final_validation"],
        }

    lines = [
        "# Assembly101 ego-30 controlled ablation report",
        "",
        f"Source commit: `{config['protocol']['source_commit']}`.",
        "The held-out test split was not evaluated in this screening wave.",
        "",
        "## Absolute validation results",
        "",
        "| Run | Factor | " + " | ".join(label for _, label in METRICS) + " | VRAM GiB | Best epoch |",
        "|---|---|" + "---:|" * (len(METRICS) + 2),
    ]
    for experiment in experiments:
        row = by_id[experiment["id"]]
        metrics = row["metrics"]
        summary = row["summary"]
        values = [f"{float(metrics.get(key, float('nan'))):.3f}" for key, _ in METRICS]
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["id"],
                    experiment["factor"],
                    *values,
                    f"{float(summary['peak_training_vram_gib']):.3f}",
                    str(summary["best_epoch"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Attributed deltas",
            "",
            "A positive delta is better except for false alerts/minute, where a negative delta is better.",
            "",
            "| Contrast | Isolated factor | " + " | ".join(label for _, label in METRICS) + " |",
            "|---|---|" + "---:|" * len(METRICS),
        ]
    )
    for experiment in experiments:
        baseline_id = experiment.get("compare_to")
        if not baseline_id:
            continue
        current = by_id[experiment["id"]]["metrics"]
        baseline = by_id[baseline_id]["metrics"]
        deltas = [
            f"{float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)):+.3f}"
            for key, _ in METRICS
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{experiment['id']}−{baseline_id}",
                    experiment["factor"],
                    *deltas,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report controlled Assembly101 ablations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
