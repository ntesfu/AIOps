#!/usr/bin/env python3
"""Run a paired frozen-backbone ablation through the existing staged trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


LABEL_KEYS = (
    "step",
    "completion",
    "component_outcome",
    "state",
    "state_mask",
    "boundary",
    "timestamps",
    "prediction_frame_indices",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_map(index_path: Path) -> dict[str, dict[str, Any]]:
    payload = _json(index_path)
    return {row["recording_id"]: row for row in payload["records"]}


def validate_cache_pair(
    swin_index: Path, videomae_index: Path, *, check_arrays: bool = True
) -> dict[str, Any]:
    """Prove that paired caches differ in representation, not labels or split."""
    left = _record_map(swin_index)
    right = _record_map(videomae_index)
    if set(left) != set(right):
        raise ValueError("Backbone caches contain different recording IDs")
    structural = (
        "split",
        "num_steps",
        "appearance_dim",
        "sensor_dim",
        "num_components",
        "num_completion_components",
        "num_frames",
        "motion_aux_dim",
    )
    label_digest = hashlib.sha256()
    for recording_id in sorted(left):
        for key in structural:
            if left[recording_id].get(key, 0) != right[recording_id].get(key, 0):
                raise ValueError(f"{recording_id}: cache contract differs at {key}")
        if not check_arrays:
            continue
        paths = (
            (swin_index.parent / left[recording_id]["path"]).resolve(),
            (videomae_index.parent / right[recording_id]["path"]).resolve(),
        )
        with np.load(paths[0], allow_pickle=False) as a, np.load(
            paths[1], allow_pickle=False
        ) as b:
            for key in LABEL_KEYS:
                if key not in a.files or key not in b.files:
                    raise ValueError(f"{recording_id}: paired caches lack {key}")
                if not np.array_equal(a[key], b[key], equal_nan=True):
                    raise ValueError(f"{recording_id}: paired labels differ at {key}")
                label_digest.update(recording_id.encode())
                label_digest.update(key.encode())
                label_digest.update(np.ascontiguousarray(a[key]).tobytes())
    return {
        "recordings": len(left),
        "splits": {
            split: sum(row["split"] == split for row in left.values())
            for split in sorted({row["split"] for row in left.values()})
        },
        "array_labels_checked": check_arrays,
        "paired_label_sha256": label_digest.hexdigest() if check_arrays else None,
        "swin_index_sha256": _sha256(swin_index),
        "videomae_index_sha256": _sha256(videomae_index),
    }


def _gpu_sample() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        row = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        ).stdout.splitlines()[0]
        memory, utilization, power = (float(value.strip()) for value in row.split(","))
        return {
            "device_memory_used_mib": memory,
            "device_utilization_percent": utilization,
            "device_power_watts": power,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run_monitored(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    samples_path: Path,
    interval: int,
) -> dict[str, Any]:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    process = subprocess.Popen(command, cwd=cwd, env=env)
    samples: list[dict[str, Any]] = []
    while process.poll() is None:
        sample: dict[str, Any] = {"elapsed_seconds": time.time() - started}
        gpu = _gpu_sample()
        if gpu:
            sample.update(gpu)
        samples.append(sample)
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample) + "\n")
        try:
            process.wait(timeout=interval)
        except subprocess.TimeoutExpired:
            pass
    elapsed = time.time() - started
    summary: dict[str, Any] = {
        "command": command,
        "return_code": process.returncode,
        "wall_seconds": elapsed,
        "samples": len(samples),
        "resource_scope": "whole GPU device; trainer metrics contain process-local peak allocation",
    }
    for key in (
        "device_memory_used_mib",
        "device_utilization_percent",
        "device_power_watts",
    ):
        values = [float(row[key]) for row in samples if key in row]
        summary[f"peak_{key}"] = max(values) if values else None
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
    return summary


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def build_jobs(
    config: dict[str, Any],
    *,
    stage: str,
    caches: dict[str, Path],
    selected_backbones: set[str],
) -> list[dict[str, Any]]:
    stage_config = config["stages"][stage]
    jobs = []
    for backbone in config["backbones"]:
        if backbone not in selected_backbones:
            continue
        for seed in stage_config["seeds"]:
            prefix = f"{config['name']}_{stage}_{backbone}_s{seed}"
            jobs.append(
                {
                    "stage": stage,
                    "backbone": backbone,
                    "seed": int(seed),
                    "cache_index": str(caches[backbone]),
                    "run_prefix": prefix,
                    "trainer_stage": stage_config["trainer_stage"],
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "validate", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--swin-cache", type=Path, required=True)
    parser.add_argument("--videomae-cache", type=Path, required=True)
    parser.add_argument("--stage", choices=("quick", "promotion"), default="quick")
    parser.add_argument(
        "--backbone",
        action="append",
        choices=("swin3d_s", "videomaev2_b"),
        help="Repeat to select backbones; default runs both.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("experiments/backbones"))
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--skip-array-check", action="store_true")
    args = parser.parse_args()

    config = _json(args.config.resolve())
    caches = {
        "swin3d_s": args.swin_cache.resolve(),
        "videomaev2_b": args.videomae_cache.resolve(),
    }
    contract = validate_cache_pair(
        caches["swin3d_s"],
        caches["videomaev2_b"],
        check_arrays=not args.skip_array_check,
    )
    backbones = set(args.backbone or config["backbones"])
    jobs = build_jobs(config, stage=args.stage, caches=caches, selected_backbones=backbones)
    manifest = {
        "protocol": config,
        "stage": args.stage,
        "cache_contract": contract,
        "jobs": jobs,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "execution": {
            "git": _git_state(args.repo_root.resolve()),
            "staged_launcher_sha256": _sha256(
                args.repo_root.resolve() / "scripts/train_industreal_staged_baseline.sh"
            ),
            "python_bin": args.python_bin,
        },
    }
    output_root = (args.repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{args.stage}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.action in {"plan", "validate"}:
        print(json.dumps({"manifest": str(manifest_path), **contract, "jobs": jobs}, indent=2))
        return

    stage_config = config["stages"][args.stage]
    for job in jobs:
        expected_metrics = (
            args.repo_root.resolve()
            / "runs"
            / f"{job['run_prefix']}_action"
            / "metrics.json"
        )
        if expected_metrics.exists():
            raise SystemExit(
                f"Refusing to reuse completed run {expected_metrics}; use a new protocol name "
                "to preserve independent trials."
            )
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config["fixed_environment"].items()})
        env.update(
            {
                "CACHE_INDEX": job["cache_index"],
                "SEED": str(job["seed"]),
                "RUN_PREFIX": job["run_prefix"],
                "ACTION_EPOCHS": str(stage_config["action_epochs"]),
                "ACTION_PATIENCE": str(stage_config["action_patience"]),
                "EVENT_EPOCHS": str(stage_config.get("event_epochs", 0)),
                "EVENT_PATIENCE": str(stage_config.get("event_patience", 1)),
                "PYTHON_BIN": args.python_bin,
            }
        )
        resource_dir = output_root / "resources" / args.stage / job["backbone"]
        resource_dir.mkdir(parents=True, exist_ok=True)
        samples_path = resource_dir / f"seed-{job['seed']}.jsonl"
        samples_path.unlink(missing_ok=True)
        summary = run_monitored(
            ["bash", "scripts/train_industreal_staged_baseline.sh", job["trainer_stage"]],
            cwd=args.repo_root.resolve(),
            env=env,
            samples_path=samples_path,
            interval=int(config.get("resource_sampling_seconds", 10)),
        )
        summary.update(job)
        summary_path = resource_dir / f"seed-{job['seed']}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if summary["return_code"] != 0:
            raise SystemExit(
                f"{job['backbone']} seed {job['seed']} failed; see {summary_path}"
            )


if __name__ == "__main__":
    main()
