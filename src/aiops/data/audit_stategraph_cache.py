from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.procedure_schema import CACHE_SCHEMA_VERSION
from aiops.data.stategraph_cache import read_cache_index


def audit_stategraph_cache(index_path: str | Path) -> dict[str, Any]:
    metadata, records = read_cache_index(index_path)
    action_names = list(metadata.get("action_ids", []))
    completion_names = list(metadata.get("completion_components", []))
    split_names = sorted({record.split for record in records})
    action_counts = {split: np.zeros(len(action_names), dtype=np.int64) for split in split_names}
    event_counts = {
        split: np.zeros((len(completion_names), 3), dtype=np.int64) for split in split_names
    }
    state_counts = {split: np.zeros(3, dtype=np.int64) for split in split_names}
    row_counts = {split: 0 for split in split_names}
    multi_component_rows = {split: 0 for split in split_names}
    invalid_records: list[dict[str, Any]] = []

    required = {
        "motion",
        "appearance",
        "sensor",
        "modality_mask",
        "step",
        "completion",
        "component_outcome",
        "state",
        "state_mask",
        "boundary",
        "timestamps",
    }
    for record in records:
        issues: list[str] = []
        with np.load(record.path, allow_pickle=False) as arrays:
            missing = sorted(required - set(arrays.files))
            if missing:
                invalid_records.append({"recording_id": record.recording_id, "issues": [f"missing {missing}"]})
                continue
            length = len(arrays["step"])
            if any(arrays[name].shape[0] != length for name in required):
                issues.append("inconsistent time dimension")
            if arrays["completion"].shape != (length, record.num_completion_components):
                issues.append("invalid completion shape")
            if arrays["component_outcome"].shape != arrays["completion"].shape:
                issues.append("component_outcome shape mismatch")
            if arrays["state"].shape != (length, record.num_components):
                issues.append("invalid state shape")
            if any(not np.isfinite(arrays[name]).all() for name in ("motion", "appearance", "sensor")):
                issues.append("non-finite feature values")
            labels = arrays["step"].astype(np.int64)
            if ((labels < 0) | (labels >= len(action_names))).any():
                issues.append("action labels outside vocabulary")
            else:
                action_counts[record.split] += np.bincount(labels, minlength=len(action_names))
            outcomes = arrays["component_outcome"].astype(np.int64)
            for outcome_index in range(3):
                event_counts[record.split][:, outcome_index] += (outcomes == outcome_index).sum(axis=0)
            state = arrays["state"].astype(np.int64)
            state_mask = arrays["state_mask"].astype(bool)
            for state_index in range(3):
                state_counts[record.split][state_index] += int(
                    ((state == state_index) & state_mask).sum()
                )
            row_counts[record.split] += length
            multi_component_rows[record.split] += int(
                (arrays["completion"].sum(axis=1) > 1).sum()
            )
        if issues:
            invalid_records.append({"recording_id": record.recording_id, "issues": issues})

    train_split = next((name for name in split_names if name.lower() == "train"), None)
    warnings: list[str] = []
    zero_train_actions: list[str] = []
    composable_zero_train_actions: list[str] = []
    uncomposable_zero_train_actions: list[str] = []
    zero_train_components: list[str] = []
    if train_split is None:
        warnings.append("No train split is present.")
    else:
        zero_train_actions = [
            action_names[index]
            for index, count in enumerate(action_counts[train_split])
            if count == 0
        ]
        descriptions = list(metadata.get("action_descriptions", action_names))
        taxonomy = {
            str(row.get("action_cls", "")).strip().lower(): (
                str(row.get("verb_cls", "")).strip().lower(),
                str(row.get("noun_cls", "")).strip().lower(),
            )
            for row in metadata.get("action_taxonomy", [])
        }
        factors = [
            taxonomy.get(action.lower(), _action_factors(description))
            for action, description in zip(action_names, descriptions)
        ]
        seen_verbs = {
            factors[index][0]
            for index, count in enumerate(action_counts[train_split])
            if count > 0
        }
        seen_objects = {
            factors[index][1]
            for index, count in enumerate(action_counts[train_split])
            if count > 0
        }
        for index, count in enumerate(action_counts[train_split]):
            if count > 0:
                continue
            target = (
                composable_zero_train_actions
                if factors[index][0] in seen_verbs and factors[index][1] in seen_objects
                else uncomposable_zero_train_actions
            )
            target.append(action_names[index])
        zero_train_components = [
            completion_names[index]
            for index, count in enumerate(event_counts[train_split].sum(axis=1))
            if count == 0
        ]
        if uncomposable_zero_train_actions:
            warnings.append(
                "Actions absent from train and not covered by seen verb/object factors: "
                f"{uncomposable_zero_train_actions}"
            )
        if zero_train_components:
            warnings.append(f"Completion components absent from train: {zero_train_components}")
    if invalid_records:
        warnings.append(f"Invalid cache records: {len(invalid_records)}")
    if int(metadata.get("schema_version", 1)) != CACHE_SCHEMA_VERSION:
        warnings.append(f"Expected cache schema v{CACHE_SCHEMA_VERSION}.")

    background_id = metadata.get("background_action_id")
    background_index = action_names.index(background_id) if background_id in action_names else None
    return {
        "index": str(Path(index_path).resolve()),
        "schema_version": int(metadata.get("schema_version", 1)),
        "label_contract": metadata.get("label_contract"),
        "recordings": len(records),
        "splits": {split: sum(record.split == split for record in records) for split in split_names},
        "rows": row_counts,
        "actions": len(action_names),
        "background_rows": {
            split: int(counts[background_index]) if background_index is not None else None
            for split, counts in action_counts.items()
        },
        "zero_train_actions": zero_train_actions,
        "composable_zero_train_actions": composable_zero_train_actions,
        "uncomposable_zero_train_actions": uncomposable_zero_train_actions,
        "event_outcome_names": list(metadata.get("event_outcomes", [])),
        "event_counts": {
            split: counts.sum(axis=0).astype(int).tolist() for split, counts in event_counts.items()
        },
        "events_by_component": {
            split: {
                name: int(count)
                for name, count in zip(completion_names, counts.sum(axis=1))
            }
            for split, counts in event_counts.items()
        },
        "zero_train_completion_components": zero_train_components,
        "multi_component_rows": multi_component_rows,
        "state_names": list(metadata.get("state_names", [])),
        "state_counts": {
            split: counts.astype(int).tolist() for split, counts in state_counts.items()
        },
        "invalid_records": invalid_records,
        "warnings": warnings,
    }


def _action_factors(description: str) -> tuple[str, str]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", description.lower()) if token]
    verb = tokens[0] if tokens else "unknown"
    object_name = "_".join(tokens[1:]) if len(tokens) > 1 else "none"
    return verb, object_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a StateGraph cache before training.")
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()
    report = audit_stategraph_cache(args.cache_index)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.fail_on_warnings and report["warnings"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
