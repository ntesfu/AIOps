"""Participant-grouped IndustReal evaluation and causal event matching.

The official test split is deliberately outside the API's eligible split set.
This module is intended for model selection on the official train/validation
development pool; the official test set remains sealed for final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aiops.data.stategraph_cache import StateGraphCacheRecord, read_cache_index


DEVELOPMENT_SPLITS = frozenset({"train", "val", "validation"})
SEALED_SPLITS = frozenset({"test"})
_OPERATOR_PATTERN = re.compile(r"^(?P<operator>[0-9]{1,3})(?:_|$)")


@dataclass(frozen=True)
class EventEpisode:
    episode_id: str
    recording_id: str
    operator_id: str
    source_split: str
    outcome: str
    outcome_index: int
    component_index: int
    action_index: int
    event_row: int
    start_row: int
    end_row_exclusive: int
    timestamp: float | None
    prediction_frame_index: int | None
    context_histogram: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "recording_id": self.recording_id,
            "operator_id": self.operator_id,
            "source_split": self.source_split,
            "outcome": self.outcome,
            "outcome_index": self.outcome_index,
            "component_index": self.component_index,
            "action_index": self.action_index,
            "event_row": self.event_row,
            "start_row": self.start_row,
            "end_row_exclusive": self.end_row_exclusive,
            "timestamp": self.timestamp,
            "prediction_frame_index": self.prediction_frame_index,
            "context_histogram": list(self.context_histogram),
        }


def parse_industreal_operator_id(
    recording_id: str,
    operator_map: Mapping[str, str] | None = None,
) -> str:
    """Return participant identity without silently guessing unknown formats."""

    if operator_map is not None and recording_id in operator_map:
        operator = str(operator_map[recording_id]).strip()
        if not operator:
            raise ValueError(f"Empty operator mapping for {recording_id!r}")
        return operator
    match = _OPERATOR_PATTERN.match(recording_id)
    if match is None:
        raise ValueError(
            f"Cannot derive an IndustReal operator from {recording_id!r}; "
            "provide an explicit operator map."
        )
    return match.group("operator")


def prepare_grouped_evaluation(
    cache_index: str | Path,
    *,
    folds: int = 5,
    seed: int = 7,
    episode_radius: int = 32,
    matches_per_incorrect: int = 3,
    operator_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build participant-disjoint folds and within-partition matched episodes.

    Only cache rows labelled train/val/validation are eligible. Test NPZ files
    are not opened, which makes accidental test-set tuning harder.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if episode_radius < 0:
        raise ValueError("episode_radius cannot be negative")
    if matches_per_incorrect < 1:
        raise ValueError("matches_per_incorrect must be positive")

    metadata, records = read_cache_index(cache_index)
    development = [
        record for record in records if record.split.strip().lower() in DEVELOPMENT_SPLITS
    ]
    if not development:
        raise ValueError("No train/validation development records were found")

    record_operators = {
        record.recording_id: parse_industreal_operator_id(record.recording_id, operator_map)
        for record in development
    }
    operators = sorted(set(record_operators.values()))
    if folds > len(operators):
        raise ValueError(f"folds={folds} exceeds the {len(operators)} development operators")

    outcome_names = tuple(
        str(value) for value in metadata.get("event_outcomes", ("correct", "incorrect", "remove"))
    )
    try:
        correct_index = outcome_names.index("correct")
        incorrect_index = outcome_names.index("incorrect")
    except ValueError as exc:
        raise ValueError(
            "Cache metadata event_outcomes must contain correct and incorrect"
        ) from exc

    events_by_record: dict[str, list[EventEpisode]] = {}
    for record in development:
        events_by_record[record.recording_id] = _read_record_episodes(
            record,
            operator_id=record_operators[record.recording_id],
            outcome_names=outcome_names,
            selected_outcomes={correct_index, incorrect_index},
            radius=episode_radius,
        )

    assignment = _balanced_operator_assignment(
        development,
        record_operators,
        events_by_record,
        folds=folds,
        seed=seed,
    )
    fold_payloads: list[dict[str, Any]] = []
    for fold_index in range(folds):
        validation_operators = sorted(
            operator for operator, assigned in assignment.items() if assigned == fold_index
        )
        validation_set = set(validation_operators)
        train_records = [
            record for record in development
            if record_operators[record.recording_id] not in validation_set
        ]
        validation_records = [
            record for record in development
            if record_operators[record.recording_id] in validation_set
        ]
        train_episodes = [
            episode
            for record in train_records
            for episode in events_by_record[record.recording_id]
        ]
        validation_episodes = [
            episode
            for record in validation_records
            for episode in events_by_record[record.recording_id]
        ]
        fold_payloads.append(
            {
                "fold": fold_index,
                "train_operator_ids": sorted(set(operators) - validation_set),
                "validation_operator_ids": validation_operators,
                "train_recording_ids": sorted(record.recording_id for record in train_records),
                "validation_recording_ids": sorted(
                    record.recording_id for record in validation_records
                ),
                "train": _episode_partition(
                    train_episodes,
                    correct_index=correct_index,
                    incorrect_index=incorrect_index,
                    matches_per_incorrect=matches_per_incorrect,
                ),
                "validation": _episode_partition(
                    validation_episodes,
                    correct_index=correct_index,
                    incorrect_index=incorrect_index,
                    matches_per_incorrect=matches_per_incorrect,
                ),
            }
        )

    return {
        "schema_version": 1,
        "dataset": "IndustReal",
        "source_cache_index": str(Path(cache_index).expanduser().resolve()),
        "policy": {
            "group_unit": "operator",
            "eligible_source_splits": ["train", "val", "validation"],
            "official_test_sealed": True,
            "sealed_source_splits": sorted(SEALED_SPLITS),
            "event_window": "causal; ends at the labelled event row",
            "matching_scope": "within fold partition only",
        },
        "seed": seed,
        "fold_count": folds,
        "episode_radius": episode_radius,
        "matches_per_incorrect": matches_per_incorrect,
        "event_outcomes": list(outcome_names),
        "development_recordings": len(development),
        "development_operators": operators,
        "folds": fold_payloads,
    }


def write_grouped_evaluation(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_record_episodes(
    record: StateGraphCacheRecord,
    *,
    operator_id: str,
    outcome_names: Sequence[str],
    selected_outcomes: set[int],
    radius: int,
) -> list[EventEpisode]:
    with np.load(record.path, allow_pickle=False) as arrays:
        outcomes = np.asarray(arrays["component_outcome"], dtype=np.int64)
        actions = np.asarray(arrays["step"], dtype=np.int64)
        timestamps = (
            np.asarray(arrays["timestamps"], dtype=np.float64)
            if "timestamps" in arrays.files
            else None
        )
        prediction_frames = (
            np.asarray(arrays["prediction_frame_indices"], dtype=np.int64)
            if "prediction_frame_indices" in arrays.files
            else None
        )
    if outcomes.ndim != 2 or actions.ndim != 1 or len(outcomes) != len(actions):
        raise ValueError(f"{record.recording_id}: invalid component_outcome/step shapes")

    action_count = max(int(actions.max(initial=0)) + 1, record.num_steps)
    episodes: list[EventEpisode] = []
    for component in range(outcomes.shape[1]):
        for outcome_index in sorted(selected_outcomes):
            active = outcomes[:, component] == outcome_index
            starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
            for row_value in starts:
                row = int(row_value)
                start = max(0, row - radius)
                end = row + 1
                context = actions[start:end]
                histogram = np.bincount(
                    context[context >= 0], minlength=action_count
                ).astype(np.float64)
                if histogram.sum() > 0:
                    histogram /= histogram.sum()
                outcome = outcome_names[outcome_index]
                episode_id = (
                    f"{record.recording_id}:r{row}:c{component}:o{outcome_index}"
                )
                episodes.append(
                    EventEpisode(
                        episode_id=episode_id,
                        recording_id=record.recording_id,
                        operator_id=operator_id,
                        source_split=record.split,
                        outcome=outcome,
                        outcome_index=outcome_index,
                        component_index=component,
                        action_index=int(actions[row]),
                        event_row=row,
                        start_row=start,
                        end_row_exclusive=end,
                        timestamp=(
                            float(timestamps[row]) if timestamps is not None else None
                        ),
                        prediction_frame_index=(
                            int(prediction_frames[row])
                            if prediction_frames is not None
                            else None
                        ),
                        context_histogram=tuple(float(value) for value in histogram),
                    )
                )
    return sorted(episodes, key=lambda item: (item.recording_id, item.event_row, item.component_index))


def _episode_partition(
    episodes: Sequence[EventEpisode],
    *,
    correct_index: int,
    incorrect_index: int,
    matches_per_incorrect: int,
) -> dict[str, Any]:
    correct = [episode for episode in episodes if episode.outcome_index == correct_index]
    incorrect = [episode for episode in episodes if episode.outcome_index == incorrect_index]
    pairs: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for error in incorrect:
        candidates = [candidate for candidate in correct if candidate.component_index == error.component_index]
        ranked = sorted(candidates, key=lambda candidate: _match_key(error, candidate))
        if not ranked:
            unmatched.append(error.episode_id)
            continue
        for rank, match in enumerate(ranked[:matches_per_incorrect], start=1):
            pairs.append(
                {
                    "incorrect_episode_id": error.episode_id,
                    "correct_episode_id": match.episode_id,
                    "rank": rank,
                    "match_quality": _match_quality(error, match),
                    "context_similarity": _context_similarity(error, match),
                }
            )
    return {
        "recording_count": len({episode.recording_id for episode in episodes}),
        "operator_count": len({episode.operator_id for episode in episodes}),
        "correct_event_count": len(correct),
        "incorrect_event_count": len(incorrect),
        "matched_pair_count": len(pairs),
        "unmatched_incorrect_episode_ids": unmatched,
        "episodes": [episode.to_dict() for episode in (*incorrect, *correct)],
        "pairs": pairs,
    }


def _match_key(error: EventEpisode, candidate: EventEpisode) -> tuple[Any, ...]:
    return (
        error.action_index != candidate.action_index,
        error.operator_id == candidate.operator_id,
        -_context_similarity(error, candidate),
        candidate.recording_id,
        candidate.event_row,
        candidate.component_index,
    )


def _match_quality(error: EventEpisode, candidate: EventEpisode) -> str:
    action = "same_action" if error.action_index == candidate.action_index else "different_action"
    operator = "cross_operator" if error.operator_id != candidate.operator_id else "same_operator"
    return f"same_component_{action}_{operator}"


def _context_similarity(left: EventEpisode, right: EventEpisode) -> float:
    size = max(len(left.context_histogram), len(right.context_histogram))
    left_values = np.zeros(size, dtype=np.float64)
    right_values = np.zeros(size, dtype=np.float64)
    left_values[: len(left.context_histogram)] = left.context_histogram
    right_values[: len(right.context_histogram)] = right.context_histogram
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    return float(np.dot(left_values, right_values) / denominator) if denominator else 0.0


def _balanced_operator_assignment(
    records: Sequence[StateGraphCacheRecord],
    record_operators: Mapping[str, str],
    events_by_record: Mapping[str, Sequence[EventEpisode]],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    stats: dict[str, dict[str, int]] = {}
    for record in records:
        operator = record_operators[record.recording_id]
        current = stats.setdefault(operator, {"recordings": 0, "incorrect": 0})
        current["recordings"] += 1
        current["incorrect"] += sum(
            episode.outcome == "incorrect"
            for episode in events_by_record[record.recording_id]
        )

    def stable_tie(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

    ordered = sorted(
        stats,
        key=lambda operator: (
            -stats[operator]["incorrect"],
            -stats[operator]["recordings"],
            stable_tie(operator),
        ),
    )
    fold_stats = [
        {"incorrect": 0, "recordings": 0, "operators": 0} for _ in range(folds)
    ]
    assignment: dict[str, int] = {}
    for operator in ordered:
        has_incorrect = stats[operator]["incorrect"] > 0

        def fold_key(fold: int) -> tuple[Any, ...]:
            if has_incorrect:
                # Rare events dominate the assignment of error-bearing groups.
                return (
                    fold_stats[fold]["incorrect"],
                    fold_stats[fold]["recordings"],
                    fold_stats[fold]["operators"],
                    stable_tie(f"{operator}:{fold}"),
                )
            # Once rare events are distributed, zero-error participants should
            # repair recording/operator load rather than all joining the fold
            # with one fewer error.
            return (
                fold_stats[fold]["recordings"],
                fold_stats[fold]["operators"],
                fold_stats[fold]["incorrect"],
                stable_tie(f"{operator}:{fold}"),
            )

        target = min(
            range(folds),
            key=fold_key,
        )
        assignment[operator] = target
        fold_stats[target]["incorrect"] += stats[operator]["incorrect"]
        fold_stats[target]["recordings"] += stats[operator]["recordings"]
        fold_stats[target]["operators"] += 1
    return assignment
