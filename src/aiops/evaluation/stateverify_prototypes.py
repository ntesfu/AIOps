from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PrototypeGroup:
    centers: np.ndarray
    residual_median: float
    residual_scale: float
    samples: int


def save_prototype_bank(
    path: str | Path,
    bank: dict[tuple[int, int | None], PrototypeGroup],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Persist a prototype bank without pickle so it is portable and auditable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, ((component, step), group) in enumerate(
        sorted(bank.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or -1))
    ):
        array_key = f"centers_{index}"
        arrays[array_key] = np.asarray(group.centers, dtype=np.float32)
        manifest.append(
            {
                "component": int(component),
                "step": None if step is None else int(step),
                "centers": array_key,
                "residual_median": float(group.residual_median),
                "residual_scale": float(group.residual_scale),
                "samples": int(group.samples),
            }
        )
    payload = {
        "format": "aiops.stateverify.prototype_bank",
        "version": 1,
        "groups": manifest,
        "metadata": metadata or {},
    }
    arrays["manifest_json"] = np.asarray(json.dumps(payload, sort_keys=True))
    np.savez_compressed(destination, **arrays)


def load_prototype_bank(
    path: str | Path,
) -> tuple[dict[tuple[int, int | None], PrototypeGroup], dict[str, object]]:
    """Load a bank written by :func:`save_prototype_bank`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = json.loads(str(archive["manifest_json"].item()))
        if (
            payload.get("format") != "aiops.stateverify.prototype_bank"
            or payload.get("version") != 1
        ):
            raise ValueError("Unsupported StateVerify prototype-bank format.")
        bank: dict[tuple[int, int | None], PrototypeGroup] = {}
        for row in payload["groups"]:
            key = (int(row["component"]), None if row["step"] is None else int(row["step"]))
            bank[key] = PrototypeGroup(
                centers=np.asarray(archive[row["centers"]], dtype=np.float32),
                residual_median=float(row["residual_median"]),
                residual_scale=float(row["residual_scale"]),
                samples=int(row["samples"]),
            )
    return bank, dict(payload.get("metadata", {}))


def fit_prototype_group(
    features: np.ndarray,
    *,
    prototypes: int = 3,
    iterations: int = 25,
    seed: int = 7,
) -> PrototypeGroup:
    values = _normalize(np.asarray(features, dtype=np.float32))
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("features must be a non-empty [samples, dimension] array.")
    count = min(max(int(prototypes), 1), len(values))
    del seed  # deterministic farthest-point initialization
    mean_direction = _normalize(values.mean(axis=0, keepdims=True))[0]
    first = int(np.argmax(values @ mean_direction))
    centers = [values[first]]
    while len(centers) < count:
        distance = _cosine_distance(values, np.stack(centers)).min(axis=1)
        index = int(np.argmax(distance))
        centers.append(values[index])
    centers_array = np.stack(centers)
    for _ in range(iterations):
        assignment = _cosine_distance(values, centers_array).argmin(axis=1)
        updated = centers_array.copy()
        for cluster in range(count):
            members = values[assignment == cluster]
            if len(members):
                updated[cluster] = _normalize(members.mean(axis=0, keepdims=True))[0]
        if np.allclose(updated, centers_array, atol=1e-5):
            centers_array = updated
            break
        centers_array = updated
    residual = _cosine_distance(values, centers_array).min(axis=1)
    median = float(np.median(residual))
    scale = max(float(np.quantile(residual, 0.90) - median), 1e-4)
    return PrototypeGroup(
        centers=centers_array.astype(np.float32),
        residual_median=median,
        residual_scale=scale,
        samples=len(values),
    )


def prototype_anomaly_score(
    features: np.ndarray, group: PrototypeGroup
) -> np.ndarray:
    values = _normalize(np.asarray(features, dtype=np.float32))
    distance = _cosine_distance(values, group.centers).min(axis=1)
    return (distance - group.residual_median) / group.residual_scale


def fit_prototype_bank(
    examples: dict[tuple[int, int], list[np.ndarray]],
    *,
    prototypes: int = 3,
    minimum_step_samples: int = 20,
    maximum_samples: int = 4000,
    seed: int = 7,
) -> dict[tuple[int, int | None], PrototypeGroup]:
    rng = np.random.default_rng(seed)
    bank: dict[tuple[int, int | None], PrototypeGroup] = {}
    by_component: dict[int, list[np.ndarray]] = {}
    for (component, step), rows in examples.items():
        if not rows:
            continue
        values = np.asarray(rows, dtype=np.float32)
        by_component.setdefault(component, []).append(values)
        if len(values) >= minimum_step_samples:
            bank[(component, step)] = fit_prototype_group(
                _subsample(values, maximum_samples, rng),
                prototypes=prototypes,
                seed=seed + 1009 * component + step,
            )
    for component, chunks in by_component.items():
        values = np.concatenate(chunks, axis=0)
        bank[(component, None)] = fit_prototype_group(
            _subsample(values, maximum_samples, rng),
            prototypes=prototypes,
            seed=seed + 1009 * component,
        )
    return bank


def score_with_bank(
    bank: dict[tuple[int, int | None], PrototypeGroup],
    component: int,
    step: int,
    feature: np.ndarray,
) -> tuple[float, str]:
    key: tuple[int, int | None] = (component, step)
    source = "component_step"
    if key not in bank:
        key = (component, None)
        source = "component_fallback"
    if key not in bank:
        return float("nan"), "missing"
    score = prototype_anomaly_score(np.asarray(feature)[None], bank[key])[0]
    return float(score), source


def binary_average_precision(scores: Iterable[float], labels: Iterable[int]) -> float:
    score = np.asarray(list(scores), dtype=np.float64)
    target = np.asarray(list(labels), dtype=np.int64)
    positives = int((target == 1).sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score, kind="stable")
    ranked = target[order] == 1
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(100.0 * precision[ranked].sum() / positives)


def calibrate_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    duration_minutes: float,
    max_false_alerts_per_minute: float = 2.0,
) -> dict[str, float]:
    score = np.asarray(list(scores), dtype=np.float64)
    target = np.asarray(list(labels), dtype=np.int64)
    finite = np.isfinite(score)
    score, target = score[finite], target[finite]
    positives = max(int((target == 1).sum()), 1)
    thresholds = np.concatenate(([np.inf], np.unique(score)[::-1]))
    best = None
    for threshold in thresholds:
        prediction = score >= threshold
        true_positive = int((prediction & (target == 1)).sum())
        false_positive = int((prediction & (target == 0)).sum())
        recall = true_positive / positives
        precision = true_positive / max(int(prediction.sum()), 1)
        false_alert_rate = false_positive / max(duration_minutes, 1e-6)
        if false_alert_rate > max_false_alerts_per_minute + 1e-12:
            continue
        candidate = (recall, precision, -false_alert_rate, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        best = (0.0, 0.0, 0.0, float("inf"))
    return {
        "threshold": best[3],
        "recall": 100.0 * best[0],
        "precision": 100.0 * best[1],
        "false_alerts_per_minute": -best[2],
    }


def threshold_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
    threshold: float,
    *,
    duration_minutes: float,
) -> dict[str, float]:
    score = np.asarray(list(scores), dtype=np.float64)
    target = np.asarray(list(labels), dtype=np.int64)
    finite = np.isfinite(score)
    score, target = score[finite], target[finite]
    prediction = score >= threshold
    true_positive = int((prediction & (target == 1)).sum())
    false_positive = int((prediction & (target == 0)).sum())
    false_negative = int((~prediction & (target == 1)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * 2.0 * precision * recall / max(precision + recall, 1e-12),
        "false_alerts_per_minute": false_positive / max(duration_minutes, 1e-6),
        "true_positives": float(true_positive),
        "false_positives": float(false_positive),
        "false_negatives": float(false_negative),
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def _cosine_distance(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return 1.0 - _normalize(values) @ _normalize(centers).T


def _subsample(
    values: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[rng.choice(len(values), size=maximum, replace=False)]
