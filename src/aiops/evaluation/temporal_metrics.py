from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compact_labels(labels: Sequence[int], ignore_index: int = -100) -> list[int]:
    compact: list[int] = []
    for label in labels:
        value = int(label)
        if value == ignore_index:
            continue
        if not compact or compact[-1] != value:
            compact.append(value)
    return compact


def edit_score(prediction: Sequence[int], target: Sequence[int], ignore_index: int = -100) -> float:
    prediction, target = _mask_ignored_targets(prediction, target, ignore_index)
    pred = compact_labels(prediction, ignore_index)
    truth = compact_labels(target, ignore_index)
    if not pred and not truth:
        return 100.0
    if not pred or not truth:
        return 0.0
    previous = list(range(len(truth) + 1))
    for row, pred_label in enumerate(pred, start=1):
        current = [row]
        for column, truth_label in enumerate(truth, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (pred_label != truth_label),
                )
            )
        previous = current
    return 100.0 * (1.0 - previous[-1] / max(len(pred), len(truth)))


def segmental_f1(
    prediction: Sequence[int],
    target: Sequence[int],
    overlap: float,
    ignore_index: int = -100,
) -> float:
    prediction, target = _mask_ignored_targets(prediction, target, ignore_index)
    pred_segments = _segments(prediction, ignore_index)
    truth_segments = _segments(target, ignore_index)
    if not pred_segments and not truth_segments:
        return 100.0
    if not pred_segments or not truth_segments:
        return 0.0
    used = np.zeros(len(truth_segments), dtype=np.bool_)
    true_positive = 0
    false_positive = 0
    for pred_label, pred_start, pred_end in pred_segments:
        best_iou = 0.0
        best_index = -1
        for index, (truth_label, truth_start, truth_end) in enumerate(truth_segments):
            if used[index] or truth_label != pred_label:
                continue
            intersection = max(0, min(pred_end, truth_end) - max(pred_start, truth_start))
            union = max(pred_end, truth_end) - min(pred_start, truth_start)
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= overlap:
            used[best_index] = True
            true_positive += 1
        else:
            false_positive += 1
    false_negative = int((~used).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return 100.0 * (2.0 * precision * recall / max(precision + recall, 1e-12))


def _mask_ignored_targets(
    prediction: Sequence[int], target: Sequence[int], ignore_index: int
) -> tuple[list[int], list[int]]:
    if len(prediction) != len(target):
        raise ValueError("prediction and target must have the same length")
    paired = [(int(pred), int(truth)) for pred, truth in zip(prediction, target) if int(truth) != ignore_index]
    return [pred for pred, _ in paired], [truth for _, truth in paired]


def _segments(labels: Sequence[int], ignore_index: int) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    current_label = None
    start = 0
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label == ignore_index:
            if current_label is not None:
                result.append((current_label, start, index))
                current_label = None
            continue
        if current_label is None:
            current_label = label
            start = index
        elif label != current_label:
            result.append((current_label, start, index))
            current_label = label
            start = index
    if current_label is not None:
        result.append((current_label, start, len(labels)))
    return result
