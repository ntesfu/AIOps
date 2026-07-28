"""Step-level recognition metrics + the fine->step aggregation baseline.

Step accuracy is the **primary** recognition metric; Edit and segmental F1@{10,25,50}
report segmentation quality at the step level. The Edit/F1 primitives are reused from
``aiops.evaluation.temporal_metrics`` (label-agnostic), so step scoring is just those
primitives applied to step-mapped sequences.

``aggregate_and_score`` is the "free day-one baseline": collapse a fine per-frame
timeline (predictions and ground truth) to steps via a ``StepTaxonomy`` and score —
the cheap step number to beat with a dedicated step head.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from aiops.evaluation.temporal_metrics import edit_score, segmental_f1
from aiops.recognition.step_taxonomy import (
    BACKGROUND_STEP,
    DEFAULT_IGNORE_INDEX,
)
from aiops.recognition.viterbi import (
    build_transition_matrix,
    viterbi_decode,
    viterbi_decode_fixed_lag,
)

DEFAULT_OVERLAPS = (0.10, 0.25, 0.50)


def map_via_lut(
    labels: Sequence[int],
    step_lut: Sequence[int],
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    background_step: int = BACKGROUND_STEP,
) -> np.ndarray:
    """Map a fine per-frame label sequence to steps through ``step_lut``.

    ``step_lut[fine_label] -> step``. ``ignore_index`` and any negative (invalid /
    padding) label are preserved as ``ignore_index``; a label past the LUT's range
    collapses to ``background_step`` (defensive)."""
    lut = np.asarray(step_lut, dtype=np.int64)
    out = np.empty(len(labels), dtype=np.int64)
    for i, raw in enumerate(labels):
        raw = int(raw)
        if raw == ignore_index or raw < 0:
            out[i] = ignore_index
        elif raw < lut.shape[0]:
            out[i] = lut[raw]
        else:
            out[i] = background_step
    return out


def marginalize_to_steps(
    fine_posteriors: np.ndarray, step_lut: Sequence[int], num_steps: int
) -> np.ndarray:
    """Sum fine-class posteriors into step-class posteriors.

    ``fine_posteriors`` is ``(T, F)``; returns ``(T, num_steps)`` where each step's
    mass is the sum of its constituent fine classes (via ``step_lut``)."""
    fine = np.asarray(fine_posteriors, dtype=np.float64)
    if fine.ndim != 2:
        raise ValueError("fine_posteriors must be (T, F)")
    lut = np.asarray(step_lut, dtype=np.int64)
    out = np.zeros((fine.shape[0], num_steps), dtype=np.float64)
    for f in range(fine.shape[1]):
        step = int(lut[f]) if f < lut.shape[0] else BACKGROUND_STEP
        if 0 <= step < num_steps:
            out[:, step] += fine[:, f]
    return out


def frame_accuracy(
    prediction: Sequence[int],
    target: Sequence[int],
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> float:
    """Percent of non-ignored frames where predicted step == target step."""
    pred = np.asarray(prediction)
    tgt = np.asarray(target)
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target must have the same shape")
    mask = tgt != ignore_index
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return 100.0 * float((pred[mask] == tgt[mask]).sum()) / total


def step_scores(
    prediction: Sequence[int],
    target: Sequence[int],
    overlaps: Sequence[float] = DEFAULT_OVERLAPS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, float]:
    """Step-level {frame_acc, edit, f1@k}. Inputs are per-frame step-id sequences."""
    scores = {
        "frame_acc": frame_accuracy(prediction, target, ignore_index),
        "edit": edit_score(prediction, target, ignore_index),
    }
    for overlap in overlaps:
        scores[f"f1@{int(round(overlap * 100))}"] = segmental_f1(
            prediction, target, overlap, ignore_index
        )
    return scores


def aggregate_and_score(
    fine_prediction: Sequence[int],
    fine_target: Sequence[int],
    step_lut: Sequence[int],
    overlaps: Sequence[float] = DEFAULT_OVERLAPS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, float]:
    """Day-one baseline: map fine prediction+target timelines to steps via a
    fine->step LUT, then score at the step level.

    Build ``step_lut`` from the model config with
    ``step_lut_from_component_indices(action_event_component_indices)``, or from the
    schema with ``StepTaxonomy.raw_to_step_lut()`` when fine labels are raw ids."""
    pred_steps = map_via_lut(fine_prediction, step_lut, ignore_index)
    target_steps = map_via_lut(fine_target, step_lut, ignore_index)
    return step_scores(pred_steps, target_steps, overlaps, ignore_index)


def mean_scores(per_sequence: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average score dicts across recordings/sequences (macro over sequences)."""
    if not per_sequence:
        return {}
    keys = per_sequence[0].keys()
    return {k: float(np.mean([float(s[k]) for s in per_sequence])) for k in keys}


def step_level_report(
    fine_prediction: Sequence[int],
    fine_target: Sequence[int],
    step_lut: Sequence[int],
    num_steps: int,
    *,
    fine_posteriors: np.ndarray | None = None,
    self_bias: float = 2.0,
    forbidden: np.ndarray | None = None,
    transition_penalty: np.ndarray | None = None,
    forward_only: bool = False,
    lag: int | None = None,
    include_causal: bool = False,
    overlaps: Sequence[float] = DEFAULT_OVERLAPS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, dict[str, float]]:
    """One recording's step-level scores under up to four decoders.

    Returns ``{"agg": {...}, "viterbi": {...}[, "causal": {...}, "online": {...}]}``:
    - **agg** — the free day-one baseline: argmax fine predictions collapsed to steps.
    - **viterbi** — MAP-decoded (full/offline) with a legal-transition prior. Uses the
      marginalized step posteriors when ``fine_posteriors`` is given; otherwise a
      one-hot emission built from the argmax steps. ``forbidden``/``forward_only``
      inject the task graph's legal transitions.
    - **causal** — present when ``include_causal`` is true: strict zero-lookahead
      decoding, reported separately from the primary near-online regime.
    - **online** — present only when ``lag`` is given: the same decode but fixed-lag
      (near-online), committing each frame using only a ``lag``-frame trailing
      lookahead.
    """
    pred_steps = map_via_lut(fine_prediction, step_lut, ignore_index)
    target_steps = map_via_lut(fine_target, step_lut, ignore_index)
    report = {"agg": step_scores(pred_steps, target_steps, overlaps, ignore_index)}

    if fine_posteriors is not None:
        step_post = marginalize_to_steps(fine_posteriors, step_lut, num_steps)
    else:
        step_post = np.full((len(pred_steps), num_steps), 1e-6, dtype=np.float64)
        valid = pred_steps != ignore_index
        rows = np.nonzero(valid)[0]
        cols = np.clip(pred_steps[valid], 0, num_steps - 1)
        step_post[rows, cols] = 1.0

    log_emissions = np.log(step_post + 1e-9)
    log_transition = build_transition_matrix(
        num_steps,
        self_bias=self_bias,
        forbidden=forbidden,
        transition_penalty=transition_penalty,
        forward_only=forward_only,
    )
    decoded = viterbi_decode(log_emissions, log_transition)
    report["viterbi"] = step_scores(decoded, target_steps, overlaps, ignore_index)
    if include_causal:
        causal = viterbi_decode_fixed_lag(log_emissions, log_transition, 0)
        report["causal"] = step_scores(causal, target_steps, overlaps, ignore_index)
    if lag is not None:
        online = viterbi_decode_fixed_lag(log_emissions, log_transition, lag)
        report["online"] = step_scores(online, target_steps, overlaps, ignore_index)
    return report
