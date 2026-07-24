"""Counterfactual error synthesis for StateVerify component supervision.

IndustReal supplies far too few real execution errors (~14-19 incorrect events in
training) to learn a discriminative fault detector directly.  This module
manufactures *hard-negative* incorrect installations from the abundant correct
installations already in the cache, so the observer's ``installed_incorrect`` /
``complete_incorrect`` heads and the matched-pair ranking loss are actually
exercised during training.

The transform is deliberately pure NumPy and operates on the window ``sample``
dicts produced by :meth:`aiops.data.stategraph_cache.StateGraphCacheDataset.__getitem__`.
It never imports PyTorch, so it is unit-testable without the ML stack and cheap
to run inside DataLoader workers.

Design
------
A synthetic incorrect event is built from a real *correct* completion event by:

1. flipping ``component_outcome`` at that ``[row, event_component]`` from
   ``correct`` (0) to ``incorrect`` (1);
2. relabelling the persistent ``state`` of the associated state component from
   the event row forward within the window: raw ``correct`` (2) becomes raw
   ``incorrect`` (0).  This yields typed state ``installed_incorrect`` and a
   typed ``complete_incorrect`` transition through the existing
   :func:`aiops.models.stateverify_effect.state_effect_targets` contract;
3. replacing the *active-object* ROI visual embedding in ``motion_aux`` around
   the event with a **wrong-part** donor embedding drawn from a different
   component's correct events.  Without this evidence perturbation the label
   flip would teach nothing, because the input would be identical to a correct
   install.

Every synthesized sample is flagged (``is_counterfactual`` / ``counterfactual_events``)
so downstream sampling can pair it with its source correct install and so it can
be verified never to leak into validation/test.

Leakage safety
--------------
:func:`assert_train_only` guards that donors and synthesis only ever touch
training records.  Validation/test records live in a separate dataset instance
and must never be passed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from aiops.data.stategraph_cache import StateGraphCacheRecord


# ROI order matches aiops.features.industreal_roi_features.ROI_NAMES:
# (left_hand=0, right_hand=1, active_object=2, interaction_context=3).
#
# Empirically on the IndustReal StateVerify cache the *active_object* detection
# is present only ~18% of frames and ~0.3% around completion events (the part is
# occluded by the hands exactly when it completes), while *interaction_context*
# — the crop spanning both hands and the manipulated part — is present ~100% of
# the time. The wrong-part evidence swap therefore defaults to the
# interaction-context ROI, which actually carries a learnable "which assembly is
# happening" signal on this cache.
DEFAULT_EVIDENCE_ROI_INDEX = 3


@dataclass(frozen=True)
class CounterfactualConfig:
    """Layout and label constants for counterfactual synthesis.

    ``roi_count`` and ``roi_dim`` describe the flat ``motion_aux`` visual block
    ``[roi_count * roi_dim]`` that precedes the presence masks and geometry.
    ``event_state_indices`` maps each completion (event) component to its
    persistent-state component, mirroring ``event_state_indices`` in the trainer.
    ``evidence_roi_index`` selects which ROI block is treated as the wrong-part
    evidence for donor collection and swapping (default: interaction_context).
    """

    roi_count: int = 4
    roi_dim: int = 768
    evidence_roi_index: int = DEFAULT_EVIDENCE_ROI_INDEX
    event_state_indices: tuple[int, ...] = ()
    correct_outcome: int = 0
    incorrect_outcome: int = 1
    remove_outcome: int = 2
    raw_state_incorrect: int = 0
    raw_state_pending: int = 1
    raw_state_correct: int = 2
    evidence_radius: int = 8

    def validate(self) -> None:
        if self.roi_count <= 0 or self.roi_dim <= 0:
            raise ValueError("roi_count and roi_dim must be positive.")
        if not 0 <= self.evidence_roi_index < self.roi_count:
            raise ValueError("evidence_roi_index is out of range.")
        if self.evidence_radius < 0:
            raise ValueError("evidence_radius cannot be negative.")
        if len(set(self.event_state_indices)) != len(self.event_state_indices):
            # Non-injective maps are allowed by the dataset, but they make the
            # persistent-state relabel ambiguous, so callers must be explicit.
            pass

    @property
    def visual_width(self) -> int:
        return self.roi_count * self.roi_dim

    def evidence_slice(self) -> slice:
        start = self.evidence_roi_index * self.roi_dim
        return slice(start, start + self.roi_dim)

    def evidence_present_index(self) -> int:
        """Index of the evidence ROI presence flag inside ``motion_aux``."""

        return self.visual_width + self.evidence_roi_index


def assert_train_only(records: Iterable[StateGraphCacheRecord]) -> None:
    """Raise if any record is not a training record.

    Counterfactuals are a training-time augmentation.  Synthesizing them from a
    validation or test record would corrupt the held-out measurement, so this
    guard is called before donor collection and synthesis.
    """

    offending = sorted(
        {record.split for record in records if record.split != "train"}
    )
    if offending:
        raise ValueError(
            "Counterfactual synthesis is train-only; received records with "
            f"splits {offending}. Keep validation/test in a separate dataset."
        )


def extract_evidence_visual(
    motion_aux: np.ndarray, config: CounterfactualConfig
) -> np.ndarray:
    """Return a copy of the ``[T, roi_dim]`` evidence-ROI embedding block."""

    if motion_aux.ndim != 2:
        raise ValueError("motion_aux must have shape [time, motion_aux_dim].")
    if motion_aux.shape[1] < config.visual_width:
        raise ValueError("motion_aux is narrower than the ROI visual block.")
    return motion_aux[:, config.evidence_slice()].copy()


def collect_wrong_part_donors(
    samples: Iterable[Mapping[str, np.ndarray]],
    config: CounterfactualConfig,
) -> dict[int, np.ndarray]:
    """Gather evidence-ROI embeddings observed at each component's correct events.

    The returned map keys are *completion (event) component* indices.  A donor
    embedding for a target component ``c`` is any embedding whose source
    component is not ``c`` (a genuinely wrong part), assembled lazily by
    :func:`donor_pool_excluding`.
    """

    config.validate()
    per_component: dict[int, list[np.ndarray]] = {}
    for sample in samples:
        motion_aux = sample.get("motion_aux")
        outcome = sample.get("component_outcome")
        present = None
        if motion_aux is not None:
            present_index = config.evidence_present_index()
            if motion_aux.shape[1] > present_index:
                present = motion_aux[:, present_index] > 0.5
        if motion_aux is None or outcome is None:
            continue
        visual = extract_evidence_visual(np.asarray(motion_aux), config)
        rows, components = np.where(np.asarray(outcome) == config.correct_outcome)
        for row, component in zip(rows.tolist(), components.tolist()):
            if present is not None and not bool(present[row]):
                continue
            per_component.setdefault(int(component), []).append(visual[row])
    return {
        component: np.stack(embeddings)
        for component, embeddings in per_component.items()
        if embeddings
    }


def donor_pool_excluding(
    donors: Mapping[int, np.ndarray], exclude_component: int
) -> np.ndarray:
    """Stack all donor embeddings whose source component is not ``exclude_component``."""

    pools = [
        embeddings
        for component, embeddings in donors.items()
        if component != exclude_component and len(embeddings)
    ]
    if not pools:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(pools, axis=0)


def synthesize_incorrect_event(
    sample: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    donors: Mapping[int, np.ndarray],
    config: CounterfactualConfig,
) -> dict[str, Any] | None:
    """Return a counterfactual copy of ``sample`` with one correct install turned incorrect.

    Returns ``None`` when the window has no eligible correct event with a wrong-part
    donor available, so the caller can fall back to the original sample.
    """

    config.validate()
    outcome = np.asarray(sample["component_outcome"])
    motion_aux = sample.get("motion_aux")
    if motion_aux is None:
        return None
    motion_aux = np.asarray(motion_aux)
    rows, components = np.where(outcome == config.correct_outcome)
    if rows.size == 0:
        return None

    order = rng.permutation(rows.size)
    for pick in order.tolist():
        row = int(rows[pick])
        event_component = int(components[pick])
        pool = donor_pool_excluding(donors, event_component)
        if pool.shape[0] == 0:
            continue
        return _apply_counterfactual(
            sample, row, event_component, pool, rng, config
        )
    return None


def _apply_counterfactual(
    sample: Mapping[str, np.ndarray],
    row: int,
    event_component: int,
    donor_pool: np.ndarray,
    rng: np.random.Generator,
    config: CounterfactualConfig,
) -> dict[str, Any]:
    new_sample: dict[str, Any] = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in sample.items()
    }
    outcome = new_sample["component_outcome"]
    outcome[row, event_component] = config.incorrect_outcome

    # Relabel the persistent state of the associated component from the event
    # row forward: a correct install (raw 2) becomes an incorrect install (raw 0).
    if event_component < len(config.event_state_indices):
        state_component = int(config.event_state_indices[event_component])
        state = new_sample.get("state")
        state_mask = new_sample.get("state_mask")
        if state is not None and 0 <= state_component < state.shape[1]:
            length = state.shape[0]
            for time in range(row, length):
                if state_mask is not None and not bool(state_mask[time, state_component]):
                    continue
                if int(state[time, state_component]) == config.raw_state_correct:
                    state[time, state_component] = config.raw_state_incorrect

    # Replace the evidence ROI visual around the event with a wrong part.
    motion_aux = new_sample["motion_aux"]
    donor = donor_pool[int(rng.integers(donor_pool.shape[0]))]
    start = max(0, row - config.evidence_radius)
    end = min(motion_aux.shape[0], row + 1)
    motion_aux[start:end, config.evidence_slice()] = donor

    events = list(new_sample.get("counterfactual_events", []))
    events.append((row, event_component))
    new_sample["counterfactual_events"] = events
    new_sample["is_counterfactual"] = True
    return new_sample


class CounterfactualWindowDataset:
    """Wrap a window dataset and emit counterfactual incorrect installs on the fly.

    With probability ``rate`` a window is replaced by a synthesized incorrect
    install (falling back to the original when no eligible event/donor exists).
    Donors are collected once from the *training* records only.  The wrapper is
    torch-free; the trainer's DataLoader consumes it exactly like the base
    dataset.
    """

    def __init__(
        self,
        dataset: Any,
        donors: Mapping[int, np.ndarray],
        config: CounterfactualConfig,
        rate: float = 0.5,
        seed: int = 7,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must lie in [0, 1].")
        config.validate()
        self.dataset = dataset
        self.donors = dict(donors)
        self.config = config
        self.rate = rate
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.dataset, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        rng = np.random.default_rng(
            self.seed + 7_919 * self.epoch + 1_299_827 * index
        )
        if self.rate <= 0.0 or rng.random() >= self.rate:
            return sample
        synthesized = synthesize_incorrect_event(sample, rng, self.donors, self.config)
        return synthesized if synthesized is not None else sample


def build_wrong_part_donors_from_records(
    records: Sequence[StateGraphCacheRecord], config: CounterfactualConfig
) -> dict[int, np.ndarray]:
    """Collect donors by reading each training record's cached arrays.

    This is the production entry point used by the trainer; unit tests exercise
    :func:`collect_wrong_part_donors` directly with in-memory samples.
    """

    assert_train_only(records)
    samples: list[dict[str, np.ndarray]] = []
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            if "motion_aux" not in arrays.files:
                continue
            samples.append(
                {
                    "motion_aux": arrays["motion_aux"].copy(),
                    "component_outcome": arrays["component_outcome"].copy(),
                }
            )
    return collect_wrong_part_donors(samples, config)
