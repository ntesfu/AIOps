"""Symbolic selection verifier: wrong-part / wrong-tool detection.

"Correct tools / correct parts" is checkable against the procedure *without any
rare error labels*.  For each believed step the procedure schema declares which
object categories (parts/tools) are expected.  When the observer's detected
active object is not in that set, the verifier raises a **selection residual**
routed to the step's component(s), which then feeds the tracker's ``procedure``
evidence channel.

This is the cheapest high-value fault signal in the plan: it needs no training,
generalizes to unseen recordings, and directly answers "wrong pin / wrong
bracket / wrong screwdriver".

Inputs are read from the same flat ``motion_aux`` contract the observer already
consumes (see :mod:`aiops.features.industreal_roi_features`):

``[roi_count * roi_dim visual | roi_count presence | 4*roi_count boxes | 3 extras]``

The 3 trailing extras are ``[active_object_category / 100, object_score,
hand_object_distance]``.  The verifier is otherwise vocabulary-agnostic: the
expected-category sets are authored in the procedure schema against the actual
IndustReal object-detection vocabulary on the remote workstation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SelectionVerifierConfig:
    roi_count: int = 4
    roi_dim: int = 768
    active_object_roi_index: int = 2
    num_state_components: int = 11
    event_state_indices: tuple[int, ...] = ()
    category_scale: float = 100.0
    # Residual emitted when the detected part is not the expected one.  A hard
    # 1.0 makes the symbolic mismatch dominate; lower it to soften.
    mismatch_residual: float = 1.0

    def validate(self) -> None:
        if self.roi_count <= 0 or self.roi_dim <= 0:
            raise ValueError("roi_count and roi_dim must be positive.")
        if not 0 <= self.active_object_roi_index < self.roi_count:
            raise ValueError("active_object_roi_index is out of range.")
        if self.num_state_components <= 0:
            raise ValueError("num_state_components must be positive.")
        if not 0.0 <= self.mismatch_residual <= 1.0:
            raise ValueError("mismatch_residual must lie in [0, 1].")

    @property
    def visual_width(self) -> int:
        return self.roi_count * self.roi_dim

    def active_object_present_index(self) -> int:
        return self.visual_width + self.active_object_roi_index

    def active_object_category_index(self) -> int:
        """Index of the normalized active-object category (third-from-last)."""

        return self.visual_width + self.roi_count + 4 * self.roi_count


@dataclass(frozen=True)
class StepExpectation:
    """Expected object categories and affected components for one step."""

    expected_categories: frozenset[int]
    components: tuple[int, ...]


def load_selection_schema(
    step_expectations: Mapping[str, Any] | str | Path,
) -> dict[int, StepExpectation]:
    """Parse a ``step_expectations`` mapping into per-step expectations.

    Accepts an already-parsed mapping or a path to a procedure-schema JSON.  Each
    entry maps a step id to ``{"expected_categories": [...], "components": [...]}``.
    A schema without expectations yields an empty map, which makes the verifier a
    safe no-op (all evidence masked) until it is authored on the workstation.
    """

    if isinstance(step_expectations, (str, Path)):
        payload = json.loads(Path(step_expectations).read_text(encoding="utf-8"))
        mapping = payload.get("step_expectations", {})
    else:
        mapping = step_expectations
    parsed: dict[int, StepExpectation] = {}
    for raw_step, entry in dict(mapping).items():
        step = int(raw_step)
        categories = frozenset(int(value) for value in entry.get("expected_categories", []))
        components = tuple(int(value) for value in entry.get("components", []))
        parsed[step] = StepExpectation(categories, components)
    return parsed


def active_object_categories_from_motion_aux(
    motion_aux: np.ndarray, config: SelectionVerifierConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(category_ids, present)`` from the flat ROI feature block.

    ``category_ids`` are integer category ids (``-1`` when no active object is
    present); ``present`` is the active-object presence flag.
    """

    config.validate()
    motion_aux = np.asarray(motion_aux)
    if motion_aux.ndim != 2:
        raise ValueError("motion_aux must have shape [time, motion_aux_dim].")
    category_index = config.active_object_category_index()
    present_index = config.active_object_present_index()
    if motion_aux.shape[1] <= category_index:
        raise ValueError("motion_aux is too narrow for the ROI feature contract.")
    present = motion_aux[:, present_index] > 0.5
    categories = np.rint(motion_aux[:, category_index] * config.category_scale).astype(
        np.int64
    )
    categories = np.where(present, categories, -1)
    return categories, present


def selection_residual(
    categories: np.ndarray,
    present: np.ndarray,
    steps: Sequence[int] | np.ndarray,
    schema: Mapping[int, StepExpectation],
    config: SelectionVerifierConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ``[T, num_state_components]`` selection residual and its mask.

    A residual of ``mismatch_residual`` is emitted on a step's state components
    when the detected active-object category is not among the step's expected
    categories.  Frames without a present object, without a step expectation, or
    on steps with no declared components are masked (neutral).
    """

    config.validate()
    categories = np.asarray(categories)
    present = np.asarray(present, dtype=bool)
    steps = np.asarray(steps)
    length = len(steps)
    if categories.shape[0] != length or present.shape[0] != length:
        raise ValueError("categories, present, and steps must share the time axis.")

    residual = np.zeros((length, config.num_state_components), dtype=np.float64)
    mask = np.zeros((length, config.num_state_components), dtype=bool)
    for time in range(length):
        if not present[time]:
            continue
        expectation = schema.get(int(steps[time]))
        if expectation is None or not expectation.components or not expectation.expected_categories:
            continue
        mismatch = int(categories[time]) not in expectation.expected_categories
        value = config.mismatch_residual if mismatch else 0.0
        for event_component in expectation.components:
            state_component = _state_component(event_component, config)
            if state_component is None:
                continue
            mask[time, state_component] = True
            residual[time, state_component] = value
    return residual, mask


def selection_residual_from_motion_aux(
    motion_aux: np.ndarray,
    steps: Sequence[int] | np.ndarray,
    schema: Mapping[int, StepExpectation],
    config: SelectionVerifierConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: recover categories from ``motion_aux`` then verify."""

    categories, present = active_object_categories_from_motion_aux(motion_aux, config)
    return selection_residual(categories, present, steps, schema, config)


def _state_component(event_component: int, config: SelectionVerifierConfig) -> int | None:
    if config.event_state_indices:
        if 0 <= event_component < len(config.event_state_indices):
            candidate = int(config.event_state_indices[event_component])
        else:
            return None
    else:
        candidate = int(event_component)
    if 0 <= candidate < config.num_state_components:
        return candidate
    return None
