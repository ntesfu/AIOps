"""The STEP + TYPE taxonomy for step-level recognition (Track A).

IndustReal's PSR action space is **33 raw actions = 11 state indices x 3 outcomes**
(`procedure_info.json`; also encoded in `configs/procedure_schemas/industreal_v1.json`).
The primary recognition target collapses this to:

- **STEP** — 11 classes: ``0 = background`` + the 10 completion components in schema
  order. This is "which procedural step are we in", the level the mistake loop and
  task graph consume, and the level `psr_tas` reaches ~75% on (vs ~35% on the fine
  taxonomy).
- **TYPE** — 4 classes: ``0 = none`` + ``correct / incorrect / remove`` (the schema
  ``event_outcomes``), carried alongside for the error loop (REMOVE especially).

Each completion component owns a contiguous triple of raw ids ordered
``[correct, incorrect, remove]``; raw ids outside the completion map (the base
state) are background. For IndustReal this makes ``step == raw_id // 3`` and
``type == raw_id % 3`` — but we derive both from the schema structure rather than
hard-coding ``// 3`` so a different dataset's schema still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from aiops.data.procedure_schema import ProcedureSchema

BACKGROUND_STEP = 0
TYPE_NONE = 0
DEFAULT_IGNORE_INDEX = -100


@dataclass(frozen=True)
class StepTaxonomy:
    """Maps raw PSR action ids to STEP (11-class) and TYPE (4-class) labels."""

    step_names: tuple[str, ...]  # index 0..N: ("background", component_0, ...)
    type_names: tuple[str, ...]  # ("none", "correct", "incorrect", "remove")
    raw_step: Mapping[int, int]  # raw_id -> step index (1..N); unmapped => background
    raw_type: Mapping[int, int]  # raw_id -> type index (1..3); unmapped => none

    @property
    def num_steps(self) -> int:
        return len(self.step_names)

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    @property
    def max_raw_id(self) -> int:
        return max(self.raw_step) if self.raw_step else 0

    @classmethod
    def from_schema(cls, schema: ProcedureSchema) -> "StepTaxonomy":
        components = list(schema.completion_components)
        step_names = ("background", *components)
        type_names = ("none", *schema.event_outcomes)

        by_component: dict[str, list[int]] = {}
        for raw_str, component in schema.raw_completion_map.items():
            by_component.setdefault(component, []).append(int(raw_str))

        raw_step: dict[int, int] = {}
        raw_type: dict[int, int] = {}
        for component, raw_ids in by_component.items():
            step_index = components.index(component) + 1
            for outcome_pos, raw_id in enumerate(sorted(raw_ids)):
                raw_step[raw_id] = step_index
                # outcome_pos 0/1/2 -> type 1/2/3 (correct/incorrect/remove)
                raw_type[raw_id] = min(outcome_pos + 1, len(type_names) - 1)
        return cls(step_names, type_names, raw_step, raw_type)

    @classmethod
    def from_schema_path(cls, path: str) -> "StepTaxonomy":
        return cls.from_schema(ProcedureSchema.load(path))

    def step_of(self, raw_id: int) -> int:
        return int(self.raw_step.get(int(raw_id), BACKGROUND_STEP))

    def type_of(self, raw_id: int) -> int:
        return int(self.raw_type.get(int(raw_id), TYPE_NONE))

    def raw_to_step_lut(self) -> np.ndarray:
        """Lookup table ``lut[raw_id] -> step`` over ``0..max_raw_id`` (bg elsewhere)."""
        lut = np.zeros(self.max_raw_id + 1, dtype=np.int64)
        for raw_id, step in self.raw_step.items():
            lut[raw_id] = step
        return lut

    def steps_from_raw(
        self, raw_labels: Sequence[int], ignore_index: int = DEFAULT_IGNORE_INDEX
    ) -> np.ndarray:
        """Collapse a fine per-frame raw-action timeline to steps (ignore preserved)."""
        return self._map(raw_labels, self.step_of, ignore_index)

    def types_from_raw(
        self, raw_labels: Sequence[int], ignore_index: int = DEFAULT_IGNORE_INDEX
    ) -> np.ndarray:
        return self._map(raw_labels, self.type_of, ignore_index)

    @staticmethod
    def _map(labels: Sequence[int], fn, ignore_index: int) -> np.ndarray:
        out = np.empty(len(labels), dtype=np.int64)
        for i, raw in enumerate(labels):
            raw = int(raw)
            out[i] = ignore_index if raw == ignore_index else fn(raw)
        return out


def step_lut_from_component_indices(
    component_indices: Sequence[int], background_step: int = BACKGROUND_STEP
) -> np.ndarray:
    """Fine-action -> STEP lookup table from ``action_event_component_indices``.

    ``stategraph_psr`` factorizes fine actions as verb x object and stores, per fine
    action, the completion-component index its object maps to (``-1`` = not a
    completion / background). The STEP is that component index + 1 (so 1..N), or the
    background step for ``-1``. This is the authoritative fine->step aggregation the
    model config already carries; feed the model's fine predictions through it to get
    step-level predictions.
    """
    comp = np.asarray(component_indices, dtype=np.int64)
    return np.where(comp < 0, background_step, comp + 1).astype(np.int64)
