"""Track A — step/action recognition utilities.

The primary recognition target is **step-level** tracking (which procedural step we
are in), reported alongside the finer per-frame action timeline. These utilities are
backbone/model-agnostic and dependency-light (numpy only) so they are testable
locally without the GPU box:

- ``StepTaxonomy`` — the 11-class STEP + 4-class TYPE taxonomy derived from a
  ``ProcedureSchema`` (IndustReal: 10 parts + background; correct/incorrect/remove).
- ``step_scores`` / ``aggregate_and_score`` — step-level frame accuracy, Edit, and
  segmental F1@{10,25,50}; the "free day-one baseline" collapses a fine timeline to
  steps and scores it.
- ``viterbi_decode`` / ``build_transition_matrix`` — MAP decode over step posteriors
  with a legal-transition prior (temporal stickiness + a task-graph ``forbidden`` mask).
"""

from aiops.recognition.step_eval import (
    aggregate_and_score,
    frame_accuracy,
    map_via_lut,
    marginalize_to_steps,
    mean_scores,
    step_level_report,
    step_scores,
)
from aiops.recognition.step_taxonomy import (
    BACKGROUND_STEP,
    TYPE_NONE,
    StepTaxonomy,
    step_lut_from_component_indices,
)
from aiops.recognition.viterbi import build_transition_matrix, viterbi_decode

__all__ = [
    "StepTaxonomy",
    "step_lut_from_component_indices",
    "BACKGROUND_STEP",
    "TYPE_NONE",
    "frame_accuracy",
    "step_scores",
    "aggregate_and_score",
    "map_via_lut",
    "marginalize_to_steps",
    "step_level_report",
    "mean_scores",
    "build_transition_matrix",
    "viterbi_decode",
]
