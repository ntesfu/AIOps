#!/usr/bin/env bash
set -euo pipefail

: "${CACHE_INDEX:?Set CACHE_INDEX to the strict-causal ROI cache index.json}"

seed="${SEED:-7}"
run_prefix="${RUN_PREFIX:-industreal_roi_dual_selection_s${seed}}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
action_prefix="${run_prefix}_action_expert"
event_prefix="${run_prefix}_event_expert"
action_checkpoint="artifacts/${action_prefix}_action.pt"
event_checkpoint="artifacts/${event_prefix}_action.pt"
output_checkpoint="artifacts/${run_prefix}_dual_expert.pt"
output_validation="runs/${run_prefix}_dual_expert/validation.json"
export PYTHONPATH=".deps:src${PYTHONPATH:+:${PYTHONPATH}}"

env \
  RUN_PREFIX="$action_prefix" \
  ACTION_SELECTION_STRATEGY=action_only \
  bash scripts/train_industreal_roi_evidence.sh action

env \
  RUN_PREFIX="$event_prefix" \
  ACTION_SELECTION_STRATEGY=operational_harmonic \
  bash scripts/train_industreal_roi_evidence.sh action

"$python_bin" scripts/compose_stategraph_experts.py \
  --action-checkpoint "$action_checkpoint" \
  --event-checkpoint "$event_checkpoint" \
  --cache-index "$CACHE_INDEX" \
  --output-checkpoint "$output_checkpoint" \
  --output-validation "$output_validation" \
  --precision bf16 \
  --batch-size 4 \
  --sequence-length 384 \
  --sequence-stride 192 \
  --num-workers 4 \
  --max-incorrect-false-alerts-per-minute \
    "${MAX_INCORRECT_FALSE_ALERTS_PER_MINUTE:-2.0}"
