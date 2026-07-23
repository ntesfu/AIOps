#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
: "${CACHE_INDEX:?Set CACHE_INDEX to an IndustReal StateGraph cache index.json}"

cache_contract="${CACHE_CONTRACT:-strict_causal}"
case "$cache_contract" in
  strict_causal|legacy_offline) ;;
  *)
    echo "CACHE_CONTRACT must be strict_causal or legacy_offline" >&2
    exit 2
    ;;
esac

seed="${SEED:-7}"
run_prefix="${RUN_PREFIX:-industreal_${cache_contract}_s${seed}}"
action_epochs="${ACTION_EPOCHS:-25}"
event_epochs="${EVENT_EPOCHS:-16}"
action_patience="${ACTION_PATIENCE:-10}"
event_patience="${EVENT_PATIENCE:-8}"
event_selection="${EVENT_SELECTION_STRATEGY:-legacy}"
max_false_alerts="${MAX_INCORRECT_FALSE_ALERTS_PER_MINUTE:-2.0}"
component_evidence="${COMPONENT_EVIDENCE:-0}"
component_rank_weight="${COMPONENT_RANK_WEIGHT:-0.0}"
component_rank_margin="${COMPONENT_RANK_MARGIN:-0.5}"
event_only_motion_aux="${EVENT_ONLY_MOTION_AUX:-0}"
factorized_mistake_detection="${FACTORIZED_MISTAKE_DETECTION:-0}"
any_mistake_weight="${ANY_MISTAKE_WEIGHT:-1.0}"
mistake_component_weight="${MISTAKE_COMPONENT_WEIGHT:-1.0}"
any_mistake_pos_weight="${ANY_MISTAKE_POS_WEIGHT:-1.0}"
incorrect_hard_negative_ratio="${INCORRECT_HARD_NEGATIVE_RATIO:-0.0}"
component_evidence_temporal_blocks="${COMPONENT_EVIDENCE_TEMPORAL_BLOCKS:-0}"
structured_roi_tokens="${STRUCTURED_ROI_TOKENS:-0}"
roi_token_count="${ROI_TOKEN_COUNT:-4}"
roi_embedding_dim="${ROI_EMBEDDING_DIM:-0}"
roi_global_dim="${ROI_GLOBAL_DIM:-3}"
roi_change_lag="${ROI_CHANGE_LAG:-4}"
hybrid_roi_residual="${HYBRID_ROI_RESIDUAL:-0}"
action_selection="${ACTION_SELECTION_STRATEGY:-action_only}"
resume="${RESUME:-0}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH=".deps:src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is not available: $python_bin (set PYTHON_BIN)" >&2
  exit 2
fi

if [[ "$cache_contract" == "strict_causal" ]]; then
  echo "Verifying strict-causal cache provenance: $CACHE_INDEX"
  "$python_bin" -m aiops.data.audit_stategraph_cache \
    --cache-index "$CACHE_INDEX" --require-causal >/dev/null
fi

action_run="runs/${run_prefix}_action"
event_run="runs/${run_prefix}_event"
action_checkpoint="artifacts/${run_prefix}_action.pt"
event_checkpoint="artifacts/${run_prefix}_event.pt"
composite_checkpoint="artifacts/${run_prefix}_dual_expert.pt"
validation_report="runs/${run_prefix}_dual_expert/validation.json"
test_report="runs/${run_prefix}_dual_expert/test.json"

common_args=(
  --cache-index "$CACHE_INDEX"
  --precision bf16
  --max-vram-gib 23
  --batch-size 4
  --accumulation-steps 2
  --sequence-length 384
  --sequence-stride 288
  --hidden-dim 384
  --num-temporal-blocks 12
  --attention-every 2
  --num-heads 8
  --num-action-refinement-stages 2
  --num-refinement-blocks 4
  --num-event-blocks 4
  --dropout 0.15
  --graph-strength 0.08
  --weight-decay 0.002
  --rare-windows-per-batch 2
  --step-class-weight-power 1.0
  --state-class-weight-power 0.5
  --state-class-weight-cap 12.0
  --normality-error-weight 10.0
  --incorrect-pos-weight-cap 100.0
  --incorrect-hard-negative-ratio "$incorrect_hard_negative_ratio"
  --event-label-horizon 2
  --incorrect-selection-weight 1.0
  --procedural-event-context
  --calibration-interval 4
  --num-workers 4
  --preload-cache
  --seed "$seed"
)

if [[ "$component_evidence" == "1" ]]; then
  common_args+=(
    --component-evidence
    --component-rank-weight "$component_rank_weight"
    --component-rank-margin "$component_rank_margin"
    --component-evidence-temporal-blocks "$component_evidence_temporal_blocks"
  )
fi
if [[ "$structured_roi_tokens" == "1" ]]; then
  common_args+=(
    --structured-roi-tokens
    --roi-token-count "$roi_token_count"
    --roi-embedding-dim "$roi_embedding_dim"
    --roi-global-dim "$roi_global_dim"
    --roi-change-lag "$roi_change_lag"
  )
fi
if [[ "$hybrid_roi_residual" == "1" ]]; then
  common_args+=(--hybrid-roi-residual)
fi
if [[ "$event_only_motion_aux" == "1" ]]; then
  common_args+=(--event-only-motion-aux)
fi
if [[ "$factorized_mistake_detection" == "1" ]]; then
  common_args+=(
    --factorized-mistake-detection
    --any-mistake-weight "$any_mistake_weight"
    --mistake-component-weight "$mistake_component_weight"
    --any-mistake-pos-weight "$any_mistake_pos_weight"
  )
fi

train_action() {
  local resume_args=()
  if [[ "$resume" == "1" && -f "$action_run/last_checkpoint.pt" ]]; then
    resume_args+=(--resume "$action_run/last_checkpoint.pt")
  fi
  "$python_bin" -m aiops.training.train_stategraph_psr \
    "${common_args[@]}" \
    --output-dir "$action_run" \
    --export-checkpoint "$action_checkpoint" \
    --epochs "$action_epochs" \
    --patience "$action_patience" \
    --learning-rate 0.0002 \
    --selection-strategy "$action_selection" \
    "${resume_args[@]}"
}

train_event() {
  if [[ ! -f "$action_checkpoint" ]]; then
    echo "Missing action checkpoint: $action_checkpoint" >&2
    exit 3
  fi
  local resume_args=()
  if [[ "$resume" == "1" && -f "$event_run/last_checkpoint.pt" ]]; then
    resume_args+=(--resume "$event_run/last_checkpoint.pt")
  fi
  "$python_bin" -m aiops.training.train_stategraph_psr \
    "${common_args[@]}" \
    --output-dir "$event_run" \
    --export-checkpoint "$event_checkpoint" \
    --epochs "$event_epochs" \
    --patience "$event_patience" \
    --learning-rate 0.00005 \
    --init-checkpoint "$action_checkpoint" \
    --freeze-action-backbone \
    --event-centered-crops-per-event 2 \
    --event-centered-crop-radius 96 \
    --selection-strategy "$event_selection" \
    --selection-max-false-alerts-per-minute "$max_false_alerts" \
    --max-incorrect-false-alerts-per-minute "$max_false_alerts" \
    "${resume_args[@]}"
}

compose_experts() {
  if [[ ! -f "$action_checkpoint" || ! -f "$event_checkpoint" ]]; then
    echo "Both action and event checkpoints are required before composition" >&2
    exit 3
  fi
  "$python_bin" scripts/compose_stategraph_experts.py \
    --action-checkpoint "$action_checkpoint" \
    --event-checkpoint "$event_checkpoint" \
    --cache-index "$CACHE_INDEX" \
    --output-checkpoint "$composite_checkpoint" \
    --output-validation "$validation_report" \
    --precision bf16 \
    --batch-size 4 \
    --sequence-length 384 \
    --sequence-stride 192 \
    --num-workers 4 \
    --max-incorrect-false-alerts-per-minute "$max_false_alerts"
}

evaluate_test() {
  if [[ "${ALLOW_TEST_EVALUATION:-0}" != "1" ]]; then
    echo "Refusing sealed-test evaluation. Set ALLOW_TEST_EVALUATION=1 after the development decision is frozen." >&2
    exit 4
  fi
  if [[ ! -f "$composite_checkpoint" ]]; then
    echo "Missing composed checkpoint: $composite_checkpoint" >&2
    exit 3
  fi
  "$python_bin" -m aiops.evaluation.evaluate_stategraph_checkpoint \
    --checkpoint "$composite_checkpoint" \
    --cache-index "$CACHE_INDEX" \
    --split test \
    --output "$test_report" \
    --precision bf16 \
    --batch-size 2 \
    --sequence-length 384 \
    --num-workers 4
}

case "$stage" in
  action) train_action ;;
  event) train_event ;;
  compose) compose_experts ;;
  all)
    train_action
    train_event
    compose_experts
    ;;
  test) evaluate_test ;;
  *)
    echo "Usage: $0 {action|event|compose|all|test}" >&2
    exit 2
    ;;
esac
