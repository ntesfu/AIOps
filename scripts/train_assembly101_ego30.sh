#!/usr/bin/env bash
set -euo pipefail

variant="${1:-base}"
epochs="${EPOCHS:-16}"
run_name="${RUN_NAME:-ego30_${variant}_${epochs}ep}"
evaluate_test="${EVALUATE_TEST:-0}"
calibration_interval="${CALIBRATION_INTERVAL:-4}"
step_class_weight_power="${STEP_CLASS_WEIGHT_POWER:-1.0}"
focal_gamma="${FOCAL_GAMMA:-1.5}"
state_class_weight_power="${STATE_CLASS_WEIGHT_POWER:-0.5}"
state_class_weight_cap="${STATE_CLASS_WEIGHT_CAP:-12.0}"
incorrect_pos_weight_cap="${INCORRECT_POS_WEIGHT_CAP:-100.0}"
incorrect_selection_weight="${INCORRECT_SELECTION_WEIGHT:-1.0}"
asl_negative_gamma="${ASL_NEGATIVE_GAMMA:-4.0}"
asl_clip="${ASL_CLIP:-0.05}"
init_checkpoint="${INIT_CHECKPOINT:-}"
freeze_action_backbone="${FREEZE_ACTION_BACKBONE:-0}"
learning_rate_override="${LEARNING_RATE:-}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
procedural_event_context="${PROCEDURAL_EVENT_CONTEXT:-0}"
learned_event_fusion="${LEARNED_EVENT_FUSION:-0}"
event_centered_crops_per_event="${EVENT_CENTERED_CROPS_PER_EVENT:-0}"
event_centered_crop_radius="${EVENT_CENTERED_CROP_RADIUS:-0}"

case "$variant" in
  base)
    architecture=(--hidden-dim 256 --num-temporal-blocks 8 --batch-size 4 --accumulation-steps 2 --learning-rate 0.0002 --dropout 0.2)
    losses=(--incorrect-onset-weight 0.7 --state-weight 0.8 --normality-weight 0.3)
    ;;
  large)
    architecture=(--hidden-dim 384 --num-temporal-blocks 12 --batch-size 3 --accumulation-steps 3 --learning-rate 0.00015 --dropout 0.15)
    losses=(--incorrect-onset-weight 0.8 --state-weight 0.9 --normality-weight 0.4)
    ;;
  event_heavy)
    architecture=(--hidden-dim 384 --num-temporal-blocks 10 --batch-size 3 --accumulation-steps 3 --learning-rate 0.0002 --dropout 0.15 --num-event-blocks 6)
    losses=(--incorrect-onset-weight 1.2 --state-weight 1.0 --normality-weight 0.5)
    ;;
  *)
    echo "Unknown variant: $variant" >&2
    exit 2
    ;;
esac

if [[ -n "$learning_rate_override" ]]; then
  architecture+=(--learning-rate "$learning_rate_override")
fi

test_args=()
if [[ "$evaluate_test" == "1" ]]; then
  test_args+=(--evaluate-test)
fi

resume_args=()
if [[ -n "$resume_checkpoint" ]]; then
  resume_args+=(--resume "$resume_checkpoint")
fi

init_args=()
if [[ -n "$init_checkpoint" ]]; then
  init_args+=(--init-checkpoint "$init_checkpoint")
fi

procedure_args=()
if [[ "$procedural_event_context" == "1" ]]; then
  procedure_args+=(--procedural-event-context)
fi
if [[ "$learned_event_fusion" == "1" ]]; then
  procedure_args+=(--learned-event-fusion)
fi
procedure_args+=(
  --event-centered-crops-per-event "$event_centered_crops_per_event"
  --event-centered-crop-radius "$event_centered_crop_radius"
)
if [[ "$freeze_action_backbone" == "1" ]]; then
  init_args+=(--freeze-action-backbone)
fi

export PYTHONPATH=".deps:src${PYTHONPATH:+:${PYTHONPATH}}"
.venv/bin/python -m aiops.training.train_stategraph_psr \
  --cache-index data/processed/assembly101_ego30_stategraph/index.json \
  --output-dir "runs/${run_name}" \
  --export-checkpoint "artifacts/${run_name}.pt" \
  --epochs "$epochs" \
  --patience "$epochs" \
  --precision bf16 \
  --sequence-length 512 \
  --sequence-stride 384 \
  --step-class-weight-power "$step_class_weight_power" \
  --focal-gamma "$focal_gamma" \
  --state-class-weight-power "$state_class_weight_power" \
  --state-class-weight-cap "$state_class_weight_cap" \
  --attention-every 2 \
  --num-heads 8 \
  --num-action-refinement-stages 2 \
  --num-refinement-blocks 4 \
  --num-event-blocks 4 \
  --graph-strength 0.08 \
  --weight-decay 0.001 \
  --rare-window-boost 6.0 \
  --rare-windows-per-batch 2 \
  --normality-error-weight 10.0 \
  --incorrect-pos-weight-cap "$incorrect_pos_weight_cap" \
  --asl-negative-gamma "$asl_negative_gamma" \
  --asl-clip "$asl_clip" \
  --incorrect-selection-weight "$incorrect_selection_weight" \
  --calibration-interval "$calibration_interval" \
  --num-workers 4 \
  --preload-cache \
  --seed 7 \
  "${architecture[@]}" \
  "${losses[@]}" \
  "${resume_args[@]}" \
  "${init_args[@]}" \
  "${procedure_args[@]}" \
  "${test_args[@]}"
