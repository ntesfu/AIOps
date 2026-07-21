#!/usr/bin/env bash
set -euo pipefail

variant="${1:-base}"
epochs="${EPOCHS:-16}"
run_name="${RUN_NAME:-ego30_${variant}_${epochs}ep}"
evaluate_test="${EVALUATE_TEST:-0}"
calibration_interval="${CALIBRATION_INTERVAL:-4}"

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

test_args=()
if [[ "$evaluate_test" == "1" ]]; then
  test_args+=(--evaluate-test)
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
  --incorrect-pos-weight-cap 100.0 \
  --incorrect-selection-weight 1.0 \
  --calibration-interval "$calibration_interval" \
  --num-workers 4 \
  --preload-cache \
  --seed 7 \
  "${architecture[@]}" \
  "${losses[@]}" \
  "${test_args[@]}"
