#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=".deps:src${PYTHONPATH:+:${PYTHONPATH}}"

.venv/bin/python -m aiops.training.train_stategraph_psr \
  --cache-index data/processed/assembly101_stategraph/index.json \
  --output-dir runs/stategraph_assembly101_80ep \
  --export-checkpoint artifacts/stategraph_assembly101_80ep_bf16.pt \
  --epochs 80 \
  --patience 100 \
  --precision bf16 \
  --batch-size 8 \
  --accumulation-steps 2 \
  --sequence-length 256 \
  --sequence-stride 192 \
  --hidden-dim 256 \
  --num-temporal-blocks 8 \
  --attention-every 2 \
  --num-heads 8 \
  --num-action-refinement-stages 1 \
  --num-refinement-blocks 4 \
  --num-event-blocks 4 \
  --dropout 0.2 \
  --graph-strength 0.12 \
  --learning-rate 0.0002 \
  --weight-decay 0.001 \
  --rare-window-boost 4.0 \
  --rare-windows-per-batch 2 \
  --normality-error-weight 8.0 \
  --incorrect-selection-weight 1.0 \
  --calibration-interval 5 \
  --seed 7 \
  --evaluate-test
