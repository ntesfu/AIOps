#!/usr/bin/env bash
set -euo pipefail

: "${CACHE_INDEX:?Set CACHE_INDEX to the strict-causal ROI cache index.json}"
: "${SOURCE_ACTION_CHECKPOINT:?Set SOURCE_ACTION_CHECKPOINT to the promoted flat action checkpoint}"

export EXPERIMENT_TIER="${EXPERIMENT_TIER:-quick}"
export RUN_PREFIX="${RUN_PREFIX:-industreal_roi_hybrid_warmstart_s${SEED:-7}}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
hybrid_action_checkpoint="artifacts/${RUN_PREFIX}_action.pt"
export PYTHONPATH=".deps:src${PYTHONPATH:+:${PYTHONPATH}}"

"$python_bin" scripts/extend_stategraph_hybrid_roi_checkpoint.py \
  --source-checkpoint "$SOURCE_ACTION_CHECKPOINT" \
  --output-checkpoint "$hybrid_action_checkpoint" \
  --roi-token-count "${ROI_TOKEN_COUNT:-4}" \
  --roi-embedding-dim "${ROI_EMBEDDING_DIM:-768}" \
  --roi-global-dim "${ROI_GLOBAL_DIM:-3}" \
  --roi-change-lag "${ROI_CHANGE_LAG:-4}"

bash scripts/run_industreal_structured_roi_experiment.sh hybrid event
bash scripts/run_industreal_structured_roi_experiment.sh hybrid compose
