#!/usr/bin/env bash
set -euo pipefail

: "${CACHE_INDEX:?Set CACHE_INDEX to the strict-causal ROI cache index.json}"

tier="${EXPERIMENT_TIER:-quick}"
seed="${SEED:-7}"
for variant in flat structured; do
  export RUN_PREFIX="${RUN_PREFIX_BASE:-industreal_roi_pair}_${variant}_${tier}_s${seed}"
  EXPERIMENT_TIER="$tier" \
  SEED="$seed" \
  bash scripts/run_industreal_structured_roi_experiment.sh "$variant" all
done
