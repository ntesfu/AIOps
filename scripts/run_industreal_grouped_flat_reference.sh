#!/usr/bin/env bash
set -euo pipefail

: "${FOLD_INDEX_DIR:?Set FOLD_INDEX_DIR to the materialized grouped-fold directory}"

fold_count="${FOLD_COUNT:-5}"
run_prefix_base="${RUN_PREFIX_BASE:-industreal_grouped_flat}"
failures=0
for ((fold=0; fold<fold_count; fold++)); do
  fold_index="${FOLD_INDEX_DIR}/fold-${fold}/index.json"
  if [[ ! -f "$fold_index" ]]; then
    echo "Missing fold index: $fold_index" >&2
    failures=$((failures + 1))
    continue
  fi
  echo "Starting grouped flat reference fold ${fold}/${fold_count}"
  if ! env \
    CACHE_INDEX="$fold_index" \
    RUN_PREFIX="${run_prefix_base}_fold${fold}_s${SEED:-7}" \
    EXPERIMENT_TIER="${EXPERIMENT_TIER:-quick}" \
    bash scripts/run_industreal_structured_roi_experiment.sh flat all
  then
    echo "Fold ${fold} ended without an operational checkpoint" >&2
    failures=$((failures + 1))
  fi
done
echo "Grouped flat reference finished: failures=${failures}/${fold_count}"
exit "$failures"
