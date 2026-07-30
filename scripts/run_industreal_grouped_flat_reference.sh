#!/usr/bin/env bash
set -euo pipefail

: "${FOLD_INDEX_DIR:?Set FOLD_INDEX_DIR to the materialized grouped-fold directory}"

fold_count="${FOLD_COUNT:-5}"
fold_start="${FOLD_START:-0}"
fold_end="${FOLD_END:-$fold_count}"
run_prefix_base="${RUN_PREFIX_BASE:-industreal_grouped_flat}"
minimum_free_gb="${MINIMUM_FREE_GB:-2}"
prune_redundant_last="${PRUNE_REDUNDANT_LAST_CHECKPOINTS:-0}"

if ((fold_start < 0 || fold_end > fold_count || fold_start >= fold_end)); then
  echo \
    "Invalid fold range [${fold_start}, ${fold_end}); expected 0 <= start < end <= ${fold_count}" \
    >&2
  exit 2
fi

if ! [[ "$minimum_free_gb" =~ ^[0-9]+$ ]]; then
  echo "MINIMUM_FREE_GB must be a non-negative integer" >&2
  exit 2
fi

prune_last_checkpoint() {
  local run_dir="$1"
  if [[ -f "${run_dir}/best_checkpoint.pt" && -f "${run_dir}/last_checkpoint.pt" ]]; then
    rm -- "${run_dir}/last_checkpoint.pt"
    echo "Pruned redundant checkpoint: ${run_dir}/last_checkpoint.pt"
  fi
}

failures=0
attempted=0
for ((fold=fold_start; fold<fold_end; fold++)); do
  available_kb="$(df -Pk . | awk 'NR == 2 {print $4}')"
  minimum_free_kb=$((minimum_free_gb * 1024 * 1024))
  if ((available_kb < minimum_free_kb)); then
    echo \
      "Stopping before fold ${fold}: ${available_kb} KiB free is below the ${minimum_free_gb} GiB safety floor" \
      >&2
    failures=$((failures + fold_end - fold))
    break
  fi

  attempted=$((attempted + 1))
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

  if [[ "$prune_redundant_last" == "1" ]]; then
    run_prefix="${run_prefix_base}_fold${fold}_s${SEED:-7}"
    prune_last_checkpoint "runs/${run_prefix}_action"
    prune_last_checkpoint "runs/${run_prefix}_event"
  fi
done
selected=$((fold_end - fold_start))
echo \
  "Grouped flat reference finished: range=[${fold_start},${fold_end}) attempted=${attempted}/${selected} failures=${failures}/${selected}"
exit "$failures"
