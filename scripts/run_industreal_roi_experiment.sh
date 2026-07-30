#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
data_root="${DATA_ROOT:-/home/aiops/Desktop/ego_psr_repro/industreal}"
archive_root="${ARCHIVE_ROOT:-${data_root}/recording_archives}"
base_cache_index="${BASE_CACHE_INDEX:-data/processed/industreal_strict_swin_convnext/index.json}"
auxiliary_root="${AUXILIARY_ROOT:-data/processed/industreal_auxiliary}"
roi_features_root="${ROI_FEATURES_ROOT:-data/processed/industreal_roi_convnext}"
roi_cache_root="${ROI_CACHE_ROOT:-data/processed/industreal_strict_swin_convnext_roi}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

extract_auxiliary() {
  "$python_bin" scripts/extract_industreal_auxiliary.py \
    --archives-dir "$archive_root" \
    --output-root "$auxiliary_root"
}

extract_roi() {
  "$python_bin" -m aiops.features.industreal_roi_features \
    --cache-index "$base_cache_index" \
    --data-root "$data_root" \
    --auxiliary-root "$auxiliary_root" \
    --output-dir "$roi_features_root" \
    --device cuda \
    --precision bf16 \
    --batch-size "${ROI_BATCH_SIZE:-32}" \
    --resume
}

attach_roi() {
  "$python_bin" scripts/attach_stategraph_motion_aux.py \
    --cache-index "$base_cache_index" \
    --features-root "$roi_features_root" \
    --output-dir "$roi_cache_root"
  "$python_bin" -m aiops.data.audit_stategraph_cache \
    --cache-index "$roi_cache_root/index.json" \
    --require-causal
}

train_roi() {
  env \
    PYTHON_BIN="$python_bin" \
    CACHE_INDEX="$roi_cache_root/index.json" \
    CACHE_CONTRACT=strict_causal \
    SEED="${SEED:-7}" \
    bash scripts/train_industreal_roi_dual_selection.sh
}

case "$stage" in
  auxiliary) extract_auxiliary ;;
  roi) extract_roi ;;
  attach) attach_roi ;;
  train) train_roi ;;
  prepare)
    extract_auxiliary
    extract_roi
    attach_roi
    ;;
  all)
    extract_auxiliary
    extract_roi
    attach_roi
    train_roi
    ;;
  *)
    echo "Usage: $0 {auxiliary|roi|attach|prepare|train|all}" >&2
    exit 2
    ;;
esac
