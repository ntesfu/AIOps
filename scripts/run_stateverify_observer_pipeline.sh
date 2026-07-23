#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/aiops/Desktop/ego_psr_repro/industreal}"
AUXILIARY_ROOT="${AUXILIARY_ROOT:-/home/aiops/AIOps-stategraph-industreal/data/processed/industreal_auxiliary}"
BASE_INDEX="${BASE_INDEX:-data/processed/industreal_strict_swin_convnext/index.json}"
SENSOR_ROOT="${SENSOR_ROOT:-data/processed/industreal_stateverify_sensor_v2}"
ROI_ROOT="${ROI_ROOT:-data/processed/industreal_stateverify_roi_v2}"
CACHE_ROOT="${CACHE_ROOT:-data/processed/industreal_stateverify_v2}"
RUN_ROOT="${RUN_ROOT:-runs/industreal_stateverify_observer_s${SEED:-7}}"

rebuild_sensors() {
  "$PYTHON" scripts/rebuild_stategraph_sensors.py \
    --cache-index "$BASE_INDEX" \
    --data-root "$DATA_ROOT" \
    --auxiliary-root "$AUXILIARY_ROOT" \
    --output-dir "$SENSOR_ROOT"
}

extract_roi() {
  "$PYTHON" -m aiops.features.industreal_roi_features \
    --cache-index "$SENSOR_ROOT/index.json" \
    --data-root "$DATA_ROOT" \
    --auxiliary-root "$AUXILIARY_ROOT" \
    --output-dir "$ROI_ROOT" \
    --batch-size "${ROI_BATCH_SIZE:-48}" \
    --interaction-context-scale "${INTERACTION_CONTEXT_SCALE:-1.4}" \
    --hand-box-scale "${HAND_BOX_SCALE:-1.5}" \
    --resume
}

attach_roi() {
  "$PYTHON" scripts/attach_stategraph_motion_aux.py \
    --cache-index "$SENSOR_ROOT/index.json" \
    --features-root "$ROI_ROOT" \
    --output-dir "$CACHE_ROOT"
}

smoke_train() {
  "$PYTHON" scripts/train_stateverify_observer.py \
    --cache-index "$CACHE_ROOT/index.json" \
    --output-dir "${RUN_ROOT}_smoke" \
    --epochs 1 \
    --batch-size 2 \
    --sequence-length 96 \
    --sequence-stride 96 \
    --hidden-dim 128 \
    --temporal-blocks 2 \
    --workers 0 \
    --max-train-batches 3
}

full_train() {
  "$PYTHON" scripts/train_stateverify_observer.py \
    --cache-index "$CACHE_ROOT/index.json" \
    --output-dir "$RUN_ROOT" \
    --epochs "${EPOCHS:-20}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --sequence-length "${SEQUENCE_LENGTH:-192}" \
    --sequence-stride "${SEQUENCE_STRIDE:-128}" \
    --hidden-dim "${HIDDEN_DIM:-256}" \
    --temporal-blocks "${TEMPORAL_BLOCKS:-6}" \
    --workers "${WORKERS:-2}" \
    --preload
}

case "$PHASE" in
  sensors) rebuild_sensors ;;
  roi) extract_roi ;;
  attach) attach_roi ;;
  smoke) smoke_train ;;
  train) full_train ;;
  all)
    rebuild_sensors
    extract_roi
    attach_roi
    smoke_train
    ;;
  *)
    echo "Usage: $0 {sensors|roi|attach|smoke|train|all}" >&2
    exit 2
    ;;
esac
