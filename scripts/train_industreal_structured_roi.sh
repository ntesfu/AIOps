#!/usr/bin/env bash
set -euo pipefail

# Structured ROI v2 contract produced by industreal_roi_features.py:
# left/right hand, active object, and interaction-context embeddings; four
# presence flags, four boxes, category, object confidence, and hand distance.
export RUN_PREFIX="${RUN_PREFIX:-industreal_structured_roi_s${SEED:-7}}"
export STRUCTURED_ROI_TOKENS=1
export ROI_TOKEN_COUNT="${ROI_TOKEN_COUNT:-4}"
export ROI_EMBEDDING_DIM="${ROI_EMBEDDING_DIM:-768}"
export ROI_GLOBAL_DIM="${ROI_GLOBAL_DIM:-3}"
export ROI_CHANGE_LAG="${ROI_CHANGE_LAG:-4}"

exec bash scripts/train_industreal_roi_evidence.sh "${1:-all}"
