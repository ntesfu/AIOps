#!/usr/bin/env bash
set -euo pipefail

variant="${1:-}"
stage="${2:-all}"
case "$variant" in
  flat)
    export STRUCTURED_ROI_TOKENS=0
    export HYBRID_ROI_RESIDUAL=0
    ;;
  structured)
    export STRUCTURED_ROI_TOKENS=1
    export HYBRID_ROI_RESIDUAL=0
    export ROI_TOKEN_COUNT="${ROI_TOKEN_COUNT:-4}"
    export ROI_EMBEDDING_DIM="${ROI_EMBEDDING_DIM:-768}"
    export ROI_GLOBAL_DIM="${ROI_GLOBAL_DIM:-3}"
    export ROI_CHANGE_LAG="${ROI_CHANGE_LAG:-4}"
    ;;
  hybrid)
    export STRUCTURED_ROI_TOKENS=1
    export HYBRID_ROI_RESIDUAL=1
    export ROI_TOKEN_COUNT="${ROI_TOKEN_COUNT:-4}"
    export ROI_EMBEDDING_DIM="${ROI_EMBEDDING_DIM:-768}"
    export ROI_GLOBAL_DIM="${ROI_GLOBAL_DIM:-3}"
    export ROI_CHANGE_LAG="${ROI_CHANGE_LAG:-4}"
    ;;
  *)
    echo "Usage: $0 {flat|structured|hybrid} {action|event|compose|all}" >&2
    exit 2
    ;;
esac

: "${EXPERIMENT_TIER:=quick}"
case "$EXPERIMENT_TIER" in
  smoke)
    export ACTION_EPOCHS="${ACTION_EPOCHS:-2}"
    export ACTION_PATIENCE="${ACTION_PATIENCE:-2}"
    export EVENT_EPOCHS="${EVENT_EPOCHS:-2}"
    export EVENT_PATIENCE="${EVENT_PATIENCE:-2}"
    # Smoke verifies the complete training/composition path. Two epochs are
    # intentionally too short to demand the operational promotion gate.
    export EVENT_SELECTION_STRATEGY="${EVENT_SELECTION_STRATEGY:-legacy}"
    ;;
  quick)
    export ACTION_EPOCHS="${ACTION_EPOCHS:-12}"
    export ACTION_PATIENCE="${ACTION_PATIENCE:-12}"
    export EVENT_EPOCHS="${EVENT_EPOCHS:-8}"
    export EVENT_PATIENCE="${EVENT_PATIENCE:-8}"
    export EVENT_SELECTION_STRATEGY="${EVENT_SELECTION_STRATEGY:-operational_harmonic}"
    ;;
  promotion)
    export ACTION_EPOCHS="${ACTION_EPOCHS:-30}"
    export ACTION_PATIENCE="${ACTION_PATIENCE:-12}"
    export EVENT_EPOCHS="${EVENT_EPOCHS:-16}"
    export EVENT_PATIENCE="${EVENT_PATIENCE:-8}"
    export EVENT_SELECTION_STRATEGY="${EVENT_SELECTION_STRATEGY:-operational_harmonic}"
    ;;
  *)
    echo "EXPERIMENT_TIER must be smoke, quick, or promotion" >&2
    exit 2
    ;;
esac

export RUN_PREFIX="${RUN_PREFIX:-industreal_roi_${variant}_${EXPERIMENT_TIER}_s${SEED:-7}}"
exec bash scripts/train_industreal_roi_evidence.sh "$stage"
