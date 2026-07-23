#!/usr/bin/env bash
set -euo pipefail

export RUN_PREFIX="${RUN_PREFIX:-industreal_strict_causal_roi_s${SEED:-7}}"
export COMPONENT_EVIDENCE=1
export COMPONENT_RANK_WEIGHT="${COMPONENT_RANK_WEIGHT:-0.25}"
export COMPONENT_RANK_MARGIN="${COMPONENT_RANK_MARGIN:-0.5}"
export EVENT_ONLY_MOTION_AUX=1

exec bash scripts/train_industreal_staged_baseline.sh "${1:-all}"
