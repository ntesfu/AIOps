#!/usr/bin/env bash
set -euo pipefail

export RUN_PREFIX="${RUN_PREFIX:-industreal_strict_causal_roi_factorized_s${SEED:-7}}"
export COMPONENT_EVIDENCE=1
export COMPONENT_RANK_WEIGHT="${COMPONENT_RANK_WEIGHT:-0.25}"
export COMPONENT_RANK_MARGIN="${COMPONENT_RANK_MARGIN:-0.5}"
export EVENT_ONLY_MOTION_AUX=1
export FACTORIZED_MISTAKE_DETECTION=1
# The cache audit contains 235 correct versus 14 incorrect training events
# (16.8:1), so use the measured event-level imbalance.
export ANY_MISTAKE_POS_WEIGHT="${ANY_MISTAKE_POS_WEIGHT:-17.0}"
export ANY_MISTAKE_WEIGHT="${ANY_MISTAKE_WEIGHT:-1.0}"
export MISTAKE_COMPONENT_WEIGHT="${MISTAKE_COMPONENT_WEIGHT:-1.0}"
export EVENT_SELECTION_STRATEGY="${EVENT_SELECTION_STRATEGY:-operational_harmonic}"

exec bash scripts/train_industreal_staged_baseline.sh "${1:-all}"
