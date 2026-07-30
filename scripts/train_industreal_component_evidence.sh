#!/usr/bin/env bash
set -euo pipefail

# Matched comparison against industreal_strict_causal_s7. The action/event
# budgets, cache, split, and seed stay fixed; only the event evidence path and
# its same-component ranking objective are enabled.
export RUN_PREFIX="${RUN_PREFIX:-industreal_strict_causal_evidence_s${SEED:-7}}"
export COMPONENT_EVIDENCE=1
export COMPONENT_RANK_WEIGHT="${COMPONENT_RANK_WEIGHT:-0.25}"
export COMPONENT_RANK_MARGIN="${COMPONENT_RANK_MARGIN:-0.5}"

exec bash scripts/train_industreal_staged_baseline.sh "${1:-all}"
