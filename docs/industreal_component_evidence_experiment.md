# IndustReal component-evidence pilot

## Why this experiment exists

The strict-causal baseline reached 16.73 F1@50 for action segmentation but produced zero
incorrect-event recall. Its normality average precision was 6.87% at 4.39% prevalence, showing
weak ranking signal that could not support an alert under the two-false-alerts-per-minute limit.
Increasing temporal-head capacity is therefore not the next useful intervention.

This pilot changes the event representation. It creates a feature for every procedure component
from three terms: causal event-temporal context, a learned component query, and a separately
projected sensor vector. The cached sensor vector contains the dataset's available hand, gaze, and
pose measurements. Shared component heads predict outcome, incorrect onset, and normality from
these component-conditioned features. The action expert still uses the original global fusion.

The loss also adds a cheap matched-bag ranking term. Within each batch and component, it pools
observed correct and incorrect state scores and enforces a margin between their means. This uses
dense component-state supervision, avoids quadratic frame pairs, and is robust to uncertain exact
onset timing.

## Controlled comparison

The run keeps the strict-causal cache, split, seed 7, Large model dimensions, optimizer, epoch
budgets, event-centered crops, dual-expert routing, and validation threshold calibration identical
to the baseline. The only enabled additions are `--component-evidence` and a ranking loss with
weight 0.25 and margin 0.5.

Launch:

```bash
PYTHON_BIN=/home/aiops/miniconda3/envs/psr_env/bin/python \
CACHE_INDEX=data/processed/industreal_strict_swin_convnext/index.json \
CACHE_CONTRACT=strict_causal SEED=7 \
bash scripts/train_industreal_component_evidence.sh all
```

Completion is indicated by:

```text
runs/industreal_strict_causal_evidence_s7_dual_expert/validation.json
```

The primary gate is non-zero incorrect-event recall at no more than two false alerts per minute.
Secondary comparisons are normality AP, incorrect-state F1, and action F1@50.

## Important limitation

This is a component-conditioned sensor-evidence pilot, not the final hand/object image-ROI system.
At launch time the original raw IndustReal RGB directory was no longer present on the Ubuntu
machine, while the verified causal feature cache remained available. True ROI embeddings require
restoring those frames. The code does not claim or simulate spatial crops from globally pooled
features.
