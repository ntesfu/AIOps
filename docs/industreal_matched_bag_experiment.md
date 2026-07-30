# IndustReal matched component-bag experiment

## Decision and hypothesis

The grouped flat and learned-fusion experiments each produced an operational
checkpoint in only one of five operator-disjoint folds. Learned fusion moved the
successful fold from 3 to 4 but did not improve the pass rate. Four-fold
zero-recall behavior therefore remains a supervision problem, not evidence that
another fusion layer or a longer schedule is justified.

The existing event expert already uses two event-centered crops per incorrect
installation and a component-ranking loss. However, ordinary outcome-aware
batches guarantee incorrect windows without guaranteeing that a correct
installation of the same component is present in the same batch. The ranking
term is consequently inactive or poorly matched in many updates.

This experiment changes only event-stage batch construction:

1. sample the existing incorrect-event windows;
2. identify their labelled component and action;
3. pair each with a correct-install window for the same component;
4. prefer a correct event with the same action and then a different recording;
5. fill remaining batch slots with the unchanged sampling distribution.

Validation records are instantiated in a separate dataset and can never become
training candidates. The model, loss weights, action stage, operator folds,
optimizer, calibration, false-alert budget, and sealed-test policy remain
unchanged.

## Run

```bash
PYTHON_BIN=/home/aiops/miniconda3/envs/psr_env/bin/python \
FOLD_INDEX_DIR=experiments/industreal_grouped_folds_v2_s7 \
RUN_PREFIX_BASE=industreal_grouped_matchedbag_<commit> \
MATCHED_COMPONENT_BATCHES=1 \
EXPERIMENT_TIER=quick \
MINIMUM_FREE_GB=20 \
PRUNE_REDUNDANT_LAST_CHECKPOINTS=1 \
bash scripts/run_industreal_grouped_flat_reference.sh
```

The summary is generated with
`scripts/summarize_industreal_grouped_reference.py`.

## Promotion gate

Promote matched bags only if all conditions hold:

- operational checkpoints in at least two of five folds;
- positive strict incorrect-event recall in those folds;
- no more than two incorrect false alerts per minute;
- mean normality AP does not regress materially from the flat reference;
- action F1@50 is unchanged within deterministic training tolerance;
- peak VRAM remains below 23 GiB;
- the official test remains sealed.

If the pass rate remains 1/5, the next step is a true episode-level bag loss or
typed belief tracker evaluated from saved emissions—not more epochs on this
sampler.
