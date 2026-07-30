# Strict-causal IndustReal ROI evidence experiment

## Objective

This experiment addresses the principal failure of the global-feature baseline: it cannot reliably
see which hand, part, tool, or connection region caused a completion error. It retains the frozen
global Swin3D-S and ConvNeXt features and adds four current-frame ROI embeddings:

1. left-hand keypoint box;
2. right-hand keypoint box;
3. the active object selected by proximity to either hand;
4. an interaction-context crop spanning the hands and active object.

Every crop is encoded by a frozen ConvNeXt-Tiny model. Missing ROIs are represented by explicit
masks and zero embeddings. Normalized boxes, hand-object distance, detection confidence, and the selected object category
are appended to the ROI vector.

## Causal contract

Every ROI source frame is exactly the prediction frame. Each recording stores prediction and
source indices, and the merged cache is rejected by `--require-causal` if any auxiliary source is
in the future or if its provenance is absent. No centered temporal crop or interpolated feature is
used.

## Event isolation

The ROI vector is passed as `motion_aux`, but `--event-only-motion-aux` excludes it from the action
modality gate. A separate projection feeds ROI evidence only into the component-conditioned event
branch. This isolates fault-evidence gains from action segmentation and keeps the comparison with
the strict-causal baseline interpretable.

Mistake detection is factorized into a shared temporal decision and a conditional component
selector. The temporal detector max-pools the component-conditioned ROI tokens, so one strong
hand/object/connection anomaly can trigger an alert without ten independent component detectors
each paying the full class-imbalance penalty. Its positive weight is 17, derived from the audited
235 correct versus 14 incorrect training events. Event checkpoints use the operational harmonic
selector and must satisfy the two-false-alerts-per-minute budget.

The implementation includes optional causal ROI temporal blocks and hard-negative timing. The
seed-7 validation ablation rejected both: temporal blocks produced 20% recall only at roughly
12 false alerts/minute, and later collapsed; hard-negative timing did not produce an eligible
checkpoint. The selected configuration therefore uses frame-local ROI evidence with dense
asymmetric negatives.

## Separate checkpoint selectors

One checkpoint cannot optimize both dense action segmentation and five validation mistakes. The
reproducible workflow trains two copies from the same seed:

- the action expert uses `action_only` selection;
- the event expert uses `operational_harmonic` selection and is rejected unless incorrect recall
  is positive, false alerts are at most two/minute, and normality AP exceeds prevalence.

The experts are then composed and recalibrated with the identical validation protocol used during
training: sequence length 384, stride 192, batch size 4, BF16.

## Seed-7 validation result

| Metric | Result |
|---|---:|
| Action F1@50 | 17.13 |
| Incorrect-event recall | 20.0% (1/5 events) |
| Incorrect-event precision | 1.30% |
| Incorrect false alerts/minute | 1.20 |
| Incorrect normality AP | 8.15% |

This passes the predefined development gate and slightly exceeds the 16.73 F1@50 baseline. It is
not a final performance claim: only five incorrect events exist in validation, so uncertainty is
large and the sealed test split remains untouched.

## Storage strategy

The 29 GB recording archives are not expanded. Only `hands.csv`, `pose.csv`, and
`OD_labels.json` are selectively extracted to a small overlay. Video frames are decoded from the
86 root-level MP4 files. This avoids duplicating the 63 GB release on a nearly full workstation.

## Reproduction

```bash
PYTHON_BIN=/home/aiops/miniconda3/envs/psr_env/bin/python \
bash scripts/run_industreal_roi_experiment.sh all
```

The final validation report is:

```text
runs/industreal_roi_dual_selection_s7_dual_expert/validation.json
```

## Next controlled architecture experiment: structured ROI change tokens

The original ROI branch projects the complete 3,095-dimensional vector into one
frame-level feature. That is inexpensive, but it erases the distinction between
left hand, right hand, active object, and interaction context before the component detector can
reason about them. The opt-in structured branch keeps the same frozen cache and
decodes its existing v1 layout:

- four 768-dimensional ROI visual tokens;
- four presence flags and four normalized boxes;
- active-object category, confidence, and normalized hand-object distance;
- learned ROI-type embeddings;
- explicit current-minus-previous and current-minus-lagged changes;
- per-component queries attending only to present ROI tokens.

All changes are past-only. The auxiliary path remains event-only, so enabling it
cannot alter action logits through the forward graph. It is disabled by default
for checkpoint compatibility.

Run the paired seed-7 screen with identical cache, optimizer, windows, and
checkpoint selectors:

```bash
export CACHE_INDEX=data/processed/industreal_strict_swin_convnext_roi/index.json
export PYTHON_BIN=/home/aiops/miniconda3/envs/psr_env/bin/python

EXPERIMENT_TIER=quick bash scripts/run_industreal_structured_roi_experiment.sh flat all
EXPERIMENT_TIER=quick bash scripts/run_industreal_structured_roi_experiment.sh structured all
```

Or run the two variants sequentially with one command:

```bash
EXPERIMENT_TIER=quick bash scripts/run_industreal_structured_roi_pair.sh
```

`EXPERIMENT_TIER=smoke` is only an end-to-end execution check. It uses two
epochs per stage and a permissive selector so it can exercise checkpoint
composition; its metrics must never be used to promote an architecture.

The structured branch is promoted to seeds 7, 17, and 29 only if it:

1. preserves action F1@50 within 1 point of the paired flat branch;
2. has positive incorrect-event recall;
3. does not exceed 2 incorrect false alerts per minute;
4. improves either incorrect-event AP/F1 or matched-episode separation;
5. remains below 23 GiB peak training VRAM.

Use the operator-grouped manifest for the promotion runs:

```bash
PYTHONPATH=src "$PYTHON_BIN" scripts/prepare_industreal_grouped_evaluation.py \
  --cache-index "$CACHE_INDEX" \
  --output experiments/industreal_grouped_eval_s7.json \
  --fold-index-dir experiments/industreal_grouped_folds_s7 \
  --folds 5 --seed 7 --episode-radius 32 --matches-per-incorrect 3
```

This grouped evaluation pools official train and validation data only. Operators
are disjoint across folds, matching stays inside each fold partition, causal
episodes end at the labelled event row, and official test arrays are never
opened. The current cache yields 52 development recordings, 17 operators, and
19 incorrect events; each incorrect event has three same-component correct
matches. This is a more defensible rare-event screen than repeatedly tuning on
the five mistakes in the official validation split.

Train the flat reference across all operator-disjoint folds before comparing
another architecture:

```bash
FOLD_INDEX_DIR=experiments/industreal_grouped_folds_s7 \
RUN_PREFIX_BASE=industreal_grouped_flat_v1 \
EXPERIMENT_TIER=quick \
bash scripts/run_industreal_grouped_flat_reference.sh

PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_industreal_grouped_reference.py \
  --run-prefix-base industreal_grouped_flat_v1 \
  --output experiments/industreal_grouped_flat_v1_summary.json
```

The runner attempts every fold even when one fold cannot produce an operational
checkpoint. The summary reports that as a failure rather than averaging only
successful folds silently.

If folds show useful normality AP but no operational onset recall, run the
pre-existing learned event-fusion ablation. It combines onset, normality,
outcome, state, and action-mistake scores before calibration:

```bash
FOLD_INDEX_DIR=experiments/industreal_grouped_folds_v2_s7 \
RUN_PREFIX_BASE=industreal_grouped_fusion_v1 \
LEARNED_EVENT_FUSION=1 \
EXPERIMENT_TIER=quick \
bash scripts/run_industreal_grouped_flat_reference.sh
```

Interrupted grouped evaluations can resume without overwriting completed folds:

```bash
FOLD_START=1 \
FOLD_END=5 \
MINIMUM_FREE_GB=2 \
PRUNE_REDUNDANT_LAST_CHECKPOINTS=1 \
bash scripts/run_industreal_grouped_flat_reference.sh
```

`FOLD_END` is exclusive. The optional disk-space guard stops before beginning a
fold when available storage falls below the configured floor. Checkpoint
pruning removes only `last_checkpoint.pt` when the same run already has a
`best_checkpoint.pt`; best checkpoints, exported artifacts, metrics, and
TensorBoard histories are retained.

Promotion requires more than the flat reference's one operational fold out of
five while retaining the two-false-alerts-per-minute budget.

After learned fusion also achieved only one operational fold out of five, the
next controlled supervision experiment became same-component matched batching.
Its rationale, leakage controls, command, and promotion gate are specified in
`docs/industreal_matched_bag_experiment.md`.

## Hybrid recovery experiment

The structured-only quick screen preserved action segmentation within one
point, but event refinement produced no checkpoint with positive incorrect
recall under the two-false-alerts-per-minute budget. The next candidate retains
the proven flat ROI projection and adds the structured component-attention
feature through a bounded `tanh` gate initialized to exactly zero:

```text
component evidence = flat ROI + tanh(gate) * structured ROI change
```

This makes the initial evidence path identical to the flat branch and lets
training adopt or reject structured evidence independently for each component.
Run it against the already completed paired flat seed-7 baseline:

```bash
EXPERIMENT_TIER=quick \
RUN_PREFIX=industreal_roi_hybrid_quick_s7 \
bash scripts/run_industreal_structured_roi_experiment.sh hybrid all
```

TensorBoard records the mean, absolute mean, and per-component residual gates
under `optimizer/structured_roi_gate*`. A gate that remains near zero means the
structured evidence is not useful under the current supervision; a large gate
without improved operational metrics is evidence of overfitting.

If a cold-start hybrid fails while the flat branch succeeds, control for rare
event initialization by extending the promoted flat action checkpoint. The
extension copies every existing tensor exactly, adds only the structured ROI
modules, and verifies that the zero gate preserves flat action and event logits:

```bash
SOURCE_ACTION_CHECKPOINT=artifacts/industreal_roi_pair_2ac65fe_flat_quick_s7_action.pt \
RUN_PREFIX=industreal_roi_hybrid_warmstart_s7 \
EXPERIMENT_TIER=quick \
bash scripts/run_industreal_hybrid_warmstart.sh
```

This stage trains only the event branch. It is the appropriate controlled test
after a cold-start result because the action expert and initial mistake signal
are inherited from the successful flat checkpoint rather than redrawn.

The separate frozen-backbone experiment is specified in
`docs/industreal_backbone_ablation_protocol.md`. It compares Swin3D-S with a
genuine VideoMAE V2-B strict-causal cache; it must not relabel the extractor's
default Giant checkpoint as Base.

The primary gate is positive incorrect-event recall at no more than two false alerts per minute.
Normality AP, incorrect-state F1, detection delay, and action F1@50 are secondary metrics.
