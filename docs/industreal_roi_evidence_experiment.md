# Strict-causal IndustReal ROI evidence experiment

## Objective

This experiment addresses the principal failure of the global-feature baseline: it cannot reliably
see which hand, part, tool, or connection region caused a completion error. It retains the frozen
global Swin3D-S and ConvNeXt features and adds four current-frame ROI embeddings:

1. left-hand keypoint box;
2. right-hand keypoint box;
3. gaze-centered connection region;
4. the detected object nearest gaze or either hand.

Every crop is encoded by a frozen ConvNeXt-Tiny model. Missing ROIs are represented by explicit
masks and zero embeddings. Normalized boxes, gaze coordinates, and the selected object category
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

The 29 GB recording archives are not expanded. Only `hands.csv`, `gaze.csv`, `pose.csv`, and
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

The primary gate is positive incorrect-event recall at no more than two false alerts per minute.
Normality AP, incorrect-state F1, detection delay, and action F1@50 are secondary metrics.
