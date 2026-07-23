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
runs/industreal_strict_causal_roi_s7_dual_expert/validation.json
```

The primary gate is positive incorrect-event recall at no more than two false alerts per minute.
Normality AP, incorrect-state F1, detection delay, and action F1@50 are secondary metrics.
