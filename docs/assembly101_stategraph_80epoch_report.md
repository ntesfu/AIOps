# Assembly101 ego-30 StateGraph-PSR: 80-epoch report

## Result

The selected model completed 80 epochs on the official Assembly101 egocentric
videos. Validation selected epoch 78. The experiment uses 203 official coarse
action classes, 59 component states, a real temporal video backbone, and an
actor-disjoint 73/15/10 recording split.

On the held-out test actors it obtains **9.71% frame accuracy**, **4.35 edit**,
**1.22 F1@50**, **43.58% state macro-F1**, and **33.26% mistake-normality AP**.
This is a material improvement over the 16-epoch selected architecture on the
same hard task for validation frame accuracy (11.13% to 14.97%), F1@50 (1.37 to
2.74), and state macro-F1 (39.78% to 44.15%).

Exact component-and-time mistake alerts are not solved: validation incorrect
event F1 is 0.68%, and held-out test incorrect-event F1 is 0%. The checkpoint
does produce useful actor-held-out mistake ranking (33.26% normality AP versus
23.30% incorrect prevalence among install events) and 1.11% incorrect-state
F1, but it is not suitable for deployment as a point-alert detector.

## Data and temporal sampling

- Official source: `cvml-nus/assembly101`, revision
  `bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`.
- 98 one-view `e1` egocentric MP4 recordings (5.79 GiB).
- Actor-disjoint split: 73 train, 15 validation, 10 test recordings.
- Raw video is 60 fps and is decoded on the official 30 fps annotation clock.
- Each causal Swin3D clip contains 32 genuine frames at 30 fps (1.07 seconds).
- Features are emitted every 8 annotation frames, or 3.75 observations/second.
- 166,682 cached rows: 128,295 train, 24,319 validation, 14,068 test.
- Sparse events: train `[630 correct, 140 incorrect, 62 remove]`, validation
  `[132, 35, 17]`, test `[79, 24, 10]`.
- Test actors and test labels were not used for architecture, threshold, or
  checkpoint selection. Validation thresholds were frozen before test.

## Selected architecture

The 34,804,103-parameter model uses:

- frozen Kinetics-400 Swin3D-S causal motion features (768 dimensions);
- frozen ConvNeXt-Tiny current-frame appearance features (768 dimensions);
- a 384-dimensional causal temporal trunk with 10 dilated blocks and attention
  every second block;
- two four-block action-refinement stages;
- a six-block event branch that retains modality disagreement;
- official verb/noun factorization across all 203 actions;
- component-aware fusion of official `attempt to ...` action probability into
  mistake-state and mistake-onset evidence;
- a differentiable first-order procedure graph;
- outcome-specific event NMS, allowing a failed attempt and nearby successful
  retry to coexist;
- validation-calibrated event and rare-state thresholds;
- epoch-dependent, deterministic temporal crop augmentation.

Square-root action weights were more stable than full inverse-frequency action
weights. Sparse incorrect onsets use asymmetric BCE and mistake-heavy windows
are guaranteed in every training batch. Checkpoint selection weights mistake
quality twice while penalizing false alerts.

## Metrics

| Metric | Validation (epoch 78) | Actor-held-out test |
|---|---:|---:|
| Frame accuracy | 14.97% | 9.71% |
| Edit | 5.96 | 4.35 |
| F1@10 | 5.64% | 3.77% |
| F1@25 | 4.32% | 2.40% |
| F1@50 | 2.74% | 1.22% |
| State accuracy | 66.73% | 68.57% |
| State macro-F1 | 44.15% | 43.58% |
| Incorrect-state F1 | 0.39% | 1.11% |
| Mistake-normality AP | 24.56% | 33.26% |
| Exact incorrect-event F1 | 0.68% | 0.00% |
| Incorrect-event F1 at ±4 s | 1.36% | 0.00% |
| False alerts/minute | 2.03 | 2.53 |

The older fixed-camera 1 fps experiment used only 87 derived action classes,
so its much larger raw accuracy is not an apples-to-apples baseline. This run
uses the complete 203-class official coarse taxonomy, genuine 30 fps temporal
clips, one egocentric view, and stricter actor isolation.

## Reproduction and artifact

```bash
EPOCHS=80 \
RUN_NAME=ego30_taxonomy_fusion_selected_80ep \
EVALUATE_TEST=1 \
CALIBRATION_INTERVAL=2 \
STEP_CLASS_WEIGHT_POWER=0.5 \
FOCAL_GAMMA=1.0 \
STATE_CLASS_WEIGHT_POWER=0.5 \
STATE_CLASS_WEIGHT_CAP=12 \
INCORRECT_POS_WEIGHT_CAP=500 \
INCORRECT_SELECTION_WEIGHT=2.0 \
bash scripts/train_assembly101_ego30.sh event_heavy
```

- Checkpoint: `artifacts/ego30_taxonomy_fusion_selected_80ep.pt`
- Size: approximately 67 MiB
- SHA-256: `ea1041054777e5ba4e410cb8284b56c34630c4dd98422b78bcea00a0205014b0`
- Selected epoch: 78
- Peak PyTorch reserved VRAM: 2.742 GiB (hard limit: 23 GiB)
- Full suite at publication: 52 tests passing

The BF16 inference artifact contains model weights, architecture configuration,
label metadata, transition matrix, validation-frozen thresholds, and both
validation and test metrics. Optimizer state and raw videos are excluded.

## Remaining limitation

Action and state generalization improve throughout the long schedule, but the
140 training mistake onsets are insufficient for component-exact point alerts
on unseen actors. The next high-value experiment is mistake-centered clip
pretraining or a larger mistake-bearing training subset, not simply a larger
temporal head. Report normality AP and state F1 alongside exact event F1; the
exact alert metric must not be hidden by the stronger ranking result.
