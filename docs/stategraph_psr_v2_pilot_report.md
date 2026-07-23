# StateGraph-PSR cache-v2 training report

Date: 2026-07-20

## Decision

Do **not** start the 80-epoch experiment yet. Cache schema v2 and the training
pipeline passed structural, tensor, and small-subset learning checks, but the
25-epoch official-split pilot did not learn a useful incorrect-install detector.
An 80-epoch run would spend more compute on a decision rule and rare-event
sampling setup that the pilot has already shown to be inadequate.

This is an engineering gate result, not a claim about the final architecture's
accuracy. The experiment used the fallback cached Swin3D-S motion and
ConvNeXt-Tiny appearance features. It did not use the group's preferred
VideoMAEv2-giant SSv2 features.

## Reproducible environment

- Desktop: NVIDIA GeForce RTX 4090, 24 GB
- Checkout: `/home/aiops/AIOps-stategraph`
- Cache: `/home/aiops/AIOps/data/processed/stategraph_v2_swin_convnext`
- Output: `/home/aiops/AIOps/runs/stategraph_psr_v2_pilot25_seed7`
- Dataset split: 36 train recordings, 16 validation recordings, no test access
- Cache size: 23,414 rows; 15,801 train and 7,613 validation
- Train events: 235 correct, 14 incorrect, 66 remove
- Validation events: 109 correct, 5 incorrect, 34 remove
- Model: 3,332,484 trainable parameters
- Precision: BF16
- Seed: 7

The exact pilot command was:

```bash
PYTHONPATH=/home/aiops/AIOps-stategraph/src python3 -m aiops.training.train_stategraph_psr \
  --cache-index /home/aiops/AIOps/data/processed/stategraph_v2_swin_convnext/index.json \
  --output-dir /home/aiops/AIOps/runs/stategraph_psr_v2_pilot25_seed7 \
  --device cuda --precision bf16 \
  --batch-size 2 --accumulation-steps 4 \
  --sequence-length 256 --sequence-stride 192 \
  --epochs 25 --patience 10 --learning-rate 0.0003 \
  --num-workers 2 --seed 7
```

The run completed in 176 seconds. It created 88 training windows, including 17
rare-fault windows, and used the default rare-window sampling multiplier of 4.

## Cache and wiring gates

The strict cache audit passed with no warnings. All ten completion components
are represented. Two validation-only atomic actions,
`plug_small_screw_pin` and `pull_small_screw_pin`, are valid compositions of
verbs and an object seen during training; the factorized action head therefore
keeps them evaluable without leaking validation examples.

The corrected four-recording overfit run used recordings
`01_assy_0_1`, `01_assy_1_1`, `01_main_0_1`, and `02_assy_0_1`. Its total loss
fell from 12.13 to roughly 1--2, action F1@50 reached 46.81, and completion-event
recall reached 62.96. Incorrect-event recall reached 100 on this deliberately
small subset, although precision was only 2.90. This establishes that gradients,
masks, targets, and every task head are connected. It does not establish
generalization or calibration.

## Pilot results

The selected checkpoint was epoch 16, using the registered composite selection
score. Metrics below are percentages.

| Metric | Best checkpoint |
|---|---:|
| Frame accuracy | 21.35 |
| Edit | 30.52 |
| F1@10 | 23.72 |
| F1@25 | 17.28 |
| F1@50 | 9.70 |
| Completion-event precision | 2.67 |
| Completion-event recall | 15.20 |
| Completion-event F1 | 4.54 |
| Correct-event F1 | 4.78 |
| Incorrect-event F1 | 0.00 |
| Remove-event F1 | 4.80 |
| Component-state accuracy | 81.59 |
| Unseen-composition accuracy (19 frames) | 0.00 |

The total training loss fell from 19.32 at epoch 1 to 4.49 at epoch 25. At the
last epoch, frame accuracy was 19.46, F1@50 was 8.07, completion-event F1 was
5.25, and state accuracy was 83.32. Action F1@50 peaked at epoch 16, so simply
continuing the same schedule is not supported by the observed trajectory.

Incorrect-event F1 was zero at the selected checkpoint. The only non-zero pilot
value occurred at epoch 5 and was 0.17 F1 (0.09 precision, 16.67 recall). With
only 14 train and 5 validation incorrect events, this metric is intrinsically
high variance; it must be reported over multiple seeds and inspected event by
event.

## Interpretation

The model learns persistent component state substantially more easily than
sparse event timing. Completion recall with very low precision indicates that
the fixed global 0.5 peak threshold is not calibrated to the component score
distributions. The rare outcome head can learn on a small repeated subset, but
the default full-dataset sampling does not preserve that signal reliably.

The unseen-composition result is inconclusive rather than evidence against the
factorization: it contains only 19 validation frames and neither composition was
predicted correctly. Retain the generic verb/object head, but compare it with a
closed-set head and report the 19-frame denominator explicitly.

## Required next experiment

Before an 80-epoch run:

1. Add validation-only threshold sweeps and report PR-AUC for each outcome and
   component; freeze thresholds before any test evaluation.
2. Compare rare-window boosts 4 and 8, plus an outcome-aware batch sampler that
   guarantees incorrect examples in optimizer steps.
3. Increase the incorrect-event contribution to checkpoint selection, or use a
   Pareto rule requiring non-zero incorrect precision and recall.
4. Run a 20--30 epoch pilot for at least three seeds. Inspect every one of the
   five validation incorrect events and all predicted false alerts.
5. Repeat the winning head with VideoMAEv2-giant SSv2 motion features. Attribute
   gains separately to features, sampling, threshold calibration, and model
   structure.

Only after the validation precision/recall trade-off is non-degenerate should
80 epochs be used as an early-stopped maximum with patience 15. Preserve the
dataset-independent boundary: dataset adapters and procedure-schema JSON files
produce the cache-v2 tensors, while the causal model, losses, graph filter,
trainer, and metrics consume only that portable contract.
