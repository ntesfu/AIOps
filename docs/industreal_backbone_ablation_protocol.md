# IndustReal frozen-backbone ablation

## Question and experimental unit

This protocol answers one narrow question: does a frozen VideoMAE V2-B motion
representation improve StateGraph-PSR over frozen Kinetics-400 Swin3D-S?
It does not change the temporal head, ROI evidence, optimizer, data split,
checkpoint selectors, or alert calibration.

The comparison is paired by recording and seed. The official validation split
is used for all decisions. The test split remains sealed.

## Required caches

Prepare two strict-causal StateGraph cache indexes:

1. `swin3d_s`: the existing trailing-clip Swin3D-S + ConvNeXt + ROI cache.
2. `videomaev2_b`: the same cache rebuilt with only the global motion array
   replaced by frozen VideoMAE V2-B features.

Pin the exact VideoMAE model identifier, repository revision, checkpoint hash,
clip anchor, frame count, sampling rate, and pooling in its feature manifest.
The repository's extractor supports this through
`aiops.features.videomaev2_features`; do a one-recording extraction before a
full cache. Do not silently use the extractor's default Giant model when the
experiment is labeled `VideoMAE V2-B`.

The ablation runner checks:

- identical recording IDs and official splits;
- identical sequence lengths and label taxonomy;
- exact equality of step, completion, outcome, state, boundary, timestamp, and
  prediction-frame arrays;
- strict-causal cache audits performed by the staged launcher.

The paired-label digest is saved in the experiment manifest.

## Two-stage budget

The declared settings live in
`configs/industreal_backbone_ablation.json`.

### Quick screen

- Both backbones, seed 7.
- 12 action-training epochs with no event refinement.
- Same action-only checkpoint selection.
- Purpose: reject an expensive representation that does not improve action
  segmentation before spending the rare-event training budget.

VideoMAE V2-B proceeds only if its paired F1@50 improvement is at least 2
percentage points, frame accuracy regresses by no more than 2 points, and
training stays below 23 GiB.

### Promotion

- Both backbones, seeds 7, 17, and 29.
- 30 action epochs followed by 16 frozen-action event-refinement epochs.
- Independent action-best and operational-event-best selection.
- Dual-expert composition on validation.

Promotion additionally requires positive mean incorrect-event recall and no
more than two false alerts per minute. Report every seed; with only a few
validation mistakes, the mean alone is not sufficient evidence.

## Commands

First validate and write the immutable manifest:

```bash
PYTHONPATH=src python scripts/run_industreal_backbone_ablation.py validate \
  --config configs/industreal_backbone_ablation.json \
  --swin-cache data/processed/industreal_roi_swin/index.json \
  --videomae-cache data/processed/industreal_roi_videomaev2_b/index.json
```

Inspect the planned jobs without training:

```bash
PYTHONPATH=src python scripts/run_industreal_backbone_ablation.py plan \
  --config configs/industreal_backbone_ablation.json \
  --swin-cache data/processed/industreal_roi_swin/index.json \
  --videomae-cache data/processed/industreal_roi_videomaev2_b/index.json \
  --stage quick
```

Run and compare the quick screen:

```bash
PYTHONPATH=src python scripts/run_industreal_backbone_ablation.py run \
  --config configs/industreal_backbone_ablation.json \
  --swin-cache data/processed/industreal_roi_swin/index.json \
  --videomae-cache data/processed/industreal_roi_videomaev2_b/index.json \
  --stage quick

PYTHONPATH=src python scripts/compare_industreal_backbones.py \
  --config configs/industreal_backbone_ablation.json \
  --stage quick \
  --output experiments/backbones/quick_comparison.json
```

Only after the JSON gate says `"promote_videomaev2_b": true`, repeat the run
and comparison with `--stage promotion`.

## Outputs and resource interpretation

- `experiments/backbones/{stage}_manifest.json`: cache hashes, paired-label
  digest, host, exact protocol, and jobs.
- `experiments/backbones/resources/.../*.jsonl`: sampled whole-device GPU
  memory, utilization, and power.
- `*.summary.json`: wall time and resource summaries per job.
- Training `metrics.json`: process-local peak PyTorch allocation.
- `*_comparison.json` and `.md`: per-seed metrics, mean/stdev, paired deltas,
  individual gate outcomes, and the promotion decision.

Whole-device telemetry can include another process. Use the trainer's
process-local peak VRAM for the 23-GiB gate and the sampled device telemetry for
capacity planning.

## Interpretation

The candidate wins only by the predeclared paired rule. A higher frame accuracy
alone is not enough. If VideoMAE V2-B improves F1@50 but fails rare-event
operational gates, retain it as the action expert and keep the better event
expert; record that as a later dual-backbone experiment rather than changing
this ablation post hoc.
