# Codex handoff context: StateGraph-PSR Lite

Last updated: 2026-07-20

## Objective

Build a resource-efficient Stage-1 system for IndustReal egocentric procedure-step recognition that improves on the group's motion-only DiffAct and motion/appearance ASFormer baselines. The longer-term goal is live assembly monitoring with specific fault identification, future-state prediction, and recovery recommendations.

## Implemented architecture

The current candidate is **StateGraph-PSR Lite**:

- frozen/cached motion features, preferably the group's 1408-d SSv2 VideoMAEv2-giant features;
- a frozen ConvNeXt-Tiny keyframe encoder for tools, parts, and connection appearance;
- optional IndustReal hands, gaze, and headset-pose vectors;
- per-time-step modality gating into a 192-d representation;
- eight causal gated depthwise temporal blocks with dilations 1–16;
- sparse causal attention every second block;
- one learnable action prototype per procedure step;
- a differentiable sparse transition-graph filter;
- joint heads for AR action segmentation, multi-label component completion, per-component event outcome, eleven component states, boundary, and next action;
- entropy and energy scores for later alert calibration.

The trainable head is roughly four million parameters. The large encoders remain outside the training graph, making the default configuration suitable for a 24 GB GPU.

Primary files:

- `src/aiops/models/stategraph_psr.py`
- `src/aiops/features/industreal_cache.py`
- `src/aiops/data/industreal.py`
- `src/aiops/data/stategraph_cache.py`
- `src/aiops/training/train_stategraph_psr.py`
- `src/aiops/evaluation/temporal_metrics.py`
- `configs/stategraph_psr_stage1.json`
- `docs/stategraph_psr_stage1_implementation.md`
- `docs/assets/stategraph_psr_stage1_architecture.{svg,png}`

## Current validation status

The full repository suite produced:

```text
12 passed, 1 skipped
```

The skipped test is `tests/test_stategraph_model.py`, because the laptop environment did not have PyTorch. It performs output-shape, causal-invariance, uncertainty-range, loss-finiteness, and backward-gradient checks and must be rerun on the GPU desktop.

The IndustReal parser, dense-label construction, cache round trip, sparse graph estimation, temporal metrics, compilation, and previous project tests passed.

## Dataset discovery result

`dataset/dataset` in the laptop checkout contains twelve large HoloLens MP4 files, but no official PSR annotation tree:

```text
official_layout: false
recording_count: 12
labeled_recordings: 0
video_only_recordings: 12
```

Do not start supervised StateGraph training from these MP4 files alone. On the desktop, locate the fully extracted IndustReal release containing:

```text
recordings/
  train|val|test/
    RECORDING_ID/
      rgb/
      PSR_labels_with_errors.csv
      PSR_labels_raw.csv
      hands.csv
      gaze.csv
      pose.csv
```

If only videos are available, complete/export step and fault annotations with the Hand Atlas labeler before training.

### GPU desktop verification

The desktop is reachable as `aiops@192.168.20.148` and was verified after the initial handoff:

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB;
- driver: 550.78;
- clean checkout: `/home/aiops/AIOps-stategraph`;
- original dirty research checkout, deliberately preserved: `/home/aiops/AIOps`;
- real data root: `/home/aiops/AIOps/data/raw/industreal`;
- split count observed: 36 train and 16 validation recording directories;
- PyTorch: 2.4.1+cu121 with CUDA available;
- torchvision was not installed at audit time.

The desktop release uses a supported hybrid layout: PSR annotations are under
`recordings/{split}/{recording_id}`, while RGB videos are root-level
`{recording_id}.mp4` files. Its official CSV files are headerless. The adapter
was extended and tested for both of these facts after inspecting
`27_main_0_1` (1,248 frames, 10 FPS, 1280×720).

## Desktop continuation procedure

After cloning or pulling the published branch:

```powershell
cd C:\path\to\AIOps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch CUDA build that matches the desktop driver. The example below is a starting point, not a substitute for checking the installed driver:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[dev,vision]"
```

Run tests first:

```powershell
python -m pytest -q
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Audit the real dataset root:

```powershell
python -m aiops.data.industreal --data-root "/home/aiops/AIOps/data/raw/industreal"
```

Build the default Swin3D-S + ConvNeXt cache:

```powershell
python -m aiops.features.industreal_cache `
  --data-root "D:\IndustReal" `
  --output-dir "D:\IndustReal_cache\stategraph_v2_swin_convnext" `
  --device cuda --mixed-precision bf16
```

Preferred path when `RECORDING_ID.npy` or `.npz` VideoMAEv2 features already exist:

```powershell
python -m aiops.features.industreal_cache `
  --data-root "D:\IndustReal" `
  --output-dir "D:\IndustReal_cache\stategraph_v2_vmae_convnext" `
  --motion-features-dir "D:\features\videomaev2_giant_ssv2" `
  --device cuda --mixed-precision bf16
```

Train without touching the test split (after the cache-v2 smoke checks below):

```powershell
python -m aiops.training.train_stategraph_psr `
  --cache-index "D:\IndustReal_cache\stategraph_v2_vmae_convnext\index.json" `
  --output-dir runs\stategraph_psr_v1 `
  --precision bf16 --batch-size 2 --accumulation-steps 8 `
  --sequence-length 256 --sequence-stride 192 `
  --epochs 80 --patience 15
```

Only after the architecture and thresholds are frozen, repeat with `--evaluate-test`.

## First desktop tasks

1. Confirm the real dataset root and run the audit.
2. Run the skipped tensor test with PyTorch installed.
3. Inspect PSR CSV column names and compare five parsed event/state sequences against the source annotations.
4. Build a one-recording cache and inspect feature/label timestamps before launching the complete extraction.
5. Run a two-recording overfit experiment. The joint training loss should fall sharply; failure here indicates a label alignment or masking bug.
6. Train the full default model and record frame accuracy, Edit, F1@10/25/50, incorrect precision/recall/F1, remove metrics, state macro-F1, and detection delay.
7. Run the A–E ablation table in `docs/stategraph_psr_stage1_implementation.md` with at least three seeds for the final candidates.

One-recording cache smoke test:

```bash
python -m aiops.features.industreal_cache \
  --data-root /home/aiops/AIOps/data/raw/industreal \
  --output-dir /home/aiops/AIOps/data/processed/stategraph_smoke \
  --recording-id 27_main_0_1 --device cuda --mixed-precision bf16
```

Verified label audit: 24 step IDs, 463 total completion events (344 correct,
19 incorrect, 100 remove), and 96.3% mean dense step coverage at stride 5.
Raw component supervision contained 2,470 correct, 1,474 pending, and only 38
incorrect states, so rare-fault sampling and PR-oriented reporting are mandatory.

### First full-cache and training-smoke results

The fallback Swin3D-S + ConvNeXt cache completed for all 52 recordings in about
32 minutes: 23,414 sampled time steps, 127 MB, no NaNs, 88 training windows,
and 20 rare-fault windows. A five-epoch end-to-end training smoke test completed
in 46 seconds with 3,284,122 trainable parameters and reduced total training
loss from 18.18 to 4.51. All 14 desktop tests passed.

Do not interpret the original smoke validation numbers as a benchmark. They
were produced by obsolete cache schema v1, where the raw 24 PSR event IDs were
treated as mutually exclusive dense steps. Those IDs include outcome variants
and simultaneous multi-component events; five IDs consequently had zero dense
frames at stride 5.

Cache schema v2 resolves this blocker. AR annotations now supervise the true
action timeline. The 24 observed PSR IDs map to ten semantic components, with a
multi-label completion matrix and separate correct/incorrect/remove outcome per
component. Same-frame completions are preserved. The mapping lives in
`configs/procedure_schemas/industreal_v1.json`; the model itself only consumes
the portable contract documented in `docs/procedure_schema_v2.md`.

The completed 128 MB desktop cache is v1 and must be regenerated. The trainer
now rejects it explicitly. Its frozen feature arrays can be reused, so a fast
relabel-only desktop rebuild is:

```bash
python -m aiops.features.industreal_cache \
  --data-root /home/aiops/AIOps/data/raw/industreal \
  --output-dir /home/aiops/AIOps/data/processed/stategraph_v2_swin_convnext \
  --motion-features-dir /home/aiops/AIOps/data/processed/stategraph_swin_convnext \
  --appearance-features-dir /home/aiops/AIOps/data/processed/stategraph_swin_convnext
```

After rebuilding, overfit two to four recordings,
then run a 20–30 epoch pilot. An 80-epoch run is recommended only as an
early-stopped maximum (`--patience 15`) after those checks. Checkpoint selection
uses incorrect-event F1 rather than recall, because the first smoke run showed
that near-100% recall could result from overpredicting the incorrect class.

The relabel-only v2 rebuild completed on 2026-07-20 at
`/home/aiops/AIOps/data/processed/stategraph_v2_swin_convnext`: 52 recordings,
23,414 rows, ten completion components, 315 train and 148 validation events,
95/47 multi-component rows, and no invalid/NaN records. The audit found two
validation-only AR classes: `66 plug_small_screw_pin` and
`72 pull_small_screw_pin`. Training contains both verbs and the object phrase
through other actions (`take/put_small_screw_pin`), so the action head now uses
shared verb/object logits and prototypes plus a seen-class residual. Graph bias
is disabled only for train-unseen compositions, and evaluation reports their
accuracy separately.

### Cache-v2 overfit and full-pilot result

The corrected four-recording overfit run passed the functional wiring gate:
total loss fell from 12.13 to roughly 1--2, action F1@50 reached 46.81, and
completion-event recall reached 62.96. Incorrect-event recall reached 100 on
the repeated subset, but precision was only 2.90, so this is evidence that the
head learns rather than evidence of generalization.

A 25-epoch seed-7 pilot then ran on the official 36-train/16-validation split
with BF16, batch size 2, accumulation 4, and the fallback Swin3D-S + ConvNeXt
cache. It completed in 176 seconds. The epoch-16 selected checkpoint obtained
21.35 frame accuracy, 30.52 Edit, 9.70 F1@50, 4.54 completion-event F1, 0.00
incorrect-event F1, and 81.59 state accuracy. Training loss fell from 19.32 to
4.49, but F1@50 peaked before the end and the incorrect detector remained
degenerate. Do not start the 80-epoch run with this configuration.

Full commands, metrics, interpretation, and the next experimental gate are in
`docs/stategraph_psr_v2_pilot_report.md`. The immediate work is validation-only
event-threshold/PR calibration, stronger outcome-aware sampling, three pilot
seeds, and then a controlled repeat with the preferred VideoMAEv2-giant SSv2
features. The test split remains untouched.

### v3 refinement and live monitoring

The v3 implementation adds bounded graph fusion, progress supervision,
asymmetric completion loss, guaranteed rare-event windows, state/action event
conditioning, component normality prototypes, balanced event-to-state mapping,
outcome-specific event peaks, and causal state-transition evidence. The desktop
suite passes 25 tests.

The four-recording overfit gate reached 60.39 frame accuracy, 45.67 F1@50,
100 incorrect-event F1, and 93.53 state accuracy. The official seed-7 20-epoch
pilot reached 28.51 frame accuracy, 26.70 Edit, 12.80 F1@50, and 87.19 state
accuracy. An intermediate decoder recovered one validation incorrect event at
below 0.1% precision, but the final training-derived refractory rule removed
that unstable match. A second seed ran for 25 epochs and improved frame
accuracy/F1@50 to 30.77/15.32 while incorrect precision, recall, F1, and PR-AUC
stayed zero. Tolerance windows through two seconds found no hidden delayed
detections. This is not sufficient evidence for the final 80-epoch
fallback-feature run.

Exact results and the go/no-go gate are in
`docs/stategraph_psr_v3_experiment_report.md`. The live TensorBoard dashboard is
`http://192.168.20.148:6006`; its isolated runtime is
`/home/aiops/.venvs/aiops-dashboard`. Training and NVIDIA telemetry are written
under `/home/aiops/AIOps/runs`.

### High-capacity causal refinement result

The optional large preset in `configs/stategraph_psr_stage1_large.json` uses a
384-d hidden state, 12 shared blocks, two four-block action-refinement stages,
and a four-block event/state branch. It has 34.1M parameters. Desktop tests pass
and a representative BF16 forward/backward used about 505 MB of PyTorch peak
allocation; real batch-4, length-384 training used about 2.4 GB total GPU
memory.

The four-recording overfit gate retained 100 incorrect-event F1. On the official
split, the selected epoch-16 checkpoint reached 33.41 frame accuracy, 29.92
Edit, and 18.05 F1@50, improving the best lite pilot by 2.73 F1@50 points.
Incorrect F1 and PR-AUC remained zero. Retain the large model for the next
VideoMAEv2-SSv2 experiment, but do not claim that head scaling solved fault
generalization. Full details are in `docs/stategraph_psr_large_refinement.md`.

### VideoMAEv2 motion upgrade

`aiops.features.videomaev2_features` now provides a resumable, cache-aligned
VideoMAEv2 extractor. The default public `OpenGVLab/VideoMAEv2-giant` encoder is
UnlabeledHybrid-pretrained and outputs 1408-d features. The preferred supervised
SSv2 checkpoint remains `vit_g_hybrid_pt_1200e_ssv2_ft`; it can be supplied with
`--checkpoint` after obtaining it through the official download-request form.
Both paths use the official 16-frame, interval-2 SSv2 sampling recipe. Feature
provenance is written to a manifest and can be propagated into cache `index.json`
with `--motion-backend-name`. See `docs/videomaev2_motion_cache.md`.

### v5 dual-motion XL result

The current strongest Stage-1 action model is the optional dual-motion v5 XL
preset (`configs/stategraph_psr_stage1_xl.json`). It keeps VideoMAEv2-giant and
Swin3D-S in separate stems, adds learned four-stream gating, preserves
cross-stream disagreement for the six-block fault/state branch, uses three
action-refinement stages, and trains rare outcomes for a causal two-row horizon
while keeping completion timing exact. The head has 87.5M trainable parameters.

The graph filter is now a vectorized first-order causal prior; this removed the
main Python/CUDA serialization bottleneck. All 31 desktop tests pass. The
four-recording timing gate obtained 66.67 incorrect F1 and 100 normality AP. On
the official split, seed 7 selected epoch 24 with 34.99 frame accuracy, 34.26
Edit, 19.12 F1@50, 8.82 incorrect AP, 0 incorrect F1, and 90.37 state accuracy.
This is the best action/step result but is not yet sufficient evidence for a
production fault alert or a final 80-epoch run. Run seeds 17 and 29 first. Full
architecture, commands, comparisons, compute telemetry, and the go/no-go gate
are in `docs/stategraph_psr_v5_dualmotion_report.md`.

## Known risks and decisions

- The first cache-v2 pilot is a diagnostic baseline, not a final accuracy claim; it used fallback Swin3D-S + ConvNeXt features and failed the incorrect-event gate.
- Incorrect-install examples are highly imbalanced. Report precision and PR-AUC as well as recall.
- Precomputed features are linearly resampled when their length differs from cache centers. Verify timestamps and replace this fallback with explicit timestamp mapping when source metadata is available.
- The transition graph is estimated from training recordings only. It retains zero-valued impossible edges so graph regularization remains meaningful.
- Validation splitting is recording-level; the trainer refuses a one-recording fallback to prevent window leakage.
- The best validation checkpoint is reloaded before final validation or optional test evaluation.
- Start with frozen encoders. Fine-tune only after the head/data ablations, ideally with LoRA or final-block unfreezing.

## Next architectural milestone

Stage 2 should add a hierarchical fault taxonomy using action belief, per-component event outcome, component-state delta, detected tool/object, and graph edge. Stage 3 should roll component-state belief forward and search a recovery graph for a safe return path. A VLM can verbalize and visually ground the selected correction, while the state/procedure graph constrains the recommendation.

## Workspace hygiene

At handoff time the checkout also contained unrelated/unpublished labeler work and a multi-gigabyte `dataset.zip`. Do not add large datasets, caches, checkpoints, or secrets to Git. Review `git status` and stage StateGraph files explicitly.
