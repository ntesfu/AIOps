# AIOps

AIOps is a research prototype for egocentric assembly step detection. The first milestone is an offline workflow: upload or pass in a video, produce temporal step predictions, validate the predicted order against an expected procedure, and save machine-readable output that can later drive annotated video rendering.

The project is designed for a CPU-first development environment. Heavy video backbones can be added later, but the initial code keeps the model interface small and testable.

## Current Scope

- Baseline step segment generation for local demos.
- Procedure validation for skipped, repeated, out-of-order, unknown, and low-confidence steps.
- CLI inference entrypoint that writes JSON predictions.
- Streamlit upload UI scaffold.
- Dataset conversion placeholder for Assembly101-style temporal annotations.
- Unit tests for the core prediction and validation logic.

## Repository Layout

```text
configs/              Example runtime and procedure configs
src/aiops/            Python package
src/aiops/data/       Dataset conversion utilities
src/aiops/inference/  CLI and runtime pipeline
src/aiops/models/     Model interfaces and baselines
src/aiops/validation/ Procedure/order validation
tests/                Unit tests
web/                  Streamlit upload UI
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,web,video]"
python -m pytest
```

Run the demo inference pipeline:

```bash
python -m aiops.inference.cli \
  --video sample.mp4 \
  --procedure configs/example_procedure.json \
  --output runs/sample_predictions.json
```

Run the Hand Atlas web UI (a browser-local MediaPipe and temporal-step dataset labeler):

```bash
streamlit run web/app.py
```

The labeler can import individual videos, an extracted folder, or a ZIP archive;
save and manually correct MediaPipe 21-joint hand keyframes; create verified
procedure-step intervals; validate overlaps and taxonomy; render an annotated
WebM; and export a training bundle containing the canonical project JSON,
temporal JSONL manifest, COCO hand keypoints, CSV tables, dataset YAML, and
procedure configuration. Large archives are indexed lazily, but extracting a
selected video from a multi-gigabyte ZIP can take time. Selecting the extracted
folder is the fastest workflow for repeated annotation sessions.

## Architecture Plan

### StateGraph-PSR Lite (IndustReal Stage 1)

The research training path uses cached motion and appearance features, IndustReal hand and headset-pose streams, component/object regions, a compact causal TCN-attention head, action prototypes, a differentiable procedure graph, and a corrected two-level label contract. AR annotations supervise the mutually-exclusive action timeline; PSR completion rows supervise multi-label part events and a per-part `correct / incorrect / remove` outcome. It is designed to train on one 24 GB GPU while keeping the path open to stronger precomputed VideoMAEv2 or InternVideo features.

The default trainable head is approximately four million parameters. Its outputs are a causal action timeline, simultaneous component completions, a separate outcome for every completed part, eleven component-state predictions, action boundaries, next-action anticipation, and calibrated uncertainty signals. Frozen encoders are run once to create compressed per-recording caches, so ordinary experiments train only the temporal/state head.

- Architecture and training guide: [`docs/stategraph_psr_stage1_implementation.md`](docs/stategraph_psr_stage1_implementation.md)
- Architecture image: [`docs/assets/stategraph_psr_stage1_architecture.svg`](docs/assets/stategraph_psr_stage1_architecture.svg)
- Codex desktop handoff: [`CODEX_CONTEXT.md`](CODEX_CONTEXT.md)
- Model: [`src/aiops/models/stategraph_psr.py`](src/aiops/models/stategraph_psr.py)
- Feature cache: [`src/aiops/features/industreal_cache.py`](src/aiops/features/industreal_cache.py)
- Trainer: [`src/aiops/training/train_stategraph_psr.py`](src/aiops/training/train_stategraph_psr.py)
- Portable label contract: [`docs/procedure_schema_v2.md`](docs/procedure_schema_v2.md)
- Reference implementation review: [`docs/reference_implementations_review.md`](docs/reference_implementations_review.md)
- Critique response and evidence-first roadmap: [`docs/stategraph_psr_critique_response_plan.md`](docs/stategraph_psr_critique_response_plan.md)
- State-centric inference core and typed belief tracker: [`docs/stateverify_psr_core.md`](docs/stateverify_psr_core.md)
- Gaze-free hand/object observer training: [`docs/stateverify_observer_training.md`](docs/stateverify_observer_training.md)
- StateVerify observer experiment results: [`docs/stateverify_observer_experiment_2026-07-23.md`](docs/stateverify_observer_experiment_2026-07-23.md)
- StateVerify normal-prototype results: [`docs/stateverify_prototype_experiment_2026-07-23.md`](docs/stateverify_prototype_experiment_2026-07-23.md)
- Strict-causal IndustReal baseline protocol: [`docs/industreal_staged_baseline_protocol.md`](docs/industreal_staged_baseline_protocol.md)
- Component-conditioned IndustReal evidence pilot: [`docs/industreal_component_evidence_experiment.md`](docs/industreal_component_evidence_experiment.md)
- Strict-causal IndustReal ROI evidence experiment: [`docs/industreal_roi_evidence_experiment.md`](docs/industreal_roi_evidence_experiment.md)

Audit an extracted release, build a cache, and train:

```bash
python -m aiops.data.industreal --data-root "D:/IndustReal"
python -m aiops.features.industreal_cache --data-root "D:/IndustReal" --output-dir "D:/IndustReal_cache/stategraph_v2" --device cuda
python -m aiops.training.train_stategraph_psr --cache-index "D:/IndustReal_cache/stategraph_v2/index.json" --output-dir runs/stategraph_psr_v2
```

The cache builder accepts the official `recordings/train|val|test/...` annotation tree with either per-recording `rgb/` frames or root-level `RECORDING_ID.mp4` videos. Headerless official AR/PSR CSVs and headered exports are supported. Video-only bundles cannot supervise the action/error/state heads. Cache schema v2 rejects the old 24-way PSR proxy target, which silently lost simultaneous events. The trainer keeps splits at recording level, uses BF16 and gradient accumulation by default, selects checkpoints using action segmentation and incorrect-event F1, and evaluates the test split only when explicitly requested.

### Assembly101 mistake-aware training

Assembly101 is the active replacement while the complete IndustReal release is unavailable. The reproducible research subset uses 60 one-frame-per-second C10095 RGB recordings from a public Assembly101-derived mirror and the official part-to-part mistake annotations. Splits are actor-disjoint (40/10/10 recordings), and validation thresholds are frozen before test evaluation. Raw data and cached features remain ignored by Git.

```bash
python scripts/download_assembly101_subset.py --workers 4

TORCH_HOME=data/model_cache PYTHONPATH=.deps:src .venv/bin/python \
  -m aiops.features.assembly101_cache \
  --manifest data/raw/assembly101/subset_manifest.json \
  --output-dir data/processed/assembly101_stategraph \
  --device cuda

PYTHONPATH=.deps:src .venv/bin/python \
  -m aiops.training.train_stategraph_psr \
  --cache-index data/processed/assembly101_stategraph/index.json \
  --output-dir runs/stategraph_assembly101_80ep \
  --epochs 80 --patience 100 --batch-size 8 --accumulation-steps 2 \
  --sequence-length 256 --sequence-stride 192 \
  --hidden-dim 256 --num-temporal-blocks 8 --attention-every 2 --num-heads 8 \
  --num-action-refinement-stages 1 --num-refinement-blocks 4 --num-event-blocks 4 \
  --learning-rate 0.0002 --rare-windows-per-batch 2 \
  --incorrect-selection-weight 1.0 --evaluate-test
```

See [`configs/stategraph_assembly101.json`](configs/stategraph_assembly101.json) for the selected architecture and exact training settings. Assembly101 is CC BY-NC 4.0; this data path is for non-commercial research.

The completed 80-epoch experiment, actor-held-out results, limitations, and artifact checksum are documented in [`docs/assembly101_stategraph_80epoch_report.md`](docs/assembly101_stategraph_80epoch_report.md). The compact BF16 inference checkpoint is [`artifacts/stategraph_assembly101_80ep_bf16.pt`](artifacts/stategraph_assembly101_80ep_bf16.pt).

Implemented temporal pipeline:

1. Decode videos into overlapping fixed-duration clips.
2. Generate one feature vector per clip. The current local backend is `statistical_clip`; the interface is shaped so a frozen VideoMAE/EgoVLP encoder can replace it.
3. Feed the clip feature sequence into a temporal probability head. The current runnable head is a deterministic temporal prior placeholder; `src/aiops/models/mstcn.py` contains the optional PyTorch MS-TCN++ style model builder for the learned version.
4. Smooth clip probabilities and decode non-overlapping temporal step segments.
5. Validate the predicted sequence against the procedure graph.
6. Export JSON for the UI and later annotated video rendering.

The baseline included here creates deterministic pseudo-segments from video metadata so the UI, validator, and output format can be developed before GPU training is available.

Run the temporal pipeline:

```bash
python -m aiops.inference.cli \
  --mode temporal \
  --video sample.mp4 \
  --procedure configs/example_procedure.json \
  --architecture configs/temporal_architecture.json \
  --output runs/temporal_predictions.json
```

## Tiny CPU Training Run

TinyVIRAT is used as a small public action-recognition smoke test while Assembly101 access is being prepared. The downloader pulls the official archive, extracts a balanced tiny subset from `tiny_train.json`, and trains a simple CPU softmax classifier over lightweight frame statistics.

```bash
python -m aiops.data.tinyvirat \
  --max-classes 3 \
  --max-per-class 8 \
  --archive data/raw/TinyVIRAT.zip \
  --output-dir data/processed/tinyvirat_action_subset

python -m aiops.training.simple_video_classifier \
  --manifest data/processed/tinyvirat_action_subset/manifest.jsonl \
  --output-dir runs/models/tinyvirat_simple \
  --epochs 250 \
  --learning-rate 0.15
```

Train the GPU-backed MS-TCN temporal head:

```bash
module load cuda-11.6
PYTHONPATH=src:.deps python3 -m aiops.training.train_mstcn \
  --manifest data/processed/tinyvirat_action_subset/manifest.jsonl \
  --output-dir runs/models/mstcn_tinyvirat \
  --epochs 80 \
  --learning-rate 0.0005 \
  --device cuda
```

Build a larger multi-step procedural dataset and train dense temporal segmentation:

```bash
PYTHONPATH=src:.deps python3 -m aiops.data.tinyvirat \
  --max-classes 8 \
  --max-per-class 20 \
  --archive data/raw/TinyVIRAT.zip \
  --output-dir data/processed/tinyvirat_action_subset_v2

PYTHONPATH=src:.deps python3 -m aiops.data.procedural_sequences \
  --source-manifest data/processed/tinyvirat_action_subset_v2/manifest.jsonl \
  --output-dir data/processed/procedural_tinyvirat_v2 \
  --num-sequences 72 \
  --steps-per-sequence 7

module load cuda-11.6
PYTHONPATH=src:.deps python3 -m aiops.training.train_mstcn \
  --manifest data/processed/procedural_tinyvirat_v2/manifest.jsonl \
  --manifest-type temporal \
  --output-dir runs/models/mstcn_procedural_v2_best \
  --epochs 60 \
  --hidden-dim 64 \
  --num-stages 3 \
  --num-layers 6 \
  --dropout 0.35 \
  --smoothing-weight 0.1 \
  --device cuda
```

Run learned checkpoint inference:

```bash
module load cuda-11.6
PYTHONPATH=src:.deps python3 -m aiops.inference.cli \
  --mode temporal \
  --video data/processed/tinyvirat_action_subset/videos/activity_walking/VIRAT_S_000203_07_001341_001458__1.mp4 \
  --procedure configs/example_procedure.json \
  --architecture configs/temporal_architecture.json \
  --checkpoint runs/models/mstcn_tinyvirat/checkpoint.pt \
  --output runs/mstcn_inference.json
```

The Streamlit UI supports baseline inference, temporal inference, learned MS-TCN checkpoint inference, clip-level labels, decoded segments, validation events, and JSON download.

## Compute Notes

The validated training host exposes an NVIDIA RTX 5000 Ada with 32 GB VRAM. The workspace-local runtime uses `PYTHONPATH=.deps:src`; pretrained torchvision weights are cached under `data/model_cache` and selected through `TORCH_HOME=data/model_cache`.
