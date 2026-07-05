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

Run the web UI:

```bash
streamlit run web/app.py
```

## Architecture Plan

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

This workspace currently sees an NVIDIA RTX 5000 Ada on PCI, but `nvidia-smi` cannot communicate with the driver and PyTorch is not installed. Treat local development as CPU-only until `/dev/nvidia*`, CUDA, and PyTorch are available.
