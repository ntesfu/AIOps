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

1. Decode videos into fixed-duration clips.
2. Generate clip features with a frozen image/video encoder.
3. Feed features into a lightweight temporal segmentation model.
4. Smooth predictions in a rolling buffer.
5. Validate the predicted sequence against a procedure graph.
6. Render annotated video and export JSON for review.

The baseline included here creates deterministic pseudo-segments from video metadata so the UI, validator, and output format can be developed before GPU training is available.

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

## Compute Notes

This workspace currently sees an NVIDIA RTX 5000 Ada on PCI, but `nvidia-smi` cannot communicate with the driver and PyTorch is not installed. Treat local development as CPU-only until `/dev/nvidia*`, CUDA, and PyTorch are available.
