#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
manifest="${MANIFEST:-data/raw/assembly101/ego30_manifest.json}"
video_root="${VIDEO_ROOT:?Set VIDEO_ROOT to the Assembly101 ego video directory}"
videomaev2_root="${VIDEOMAEV2_ROOT:?Set VIDEOMAEV2_ROOT to the official VideoMAEv2 checkout}"
checkpoint="${SSV2_CHECKPOINT:?Set SSV2_CHECKPOINT to the official giant SSv2 checkpoint}"
output_dir="${OUTPUT_DIR:-data/processed/assembly101_ego30_ssv2_stategraph}"
batch_size="${FEATURE_BATCH_SIZE:-1}"
max_recordings="${MAX_RECORDINGS:-}"

extra_args=()
if [[ -n "$max_recordings" ]]; then
  extra_args+=(--max-recordings "$max_recordings")
fi

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
"$python_bin" -m aiops.features.assembly101_video_cache \
  --manifest "$manifest" \
  --video-root "$video_root" \
  --output-dir "$output_dir" \
  --motion-backend videomaev2-ssv2 \
  --videomaev2-root "$videomaev2_root" \
  --videomaev2-checkpoint "$checkpoint" \
  --clip-frames 16 \
  --stride-frames 8 \
  --feature-batch-size "$batch_size" \
  --precision bf16 \
  --skip-existing \
  "${extra_args[@]}"

if [[ -z "$max_recordings" ]]; then
  "$python_bin" scripts/merge_assembly101_ego30_cache.py \
    --output-dir "$output_dir" \
    --manifest "$manifest"
fi
