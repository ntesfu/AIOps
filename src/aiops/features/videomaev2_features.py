from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from aiops.data.industreal import IndustRealRecording, discover_industreal_recordings


DEFAULT_MODEL_ID = "OpenGVLab/VideoMAEv2-giant"
DEFAULT_REVISION = "d27568eb41ccb2d41bb191fc2e3fe5aad74942d4"


class ClipEncoder(Protocol):
    feature_dim: int

    def encode(self, clips: list[list[np.ndarray]]) -> np.ndarray: ...


@dataclass(frozen=True)
class ExtractionRecord:
    recording_id: str
    split: str
    path: str
    num_steps: int
    feature_dim: int
    source_frames: int


def centered_clip_indices(
    center: int,
    length: int,
    num_frames: int = 16,
    sampling_rate: int = 2,
) -> np.ndarray:
    """Return a boundary-clamped, fixed-rate clip centered on a cache step."""
    if length <= 0 or num_frames <= 0 or sampling_rate <= 0:
        raise ValueError("length, num_frames, and sampling_rate must be positive")
    offsets = (np.arange(num_frames, dtype=np.int64) - (num_frames - 1) / 2.0) * sampling_rate
    return np.clip(np.rint(center + offsets).astype(np.int64), 0, length - 1)


class HuggingFaceVideoMAEv2Encoder:
    """Frozen VideoMAEv2 clip encoder with optional official fine-tuned weights."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        checkpoint: str | None = None,
        device: str | None = None,
        precision: str = "bf16",
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - GPU environment only
            raise RuntimeError(
                "Install the 'videomaev2' extra (torch, transformers, timm, safetensors)."
            ) from exc

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu" and precision != "fp32":
            precision = "fp32"
        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[precision]
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, use_fast=False
        )
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        if checkpoint:
            model = AutoModel.from_config(config, trust_remote_code=True)
            self._load_official_checkpoint(model, checkpoint)
        else:
            model = AutoModel.from_pretrained(
                model_id,
                config=config,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            )
        self.model = model.eval().to(device=self.device, dtype=self.dtype)
        model_config = getattr(config, "model_config", {})
        self.feature_dim = int(model_config.get("embed_dim", getattr(model, "embed_dim", 0)))
        if self.feature_dim <= 0:
            raise RuntimeError("Could not determine the VideoMAEv2 output dimension")

    def _load_official_checkpoint(self, model: Any, checkpoint: str) -> None:
        payload = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("model", "module"):
            if isinstance(payload, dict) and key in payload:
                payload = payload[key]
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint payload in {checkpoint}")
        state = {str(key).removeprefix("module."): value for key, value in payload.items()}
        target = getattr(model, "model", model)
        incompatible = target.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not compatible with the selected VideoMAEv2 architecture; "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )

    def encode(self, clips: list[list[np.ndarray]]) -> np.ndarray:
        if not clips:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        tensors = []
        for clip in clips:
            # OpenCV produces BGR; Hugging Face processors expect RGB arrays.
            rgb_clip = [np.ascontiguousarray(frame[..., ::-1]) for frame in clip]
            values = self.processor(rgb_clip, return_tensors="pt")["pixel_values"]
            tensors.append(values[0])
        pixel_values = self.torch.stack(tensors).permute(0, 2, 1, 3, 4)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        with self.torch.inference_mode():
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(pixel_values)
            else:
                features = self.model(pixel_values=pixel_values)
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state.mean(dim=1)
        elif isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim != 2:
            raise RuntimeError(f"Expected [batch, dim] VideoMAEv2 features, got {tuple(features.shape)}")
        return features.float().cpu().numpy()


def extract_recording(
    recording: IndustRealRecording,
    encoder: ClipEncoder,
    output_path: str | Path,
    stride_frames: int = 5,
    num_frames: int = 16,
    sampling_rate: int = 2,
    batch_size: int = 1,
) -> ExtractionRecord:
    """Extract features aligned to StateGraph cache centers for one recording."""
    if stride_frames <= 0 or batch_size <= 0:
        raise ValueError("stride_frames and batch_size must be positive")
    import cv2

    frame_paths = _frame_paths(recording)
    capture = None
    if frame_paths:
        source_length = len(frame_paths)
    elif recording.video_path is not None:
        capture = cv2.VideoCapture(str(recording.video_path))
        source_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or source_length <= 0:
            raise RuntimeError(f"Could not open {recording.video_path}")
    else:
        raise RuntimeError(f"No RGB source for {recording.recording_id}")

    frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    cache_capacity = max(32, num_frames * sampling_rate + 8)

    def read_frame(index: int) -> np.ndarray:
        if index in frame_cache:
            frame_cache.move_to_end(index)
            return frame_cache[index]
        if frame_paths:
            frame = cv2.imread(str(frame_paths[index]))
        else:
            assert capture is not None
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                frame = None
        if frame is None:
            raise RuntimeError(f"Failed to decode frame {index} from {recording.recording_id}")
        frame_cache[index] = frame
        if len(frame_cache) > cache_capacity:
            frame_cache.popitem(last=False)
        return frame

    centers = np.arange(0, source_length, stride_frames, dtype=np.int64)
    rows: list[np.ndarray] = []
    for start in range(0, len(centers), batch_size):
        clips = []
        for center in centers[start : start + batch_size]:
            indices = centered_clip_indices(int(center), source_length, num_frames, sampling_rate)
            clips.append([read_frame(int(index)) for index in indices])
        rows.append(encoder.encode(clips))
    if capture is not None:
        capture.release()
    features = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    if features.shape != (len(centers), encoder.feature_dim):
        raise RuntimeError(
            f"Feature alignment failed for {recording.recording_id}: {features.shape} "
            f"!= {(len(centers), encoder.feature_dim)}"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, features, allow_pickle=False)
    os.replace(temporary, path)
    return ExtractionRecord(
        recording_id=recording.recording_id,
        split=recording.split,
        path=str(path),
        num_steps=len(features),
        feature_dim=features.shape[1],
        source_frames=source_length,
    )


def _frame_paths(recording: IndustRealRecording) -> list[Path]:
    if recording.rgb_dir is None:
        return []
    paths = [
        path
        for path in recording.rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(paths, key=lambda path: _natural_key(path.stem))


def _natural_key(value: str) -> tuple[Any, ...]:
    import re

    return tuple(int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value))


def _eligible_recordings(data_root: str, selected: Iterable[str] | None) -> list[IndustRealRecording]:
    recordings = [
        recording
        for recording in discover_industreal_recordings(data_root)
        if recording.rgb_dir is not None or recording.video_path is not None
    ]
    if selected:
        requested = set(selected)
        recordings = [recording for recording in recordings if recording.recording_id in requested]
        missing = requested - {recording.recording_id for recording in recordings}
        if missing:
            raise RuntimeError(f"Recording IDs were not found: {sorted(missing)}")
    return recordings


def shard_items(items: list[Any], shard_index: int, num_shards: int) -> list[Any]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    return items[shard_index::num_shards]


def _read_completed_manifests(output_dir: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(output_dir.glob("videomaev2_manifest*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            completed[row["recording_id"]] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cache-aligned frozen VideoMAEv2 features.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned Hugging Face model revision for reproducible remote code and weights.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional official fine-tuned .pth (preferred: vit_g_hybrid_pt_1200e_ssv2_ft).",
    )
    parser.add_argument("--recording-id", action="append", default=None)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sampling-rate", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    recordings = _eligible_recordings(args.data_root, args.recording_id)
    if args.max_recordings is not None:
        if args.max_recordings <= 0:
            parser.error("--max-recordings must be positive")
        recordings = recordings[: args.max_recordings]
    try:
        recordings = shard_items(recordings, args.shard_index, args.num_shards)
    except ValueError as exc:
        parser.error(str(exc))
    if not recordings:
        raise RuntimeError("No RGB recordings found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / (
        "videomaev2_manifest.json"
        if args.num_shards == 1
        else f"videomaev2_manifest.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    )
    completed = {} if args.overwrite else _read_completed_manifests(output_dir)
    shard_records: dict[str, dict[str, Any]] = {}

    encoder: HuggingFaceVideoMAEv2Encoder | None = None
    for ordinal, recording in enumerate(recordings, start=1):
        output_path = output_dir / recording.split / f"{recording.recording_id}.npy"
        previous = completed.get(recording.recording_id)
        if not args.overwrite and previous and output_path.is_file():
            print(f"[{ordinal}/{len(recordings)}] skip {recording.recording_id}", flush=True)
            shard_records[recording.recording_id] = previous
            continue
        if encoder is None:
            encoder = HuggingFaceVideoMAEv2Encoder(
                args.model_id, args.revision, args.checkpoint, args.device, args.precision
            )
        print(f"[{ordinal}/{len(recordings)}] extract {recording.recording_id}", flush=True)
        record = extract_recording(
            recording,
            encoder,
            output_path,
            args.stride_frames,
            args.num_frames,
            args.sampling_rate,
            args.batch_size,
        )
        completed[recording.recording_id] = asdict(record)
        shard_records[recording.recording_id] = asdict(record)
        manifest = {
            "backend": "videomaev2_ssv2_finetuned" if args.checkpoint else "videomaev2_unlabeledhybrid",
            "model_id": args.model_id,
            "revision": args.revision,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "stride_frames": args.stride_frames,
            "num_frames": args.num_frames,
            "sampling_rate": args.sampling_rate,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "records": sorted(shard_records.values(), key=lambda row: (row["split"], row["recording_id"])),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Also persist all-skip shards and make their provenance explicit.
    if not manifest_path.is_file() or not shard_records:
        manifest = {
            "backend": "videomaev2_ssv2_finetuned" if args.checkpoint else "videomaev2_unlabeledhybrid",
            "model_id": args.model_id,
            "revision": args.revision,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "stride_frames": args.stride_frames,
            "num_frames": args.num_frames,
            "sampling_rate": args.sampling_rate,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "records": sorted(shard_records.values(), key=lambda row: (row["split"], row["recording_id"])),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "recordings": len(shard_records)}, indent=2))


if __name__ == "__main__":
    main()
