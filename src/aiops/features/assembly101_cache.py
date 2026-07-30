from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.assembly101 import dense_assembly_targets, read_mistake_segments
from aiops.data.procedure_schema import CACHE_SCHEMA_VERSION, EVENT_OUTCOME_NAMES
from aiops.data.stategraph_cache import StateGraphCacheRecord, save_cache_record, write_cache_index


class BatchedConvNeXtExtractor:
    """Frozen ImageNet features for an Assembly101 frame archive.

    Motion is represented by the causal difference between adjacent appearance
    embeddings.  This is much cheaper than running a second video backbone and
    remains a genuine visual signal rather than an annotation-derived feature.
    """

    output_dim = 768

    def __init__(self, device: str | None = None, precision: str = "bf16") -> None:
        import torch
        import torch.nn as nn
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Identity()
        self.model.eval().to(self.device)

    def extract(self, encoded_images: list[bytes], batch_size: int = 64) -> np.ndarray:
        import cv2

        rows: list[np.ndarray] = []
        for offset in range(0, len(encoded_images), batch_size):
            images = []
            for payload in encoded_images[offset : offset + batch_size]:
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("OpenCV failed to decode an Assembly101 frame")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (236, 236), interpolation=cv2.INTER_LINEAR)
                start = (236 - 224) // 2
                image = image[start : start + 224, start : start + 224]
                tensor = self.torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
                mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
                std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
                images.append((tensor - mean) / std)
            batch = self.torch.stack(images).to(self.device, non_blocking=True)
            enabled = self.device.type == "cuda" and self.precision in {"bf16", "fp16"}
            dtype = self.torch.bfloat16 if self.precision == "bf16" else self.torch.float16
            with self.torch.inference_mode(), self.torch.autocast(
                device_type=self.device.type, dtype=dtype, enabled=enabled
            ):
                features = self.model(batch)
            rows.append(features.float().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(np.float32)


def build_assembly101_cache(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_payload = list(manifest["records"])
    if args.max_recordings is not None:
        records_payload = records_payload[: args.max_recordings]
    if not records_payload:
        raise ValueError("Assembly101 manifest contains no records")
    data_root = manifest_path.parent
    annotation_root = data_root / "annotations" / "mistake-detection"
    by_recording = {
        row["recording_id"]: read_mistake_segments(annotation_root / f"{row['recording_id']}.csv")
        for row in records_payload
    }
    all_segments = [segment for values in by_recording.values() for segment in values]
    background_id = "__background__"
    action_ids = [background_id] + sorted({segment.action_id for segment in all_segments})
    action_to_index = {name: index for index, name in enumerate(action_ids)}
    description_by_id = {background_id: "background"}
    for segment in all_segments:
        description_by_id.setdefault(segment.action_id, segment.action_description)
    action_descriptions = [description_by_id[name] for name in action_ids]
    components = sorted({segment.subject for segment in all_segments})
    component_to_index = {name: index for index, name in enumerate(components)}

    extractor = BatchedConvNeXtExtractor(args.device, args.precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_records: list[StateGraphCacheRecord] = []
    for index, row in enumerate(records_payload, start=1):
        recording_id = row["recording_id"]
        archive = data_root / row["archive_relative_path"]
        if not archive.is_file():
            raise FileNotFoundError(f"Missing Assembly101 frame archive: {archive}")
        print(f"[{index}/{len(records_payload)}] {row['split']}/{recording_id}", flush=True)
        frame_names, frame_indices = _archive_frames(
            archive,
            annotation_fps=args.annotation_fps,
            mirror_fps=args.mirror_fps,
        )
        selected_positions = _sample_positions(frame_indices, args.stride_frames)
        sampled_frames = frame_indices[selected_positions]
        with zipfile.ZipFile(archive) as handle:
            encoded = [handle.read(frame_names[position]) for position in selected_positions]
        appearance = extractor.extract(encoded, args.feature_batch_size)
        motion = np.zeros_like(appearance)
        if len(motion) > 1:
            motion[1:] = appearance[1:] - appearance[:-1]
        targets = dense_assembly_targets(
            sampled_frames,
            by_recording[recording_id],
            action_to_index,
            component_to_index,
            background_index=action_to_index[background_id],
        )
        sensor = np.zeros((len(sampled_frames), 1), dtype=np.float32)
        modality_mask = np.ones((len(sampled_frames), 3), dtype=np.bool_)
        modality_mask[:, 2] = False
        destination = output_dir / row["split"] / f"{recording_id}.npz"
        save_cache_record(
            destination,
            motion=motion,
            appearance=appearance,
            sensor=sensor,
            modality_mask=modality_mask,
            timestamps=sampled_frames.astype(np.float32) / float(args.annotation_fps),
            **targets,
        )
        cache_records.append(
            StateGraphCacheRecord(
                recording_id=recording_id,
                split=row["split"],
                path=destination,
                num_steps=len(action_ids),
                motion_dim=extractor.output_dim,
                appearance_dim=extractor.output_dim,
                sensor_dim=1,
                num_components=len(components),
                num_completion_components=len(components),
                num_frames=len(sampled_frames),
            )
        )

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "procedure_schema": "assembly101_part_to_part_v1",
        "dataset": "Assembly101",
        "label_contract": "action_segmentation_plus_multicomponent_completion",
        "dataset_label_source": "coarse_action_plus_part_to_part_mistake_and_state",
        "event_outcomes": list(EVENT_OUTCOME_NAMES),
        "completion_components": components,
        "state_components": components,
        "action_ids": action_ids,
        "action_descriptions": action_descriptions,
        "background_action_id": background_id,
        "state_names": ["incorrect", "not_completed", "correct"],
        "fps": args.annotation_fps,
        "stride_frames": args.stride_frames,
        "feature_backends": {
            "appearance": "torchvision_convnext_tiny_imagenet1k_v1",
            "motion": "causal_convnext_feature_difference",
            "sensor": None,
        },
        "visual_source": manifest["mirror"],
        "official_release": manifest["official_release"],
        "selection": manifest["selection"],
        "label_provenance": manifest["label_provenance"],
    }
    index_path = output_dir / "index.json"
    write_cache_index(index_path, cache_records, metadata)
    return {
        "index": str(index_path),
        "recordings": len(cache_records),
        "actions": len(action_ids),
        "components": len(components),
        "sampled_frames": sum(record.num_frames for record in cache_records),
    }


def _archive_frames(
    path: Path, *, annotation_fps: float = 30.0, mirror_fps: float = 1.0
) -> tuple[list[str], np.ndarray]:
    with zipfile.ZipFile(path) as handle:
        candidates = [
            name
            for name in handle.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    indexed = sorted((_frame_index(name), name) for name in candidates)
    if not indexed:
        raise RuntimeError(f"No image frames found in {path}")
    # The public mirror stores one frame per second and renumbers it from zero;
    # official mistake timestamps remain raw-video frame indices at 30 fps.
    # Convert into the annotation clock before target projection.
    scale = annotation_fps / mirror_fps
    frame_indices = np.rint([index * scale for index, _ in indexed]).astype(np.int64)
    return [name for _, name in indexed], frame_indices


def _sample_positions(frame_indices: np.ndarray, stride_frames: int) -> np.ndarray:
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")
    targets = np.arange(frame_indices[0], frame_indices[-1] + 1, stride_frames)
    positions = np.searchsorted(frame_indices, targets)
    positions = np.clip(positions, 0, len(frame_indices) - 1)
    previous = np.maximum(positions - 1, 0)
    use_previous = np.abs(frame_indices[previous] - targets) < np.abs(frame_indices[positions] - targets)
    positions[use_previous] = previous[use_previous]
    return np.unique(positions).astype(np.int64)


def _frame_index(name: str) -> int:
    values = re.findall(r"\d+", Path(name).stem)
    if not values:
        raise ValueError(f"Frame filename has no numeric index: {name!r}")
    return int(values[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build StateGraph cache from Assembly101 frames.")
    parser.add_argument("--manifest", default="data/raw/assembly101/subset_manifest.json")
    parser.add_argument("--output-dir", default="data/processed/assembly101_stategraph")
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--mirror-fps", type=float, default=1.0)
    parser.add_argument("--stride-frames", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-recordings", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build_assembly101_cache(args), indent=2))


if __name__ == "__main__":
    main()
