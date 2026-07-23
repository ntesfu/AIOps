from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.industreal import (
    IndustRealRecording,
    attach_industreal_auxiliary,
    discover_industreal_recordings,
)
from aiops.data.stategraph_cache import read_cache_index


ROI_NAMES = ("left_hand", "right_hand", "gaze", "active_object")
ROI_FEATURE_VERSION = 1


class FrozenROIEncoder:
    """Frozen ConvNeXt crop encoder with bounded GPU batches."""

    def __init__(self, device: str | None = None, precision: str = "bf16") -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        except ImportError as exc:  # pragma: no cover - GPU environment only
            raise RuntimeError("Install the vision dependencies to extract ROI features.") from exc
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.model.classifier[-1] = nn.Identity()
        self.model.eval().to(self.device)
        self.output_dim = 768

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.output_dim), dtype=np.float32)
        torch = self.torch
        batch = np.stack([_normalize_crop(crop) for crop in crops])
        tensor = torch.from_numpy(batch).to(self.device)
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            autocast = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            from contextlib import nullcontext

            autocast = nullcontext()
        with torch.inference_mode(), autocast:
            features = self.model(tensor)
        return features.float().cpu().numpy()


def build_roi_features(args: argparse.Namespace) -> dict[str, Any]:
    _, cache_records = read_cache_index(args.cache_index)
    recordings = discover_industreal_recordings(args.data_root)
    if args.auxiliary_root:
        recordings = attach_industreal_auxiliary(recordings, args.auxiliary_root)
    recordings_by_id = {recording.recording_id: recording for recording in recordings}
    selected = cache_records[: args.max_recordings] if args.max_recordings else cache_records
    missing = sorted(
        record.recording_id for record in selected if record.recording_id not in recordings_by_id
    )
    if missing:
        raise RuntimeError(f"Cache recordings are missing from the dataset root: {missing}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = FrozenROIEncoder(args.device, args.precision)
    manifest_records: list[dict[str, Any]] = []
    for ordinal, cache_record in enumerate(selected, start=1):
        recording = recordings_by_id[cache_record.recording_id]
        destination = output_root / cache_record.split / f"{cache_record.recording_id}.npz"
        print(f"[{ordinal}/{len(selected)}] {cache_record.split}/{cache_record.recording_id}", flush=True)
        report = _extract_recording(
            cache_record.path,
            recording,
            destination,
            encoder,
            batch_size=args.batch_size,
            gaze_crop_size=args.gaze_crop_size,
            hand_box_scale=args.hand_box_scale,
            resume=args.resume,
        )
        manifest_records.append(report)
        _write_manifest(output_root, args, encoder.output_dim, manifest_records)
    return {
        "output_dir": str(output_root),
        "recordings": len(manifest_records),
        "feature_dim": manifest_records[0]["feature_dim"] if manifest_records else 0,
        "causal": True,
    }


def _extract_recording(
    cache_path: Path,
    recording: IndustRealRecording,
    destination: Path,
    encoder: FrozenROIEncoder,
    *,
    batch_size: int,
    gaze_crop_size: int,
    hand_box_scale: float,
    resume: bool,
) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as arrays:
        if "prediction_frame_indices" not in arrays.files:
            raise ValueError(f"{cache_path} lacks prediction_frame_indices provenance")
        prediction_frames = arrays["prediction_frame_indices"].astype(np.int64)
    feature_dim = len(ROI_NAMES) * encoder.output_dim + len(ROI_NAMES) + 16 + 3
    if resume and destination.is_file():
        with np.load(destination, allow_pickle=False) as arrays:
            if (
                arrays["motion"].shape == (len(prediction_frames), feature_dim)
                and np.array_equal(arrays["prediction_frame_indices"], prediction_frames)
            ):
                return _roi_record_report(
                    recording, destination, arrays["roi_mask"], feature_dim
                )

    if recording.video_path is None:
        raise RuntimeError(f"No MP4/video source found for {recording.recording_id}")
    hands = _read_hand_points(recording.hands)
    gaze = _read_gaze(recording.gaze)
    objects = _read_object_detections(recording.object_labels)

    import cv2

    capture = cv2.VideoCapture(str(recording.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {recording.video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    embeddings = np.zeros(
        (len(prediction_frames), len(ROI_NAMES), encoder.output_dim), dtype=np.float32
    )
    roi_mask = np.zeros((len(prediction_frames), len(ROI_NAMES)), dtype=np.bool_)
    roi_boxes = np.zeros((len(prediction_frames), len(ROI_NAMES), 4), dtype=np.float32)
    category_ids = np.full(len(prediction_frames), -1, dtype=np.int64)
    gaze_points = np.zeros((len(prediction_frames), 2), dtype=np.float32)
    pending_crops: list[np.ndarray] = []
    pending_positions: list[tuple[int, int]] = []

    def flush() -> None:
        if not pending_crops:
            return
        encoded = encoder.encode(pending_crops)
        for feature, (row, roi_index) in zip(encoded, pending_positions):
            embeddings[row, roi_index] = feature
        pending_crops.clear()
        pending_positions.clear()

    current_frame = -1
    frame = None
    for row, target_frame in enumerate(prediction_frames.tolist()):
        if target_frame < current_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            current_frame = target_frame - 1
        while current_frame < target_frame:
            ok, frame = capture.read()
            current_frame += 1
            if not ok or frame is None:
                capture.release()
                raise RuntimeError(
                    f"Failed to decode frame {target_frame} from {recording.video_path}"
                )
        assert frame is not None
        point = gaze.get(target_frame)
        if point is not None:
            gaze_points[row] = (point[0] / max(width, 1), point[1] / max(height, 1))
        left_points, right_points = hands.get(target_frame, (None, None))
        boxes: list[tuple[float, float, float, float] | None] = [
            _points_box(left_points, width, height, hand_box_scale),
            _points_box(right_points, width, height, hand_box_scale),
            _center_box(point, gaze_crop_size, width, height) if point is not None else None,
            None,
        ]
        object_detection = _select_active_object(
            objects.get(target_frame, []), point, left_points, right_points
        )
        if object_detection is not None:
            boxes[3] = _clip_box(object_detection["bbox"], width, height)
            category_ids[row] = int(object_detection["category_id"])
        for roi_index, box in enumerate(boxes):
            if box is None:
                continue
            crop = _crop_frame(frame, box)
            if crop is None:
                continue
            roi_mask[row, roi_index] = True
            roi_boxes[row, roi_index] = _normalize_box(box, width, height)
            pending_crops.append(crop)
            pending_positions.append((row, roi_index))
            if len(pending_crops) >= batch_size:
                flush()
    flush()
    capture.release()

    geometry = roi_boxes.reshape(len(prediction_frames), -1)
    normalized_category = np.maximum(category_ids, 0).astype(np.float32)[:, None] / 100.0
    features = np.concatenate(
        [
            embeddings.reshape(len(prediction_frames), -1),
            roi_mask.astype(np.float32),
            geometry,
            normalized_category,
            gaze_points,
        ],
        axis=1,
    ).astype(np.float32)
    source_frames = prediction_frames[:, None]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        motion=features,
        roi_mask=roi_mask,
        roi_boxes=roi_boxes,
        object_category_ids=category_ids,
        prediction_frame_indices=prediction_frames,
        source_frame_indices=source_frames,
    )
    temporary.replace(destination)
    return _roi_record_report(recording, destination, roi_mask, feature_dim)


def _roi_record_report(
    recording: IndustRealRecording,
    destination: Path,
    roi_mask: np.ndarray,
    feature_dim: int,
) -> dict[str, Any]:
    return {
        "recording_id": recording.recording_id,
        "split": recording.split,
        "path": str(destination),
        "rows": len(roi_mask),
        "feature_dim": feature_dim,
        "roi_coverage": {
            name: float(roi_mask[:, index].mean())
            for index, name in enumerate(ROI_NAMES)
        },
        "max_future_offset_frames": 0,
        "video_path": str(recording.video_path),
        "object_labels": str(recording.object_labels) if recording.object_labels else None,
        "hands": str(recording.hands) if recording.hands else None,
        "gaze": str(recording.gaze) if recording.gaze else None,
    }


def _write_manifest(
    output_root: Path,
    args: argparse.Namespace,
    embedding_dim: int,
    records: list[dict[str, Any]],
) -> None:
    payload = {
        "format_version": ROI_FEATURE_VERSION,
        "backend": "convnext_tiny_imagenet1k_v1",
        "roi_names": list(ROI_NAMES),
        "embedding_dim_per_roi": embedding_dim,
        "causal": True,
        "source_contract": "current_prediction_frame_only",
        "max_future_offset_frames": 0,
        "cache_index": str(Path(args.cache_index).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "auxiliary_root": str(Path(args.auxiliary_root).resolve())
        if args.auxiliary_root
        else None,
        "records": records,
    }
    (output_root / "roi_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _read_hand_points(
    path: Path | None,
) -> dict[int, tuple[np.ndarray | None, np.ndarray | None]]:
    if path is None:
        return {}
    result: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for ordinal, row in enumerate(csv.reader(handle)):
            if len(row) < 5:
                continue
            frame = _frame_index(row[0], ordinal)
            try:
                values = np.asarray([float(value) for value in row[1:]], dtype=np.float32)
            except ValueError:
                continue
            midpoint = len(values) // 2
            left = _reshape_points(values[:midpoint])
            right = _reshape_points(values[midpoint:])
            result[frame] = (left, right)
    return result


def _read_gaze(path: Path | None) -> dict[int, tuple[float, float]]:
    if path is None:
        return {}
    result: dict[int, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for ordinal, row in enumerate(csv.reader(handle)):
            if len(row) < 3:
                continue
            try:
                result[_frame_index(row[0], ordinal)] = (float(row[1]), float(row[2]))
            except ValueError:
                continue
    return result


def _read_object_detections(path: Path | None) -> dict[int, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_frames = {
        int(image["id"]): _frame_index(str(image.get("file_name", image["id"])), int(image["id"]))
        for image in payload.get("images", [])
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        image_id = int(annotation.get("image_id", -1))
        if image_id not in image_frames:
            continue
        x, y, width, height = (float(value) for value in bbox)
        grouped[image_frames[image_id]].append(
            {
                "bbox": (x, y, x + width, y + height),
                "category_id": int(annotation.get("category_id", -1)),
                "score": float(annotation.get("score", 1.0)),
            }
        )
    return dict(grouped)


def _select_active_object(
    detections: list[dict[str, Any]],
    gaze: tuple[float, float] | None,
    left_points: np.ndarray | None,
    right_points: np.ndarray | None,
) -> dict[str, Any] | None:
    if not detections:
        return None
    anchors: list[tuple[float, float]] = []
    if gaze is not None:
        anchors.append(gaze)
    for points in (left_points, right_points):
        if points is not None and len(points):
            anchors.append(tuple(points.mean(axis=0).tolist()))
    if not anchors:
        return max(
            detections,
            key=lambda item: (item["bbox"][2] - item["bbox"][0])
            * (item["bbox"][3] - item["bbox"][1]),
        )

    def distance(item: dict[str, Any]) -> float:
        x1, y1, x2, y2 = item["bbox"]
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return min((center[0] - x) ** 2 + (center[1] - y) ** 2 for x, y in anchors)

    return min(detections, key=distance)


def _reshape_points(values: np.ndarray) -> np.ndarray | None:
    if len(values) < 4:
        return None
    values = values[: len(values) // 2 * 2].reshape(-1, 2)
    finite = np.isfinite(values).all(axis=1)
    positive = (values[:, 0] >= 0) & (values[:, 1] >= 0)
    selected = values[finite & positive]
    return selected if len(selected) else None


def _points_box(
    points: np.ndarray | None, width: int, height: int, scale: float
) -> tuple[float, float, float, float] | None:
    if points is None or not len(points):
        return None
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    size = np.maximum(maximum - minimum, 48.0) * scale
    return _clip_box(
        (
            center[0] - size[0] / 2.0,
            center[1] - size[1] / 2.0,
            center[0] + size[0] / 2.0,
            center[1] + size[1] / 2.0,
        ),
        width,
        height,
    )


def _center_box(
    point: tuple[float, float], size: int, width: int, height: int
) -> tuple[float, float, float, float]:
    return _clip_box(
        (
            point[0] - size / 2.0,
            point[1] - size / 2.0,
            point[0] + size / 2.0,
            point[1] + size / 2.0,
        ),
        width,
        height,
    )


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(x2, x1 + 1, max(width, 1)))
    y2 = float(np.clip(y2, y1 + 1, max(height, 1)))
    return x1, y1, x2, y2


def _crop_frame(
    frame: np.ndarray, box: tuple[float, float, float, float]
) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _normalize_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.asarray(
        [x1 / max(width, 1), y1 / max(height, 1), x2 / max(width, 1), y2 / max(height, 1)],
        dtype=np.float32,
    )


def _normalize_crop(crop: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)


def _frame_index(value: str, fallback: int) -> int:
    digits = "".join(character for character in Path(value).stem if character.isdigit())
    return int(digits) if digits else fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--auxiliary-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gaze-crop-size", type=int, default=320)
    parser.add_argument("--hand-box-scale", type=float, default=1.5)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(build_roi_features(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
