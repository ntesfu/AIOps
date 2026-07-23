from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.industreal import (
    IndustRealRecording,
    attach_industreal_auxiliary,
    audit_industreal_root,
    dense_action_labels,
    discover_industreal_recordings,
    multi_component_targets_from_events,
    read_action_segments,
    read_completion_events,
    read_numeric_timeseries,
    read_raw_states,
)
from aiops.data.procedure_schema import ProcedureSchema
from aiops.data.stategraph_cache import StateGraphCacheRecord, save_cache_record, write_cache_index


class FrozenVisualFeatureExtractor:
    """Sequential frozen encoders for low-VRAM cache creation.

    The video and image encoders are never used together in a differentiable
    graph. This keeps extraction practical on a 24 GB card and makes later head
    training much cheaper.
    """

    def __init__(self, device: str | None = None, mixed_precision: str = "bf16") -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
            from torchvision.models.video import Swin3D_S_Weights, swin3d_s
        except ImportError as exc:  # pragma: no cover - training environment only
            raise RuntimeError("Install the 'vision' extra to extract visual features.") from exc
        self.torch = torch
        self.nn = nn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mixed_precision = mixed_precision
        self.motion_model = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)
        self.motion_model.head = nn.Identity()
        self.motion_model.eval().to(self.device)
        self.appearance_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.appearance_model.classifier[-1] = nn.Identity()
        self.appearance_model.eval().to(self.device)
        self.motion_dim = 768
        self.appearance_dim = 768

    def _autocast(self):
        if self.device.type != "cuda":
            from contextlib import nullcontext

            return nullcontext()
        dtype = self.torch.bfloat16 if self.mixed_precision == "bf16" else self.torch.float16
        return self.torch.autocast(device_type="cuda", dtype=dtype)

    def extract_motion(self, frames: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        array = np.stack([_normalize_video_frame(frame) for frame in frames], axis=1)
        tensor = torch.from_numpy(array[None]).to(self.device)
        with torch.inference_mode(), self._autocast():
            feature = self.motion_model(tensor)
        return feature.float().cpu().numpy()[0]

    def extract_appearance(self, frame: np.ndarray) -> np.ndarray:
        torch = self.torch
        tensor = torch.from_numpy(_normalize_image_frame(frame)[None]).to(self.device)
        with torch.inference_mode(), self._autocast():
            feature = self.appearance_model(tensor)
        return feature.float().cpu().numpy()[0]


def build_industreal_cache(args: argparse.Namespace) -> dict[str, Any]:
    clip_anchor = getattr(args, "clip_anchor", "trailing")
    strict_causal = bool(getattr(args, "strict_causal", False))
    if clip_anchor not in {"trailing", "centered"}:
        raise ValueError("clip_anchor must be 'trailing' or 'centered'")
    recordings = discover_industreal_recordings(args.data_root)
    if getattr(args, "auxiliary_root", None):
        recordings = attach_industreal_auxiliary(recordings, args.auxiliary_root)
    official = [
        recording
        for recording in recordings
        if (recording.rgb_dir or recording.video_path) and recording.ar_labels and recording.psr_labels
    ]
    if args.recording_id:
        requested = set(args.recording_id)
        official = [recording for recording in official if recording.recording_id in requested]
        found = {recording.recording_id for recording in official}
        missing = sorted(requested - found)
        if missing:
            raise RuntimeError(f"Requested recordings were not found with RGB, AR, and PSR labels: {missing}")
    if args.max_recordings is not None:
        official = official[: args.max_recordings]
    if not official:
        audit = audit_industreal_root(args.data_root)
        raise RuntimeError(
            "No official IndustReal recordings with RGB, AR, and PSR annotations found.\n"
            + json.dumps(audit.to_dict(), indent=2)
        )
    motion_provenance = _precomputed_motion_provenance(
        args.motion_features_dir, {recording.recording_id for recording in official}
    )
    if strict_causal and args.motion_features_dir and not motion_provenance["causality_verified"]:
        raise RuntimeError(
            "--strict-causal requires every precomputed motion root to contain a "
            "VideoMAEv2 manifest and source-index sidecars proving trailing clips with no "
            "positive future offset for every selected recording. Re-extract with "
            "--clip-anchor trailing."
        )

    schema = ProcedureSchema.load(args.procedure_schema)
    if schema.state_components and len(schema.state_components) != args.num_components:
        raise ValueError(
            f"--num-components={args.num_components} does not match procedure schema "
            f"state_components={len(schema.state_components)}."
        )
    all_actions = [
        segment
        for recording in official
        for segment in read_action_segments(recording.ar_labels)  # type: ignore[arg-type]
    ]
    action_ids = sorted({segment.action_id for segment in all_actions}, key=_natural_key)
    if not action_ids:
        raise RuntimeError("AR annotation files were found but contained no readable action labels.")
    background_action_id = next(
        (segment.action_id for segment in all_actions if _is_background_action(segment)),
        "__background__",
    )
    if background_action_id not in action_ids:
        action_ids.insert(0, background_action_id)
    action_to_index = {action_id: index for index, action_id in enumerate(action_ids)}
    action_description_by_id: dict[str, str] = {}
    for segment in all_actions:
        action_description_by_id.setdefault(segment.action_id, segment.description)
    action_description_by_id.setdefault(background_action_id, "background")
    action_descriptions = [action_description_by_id[action_id] for action_id in action_ids]

    extractor = None
    if not args.motion_features_dir or not args.appearance_features_dir:
        extractor = FrozenVisualFeatureExtractor(args.device, args.mixed_precision)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[StateGraphCacheRecord] = []
    for recording_index, recording in enumerate(official, start=1):
        print(f"[{recording_index}/{len(official)}] {recording.split}/{recording.recording_id}", flush=True)
        record = _cache_recording(
            recording,
            action_to_index,
            background_action_id,
            schema,
            extractor,
            args,
            output_dir,
        )
        records.append(record)

    metadata = {
        **schema.to_metadata(),
        "label_contract": "action_segmentation_plus_multicomponent_completion",
        "action_ids": action_ids,
        "action_descriptions": action_descriptions,
        "background_action_id": background_action_id,
        "state_names": ["incorrect", "not_completed", "correct"],
        "fps": args.fps,
        "stride_frames": args.stride_frames,
        "clip_frames": args.clip_frames,
        "clip_span_frames": args.clip_span_frames,
        "clip_anchor": clip_anchor,
        "causal_clips": clip_anchor == "trailing" and motion_provenance["causality_verified"],
        "strict_causal": strict_causal,
        "auxiliary_root": str(args.auxiliary_root) if getattr(args, "auxiliary_root", None) else None,
        "temporal_contract": (
            "prediction-aligned_past-only"
            if clip_anchor == "trailing" and motion_provenance["causality_verified"]
            else "unverified-precomputed-motion"
            if clip_anchor == "trailing"
            else "prediction-centered_bounded-lookahead"
        ),
        "motion_provenance": motion_provenance,
        "feature_backends": {
            "motion": (
                getattr(args, "motion_backend_name", None)
                if args.motion_features_dir and getattr(args, "motion_backend_name", None)
                else "precomputed"
                if args.motion_features_dir
                else "torchvision_swin3d_s"
            ),
            "appearance": "precomputed" if args.appearance_features_dir else "torchvision_convnext_tiny",
            "motion_aux": (
                getattr(args, "motion_aux_backend_name", None) or "precomputed"
                if getattr(args, "motion_aux_features_dir", None)
                else None
            ),
        },
    }
    index_path = output_dir / "index.json"
    write_cache_index(index_path, records, metadata)
    return {
        "index": str(index_path),
        "recordings": len(records),
        "actions": action_ids,
        "completion_components": list(schema.completion_components),
    }


def _cache_recording(
    recording: IndustRealRecording,
    action_to_index: dict[str, int],
    background_action_id: str,
    schema: ProcedureSchema,
    extractor: FrozenVisualFeatureExtractor | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> StateGraphCacheRecord:
    frame_paths: list[Path] = []
    video_capture = None
    if recording.rgb_dir is not None:
        frame_paths = sorted(recording.rgb_dir.glob("*"), key=lambda path: _frame_index(path.name))
        frame_paths = [path for path in frame_paths if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not frame_paths:
            raise RuntimeError(f"No RGB frames found in {recording.rgb_dir}")
        all_frame_indices = np.asarray([_frame_index(path.name) for path in frame_paths], dtype=np.int64)
    elif recording.video_path is not None:
        import cv2

        video_capture = cv2.VideoCapture(str(recording.video_path))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not video_capture.isOpened() or frame_count <= 0:
            raise RuntimeError(f"Could not open RGB video {recording.video_path}")
        all_frame_indices = np.arange(frame_count, dtype=np.int64)
    else:
        raise RuntimeError(f"No RGB source found for {recording.recording_id}")

    source_length = len(all_frame_indices)
    centers = np.arange(0, len(frame_paths), args.stride_frames, dtype=np.int64)
    if video_capture is not None:
        centers = np.arange(0, source_length, args.stride_frames, dtype=np.int64)
    sampled_frame_indices = all_frame_indices[centers]
    actions = read_action_segments(recording.ar_labels)  # type: ignore[arg-type]
    step, boundary = dense_action_labels(
        sampled_frame_indices,
        actions,
        action_to_index,
        background_index=action_to_index[background_action_id],
    )
    events = read_completion_events(recording.psr_labels)  # type: ignore[arg-type]
    completion, component_outcome = multi_component_targets_from_events(
        sampled_frame_indices,
        events,
        schema.resolve_component,
        len(schema.completion_components),
        schema.event_outcomes,
    )
    valid_rows = step >= 0
    if not valid_rows.any():
        raise RuntimeError(f"No dense AR action labels could be derived for {recording.recording_id}")

    motion = _load_precomputed_group(
        args.motion_features_dir,
        recording.recording_id,
        len(centers),
        preferred_key="motion",
        allow_resample=not bool(getattr(args, "strict_causal", False)),
    )
    appearance = _load_precomputed(
        args.appearance_features_dir,
        recording.recording_id,
        len(centers),
        preferred_key="appearance",
        allow_resample=not bool(getattr(args, "strict_causal", False)),
    )
    motion_aux = _load_precomputed(
        getattr(args, "motion_aux_features_dir", None),
        recording.recording_id,
        len(centers),
        preferred_key="motion",
        allow_resample=not bool(getattr(args, "strict_causal", False)),
    )
    clip_anchor = getattr(args, "clip_anchor", "trailing")
    nominal_motion_source_positions = np.stack(
        [
            _clip_indices(
                int(prediction), source_length, args.clip_frames, args.clip_span_frames, clip_anchor
            )
            for prediction in centers
        ]
    )
    precomputed_source_positions = _load_precomputed_source_positions(
        args.motion_features_dir, recording, centers
    )
    motion_source_positions = (
        precomputed_source_positions
        if precomputed_source_positions is not None
        else nominal_motion_source_positions
        if not args.motion_features_dir
        else None
    )
    if motion is None or appearance is None:
        if extractor is None:
            raise RuntimeError("A visual extractor is required when either cached modality is missing.")
        import cv2

        frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        def read_frame(frame_index: int) -> np.ndarray:
            if frame_index in frame_cache:
                frame_cache.move_to_end(frame_index)
                return frame_cache[frame_index]
            if frame_paths:
                frame = cv2.imread(str(frame_paths[frame_index]))
            else:
                assert video_capture is not None
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = video_capture.read()
                if not ok:
                    frame = None
            if frame is None:
                source = frame_paths[frame_index] if frame_paths else recording.video_path
                raise RuntimeError(f"Failed to decode RGB frame {frame_index} from {source}")
            frame_cache[frame_index] = frame
            if len(frame_cache) > max(64, args.clip_span_frames * 4):
                frame_cache.popitem(last=False)
            return frame

        motion_rows = [] if motion is None else None
        appearance_rows = [] if appearance is None else None
        for center, clip_indices in zip(centers, nominal_motion_source_positions):
            if motion_rows is not None:
                clip_frames = [read_frame(index) for index in clip_indices]
                motion_rows.append(extractor.extract_motion(clip_frames))
            if appearance_rows is not None:
                frame = read_frame(int(center))
                appearance_rows.append(extractor.extract_appearance(frame))
        if motion_rows is not None:
            motion = np.stack(motion_rows).astype(np.float32)
        if appearance_rows is not None:
            appearance = np.stack(appearance_rows).astype(np.float32)

    if video_capture is not None:
        video_capture.release()

    sensor, sensor_present = _sample_sensors(recording, sampled_frame_indices, args.sensor_dim)
    modality_mask = np.ones((len(centers), 3), dtype=np.bool_)
    modality_mask[:, 2] = sensor_present

    state = np.ones((len(centers), args.num_components), dtype=np.int64)
    state_mask = np.zeros_like(state, dtype=np.bool_)
    if recording.psr_raw_labels:
        raw_states = read_raw_states(recording.psr_raw_labels, args.num_components)
        state, state_mask = _sample_carry_forward(raw_states, sampled_frame_indices, args.num_components)

    # Filenames carry the dataset frame index; array positions do not when
    # frames are missing or the sequence starts at a non-zero frame.
    timestamps = sampled_frame_indices.astype(np.float32) / float(args.fps)
    destination = output_dir / recording.split / f"{recording.recording_id}.npz"
    save_cache_record(
        destination,
        motion=motion,
        appearance=appearance,
        sensor=sensor,
        modality_mask=modality_mask,
        step=step,
        completion=completion,
        component_outcome=component_outcome,
        state=state,
        state_mask=state_mask,
        boundary=boundary,
        timestamps=timestamps,
        motion_aux=motion_aux,
        prediction_frame_indices=sampled_frame_indices,
        motion_source_frame_indices=(
            all_frame_indices[motion_source_positions]
            if motion_source_positions is not None
            else None
        ),
    )
    return StateGraphCacheRecord(
        recording_id=recording.recording_id,
        split=recording.split,
        path=destination,
        num_steps=len(action_to_index),
        motion_dim=int(motion.shape[1]),
        appearance_dim=int(appearance.shape[1]),
        sensor_dim=int(sensor.shape[1]),
        num_components=args.num_components,
        num_completion_components=len(schema.completion_components),
        num_frames=len(step),
        motion_aux_dim=int(motion_aux.shape[1]) if motion_aux is not None else 0,
    )


def _sample_sensors(
    recording: IndustRealRecording, frame_indices: np.ndarray, sensor_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    maps = [
        read_numeric_timeseries(path)
        # Gaze is intentionally excluded from the StateVerify contract. Hand
        # joints and head pose are stable interaction/motion evidence, while
        # RGB/object/depth streams provide the component state evidence.
        for path in (recording.hands, recording.pose)
        if path is not None
    ]
    rows = np.zeros((len(frame_indices), sensor_dim), dtype=np.float32)
    present = np.zeros(len(frame_indices), dtype=np.bool_)
    for row_index, frame_index in enumerate(frame_indices):
        values = []
        for mapping in maps:
            item = _nearest_item(mapping, int(frame_index))
            if item is not None:
                values.extend(item.tolist())
        if values:
            array = np.asarray(values[:sensor_dim], dtype=np.float32)
            rows[row_index, : len(array)] = array
            present[row_index] = True
    return rows, present


def _sample_carry_forward(
    mapping: dict[int, np.ndarray], frame_indices: np.ndarray, num_components: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.ones((len(frame_indices), num_components), dtype=np.int64)
    mask = np.zeros_like(values, dtype=np.bool_)
    keys = np.asarray(sorted(mapping), dtype=np.int64)
    if keys.size == 0:
        return values, mask
    for index, frame in enumerate(frame_indices):
        position = int(np.searchsorted(keys, frame, side="right") - 1)
        if position >= 0:
            values[index] = mapping[int(keys[position])]
            mask[index] = True
    return values, mask


def _nearest_item(mapping: dict[int, np.ndarray], frame_index: int) -> np.ndarray | None:
    if not mapping:
        return None
    if frame_index in mapping:
        return mapping[frame_index]
    key = min(mapping, key=lambda value: abs(value - frame_index))
    return mapping[key]


def _load_precomputed(
    directory: str | None,
    recording_id: str,
    length: int,
    preferred_key: str = "features",
    allow_resample: bool = True,
) -> np.ndarray | None:
    if not directory:
        return None
    root = Path(directory)
    candidates = [root / f"{recording_id}.npy", root / f"{recording_id}.npz"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None and root.is_dir():
        path = next(
            (
                candidate
                for suffix in (".npy", ".npz")
                for candidate in root.rglob(f"{recording_id}{suffix}")
            ),
            None,
        )
    if path is None:
        return None
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as payload:
            key = (
                preferred_key
                if preferred_key in payload.files
                else "features"
                if "features" in payload.files
                else payload.files[0]
            )
            array = payload[key]
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [time, dim] features in {path}, got {array.shape}")
    if len(array) != length:
        if not allow_resample:
            raise ValueError(
                f"Strict temporal alignment requires {length} rows in {path}, got {len(array)}; "
                "index resampling is disabled."
            )
        positions = np.linspace(0, len(array) - 1, length).round().astype(np.int64)
        array = array[positions]
    return array


def _load_precomputed_group(
    directories: str | list[str] | tuple[str, ...] | None,
    recording_id: str,
    length: int,
    preferred_key: str = "features",
    allow_resample: bool = True,
) -> np.ndarray | None:
    """Load one feature source or concatenate several aligned sources."""
    if not directories:
        return None
    roots = [directories] if isinstance(directories, str) else list(directories)
    arrays = [
        _load_precomputed(
            root,
            recording_id,
            length,
            preferred_key=preferred_key,
            allow_resample=allow_resample,
        )
        for root in roots
    ]
    missing = [root for root, array in zip(roots, arrays) if array is None]
    if missing:
        raise FileNotFoundError(
            f"Missing precomputed {preferred_key} features for {recording_id} in {missing}"
        )
    return np.concatenate([array for array in arrays if array is not None], axis=1)


def _clip_indices(
    prediction_frame: int,
    length: int,
    clip_frames: int,
    span_frames: int,
    anchor: str = "trailing",
) -> list[int]:
    if min(length, clip_frames, span_frames) <= 0:
        raise ValueError("length, clip_frames, and span_frames must be positive")
    if anchor == "trailing":
        start = prediction_frame - span_frames + 1
    elif anchor == "centered":
        start = prediction_frame - span_frames // 2
    else:
        raise ValueError("anchor must be 'trailing' or 'centered'")
    positions = np.linspace(start, start + span_frames - 1, clip_frames).round().astype(np.int64)
    return np.clip(positions, 0, length - 1).tolist()


def _precomputed_motion_provenance(
    directories: str | list[str] | tuple[str, ...] | None,
    required_recording_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize causal declarations from VideoMAEv2 feature manifests."""
    if not directories:
        return {"kind": "native_extraction", "causality_verified": True, "sources": []}
    roots = [directories] if isinstance(directories, str) else list(directories)
    provenance_sources: list[dict[str, Any]] = []
    verified = True
    for directory in roots:
        root = Path(directory)
        manifests = sorted(root.glob("videomaev2_manifest*.json")) if root.is_dir() else []
        if not manifests:
            provenance_sources.append({"root": str(root), "status": "unknown_no_manifest"})
            verified = False
            continue
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
        records = {
            str(record["recording_id"]): record
            for payload in payloads
            for record in payload.get("records", [])
        }
        required = required_recording_ids or set(records)
        missing = sorted(required - set(records))
        sidecars_verified = True
        maximum_future_offset: int | None = None
        for recording_id in sorted(required & set(records)):
            record = records[recording_id]
            sidecar = root / str(record.get("split", "")) / f"{recording_id}.provenance.npz"
            if not sidecar.is_file():
                sidecars_verified = False
                continue
            with np.load(sidecar, allow_pickle=False) as arrays:
                if not {"prediction_frame_indices", "source_frame_indices"}.issubset(arrays.files):
                    sidecars_verified = False
                    continue
                predictions = arrays["prediction_frame_indices"].astype(np.int64)
                clip_sources = arrays["source_frame_indices"].astype(np.int64)
                if (
                    predictions.ndim != 1
                    or clip_sources.ndim != 2
                    or len(predictions) != len(clip_sources)
                ):
                    sidecars_verified = False
                    continue
                offset = int(np.max(clip_sources - predictions[:, None]))
                maximum_future_offset = (
                    offset if maximum_future_offset is None else max(maximum_future_offset, offset)
                )
                sidecars_verified &= offset <= 0
        causal = not missing and sidecars_verified and all(
            payload.get("clip_anchor") == "trailing"
            and payload.get("causal_clips") is True
            and all(
                int(record.get("max_future_offset_frames", 1)) <= 0
                for record in payload.get("records", [])
            )
            for payload in payloads
        )
        provenance_sources.append(
            {
                "root": str(root),
                "status": "verified_trailing" if causal else "noncausal_or_unverified",
                "manifests": [path.name for path in manifests],
                "missing_recording_ids": missing,
                "max_future_offset_frames": maximum_future_offset,
            }
        )
        verified &= causal
    return {
        "kind": "precomputed",
        "causality_verified": verified,
        "sources": provenance_sources,
    }


def _load_precomputed_source_positions(
    directories: str | list[str] | tuple[str, ...] | None,
    recording: IndustRealRecording,
    expected_predictions: np.ndarray,
) -> np.ndarray | None:
    """Load and combine exact source positions from precomputed feature sidecars."""
    if not directories:
        return None
    roots = [directories] if isinstance(directories, str) else list(directories)
    source_groups: list[np.ndarray] = []
    for directory in roots:
        path = Path(directory) / recording.split / f"{recording.recording_id}.provenance.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as arrays:
            if not {"prediction_frame_indices", "source_frame_indices"}.issubset(arrays.files):
                return None
            predictions = arrays["prediction_frame_indices"].astype(np.int64)
            sources = arrays["source_frame_indices"].astype(np.int64)
        if not np.array_equal(predictions, expected_predictions):
            raise ValueError(
                f"Precomputed temporal provenance for {recording.recording_id} is not aligned "
                "with the requested StateGraph stride."
            )
        source_groups.append(sources)
    return np.concatenate(source_groups, axis=1)


def _normalize_video_frame(frame: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = _resize_center_crop(rgb, 224)
    array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)[:, None, None]
    std = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)[:, None, None]
    return (array - mean) / std


def _normalize_image_frame(frame: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = _resize_center_crop(rgb, 224)
    array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (array - mean) / std


def _resize_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))))
    top = max(0, (resized.shape[0] - size) // 2)
    left = max(0, (resized.shape[1] - size) // 2)
    return resized[top : top + size, left : left + size]


def _frame_index(name: str) -> int:
    matches = re.findall(r"\d+", Path(name).stem)
    return int(matches[-1]) if matches else 0


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value))


def _is_background_action(segment) -> bool:
    candidates = {
        re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        for value in (segment.action_id, segment.description)
    }
    return bool(candidates & {"background", "idle", "none", "no action"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a low-VRAM StateGraph feature cache from IndustReal.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--motion-features-dir",
        action="append",
        default=None,
        help="Precomputed motion root; repeat to concatenate complementary aligned encoders.",
    )
    parser.add_argument(
        "--motion-backend-name",
        default=None,
        help="Provenance label stored in index.json for precomputed motion features.",
    )
    parser.add_argument("--motion-aux-features-dir", default=None)
    parser.add_argument("--motion-aux-backend-name", default=None)
    parser.add_argument("--appearance-features-dir", default=None)
    parser.add_argument(
        "--auxiliary-root",
        default=None,
        help="Overlay root containing recordings/{split}/{id}/hands.csv, pose.csv, and OD_labels.json.",
    )
    parser.add_argument(
        "--recording-id",
        action="append",
        default=None,
        help="Cache only this recording ID; repeat the option to select several recordings.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Limit the number of discovered recordings for a smoke test.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--clip-span-frames", type=int, default=32)
    parser.add_argument(
        "--clip-anchor",
        choices=["trailing", "centered"],
        default="trailing",
        help="Trailing is strict online mode; centered is an offline/look-ahead comparison.",
    )
    parser.add_argument(
        "--strict-causal",
        action="store_true",
        help="Reject precomputed motion features without a verified past-only provenance manifest.",
    )
    parser.add_argument("--sensor-dim", type=int, default=128)
    parser.add_argument("--num-components", type=int, default=11)
    parser.add_argument(
        "--procedure-schema",
        default=str(Path(__file__).resolve().parents[3] / "configs" / "procedure_schemas" / "industreal_v1.json"),
        help="Dataset adapter mapping raw completion IDs to canonical components.",
    )
    args = parser.parse_args()
    if args.stride_frames <= 0 or args.clip_frames <= 0 or args.clip_span_frames <= 0:
        parser.error("frame sampling parameters must be positive")
    if args.max_recordings is not None and args.max_recordings <= 0:
        parser.error("max-recordings must be positive")
    result = build_industreal_cache(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
