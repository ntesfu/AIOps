"""Extract frozen VideoMAEv2 clip features per CaptainCook4D step segment.

For each observable step segment we sample K short clips spread across the
segment (K scales with duration), encode each with the frozen VideoMAEv2 clip
encoder, and store both the per-clip features (for a V2 transformer head) and
the segment mean pool (for a V1 MLP/linear head).  Output is one resumable
``.npz`` per recording plus a manifest.

Run on the remote GPU box inside ``psr_env``:

    PYTHONPATH=src python scripts/extract_captaincook4d_features.py \
        --annotations-root /media/.../captaincook4d/annotations \
        --data-root        /media/.../captaincook4d/data \
        --output-dir       /media/.../captaincook4d/features/videomaev2 \
        --clip-seconds 2.0 --max-clips 16 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aiops.data.captaincook4d import (
    ERROR_CATEGORIES,
    CaptainCook4DSegment,
    load_segments,
    video_path,
)
from aiops.features.videomaev2_features import HuggingFaceVideoMAEv2Encoder

CATEGORY_INDEX = {name: i for i, name in enumerate(ERROR_CATEGORIES)}


def segment_clip_frames(
    start_frame: int,
    end_frame: int,
    length: int,
    num_clips: int,
    num_frames: int,
    sampling_rate: int,
) -> list[np.ndarray]:
    """Evenly spaced clips inside ``[start_frame, end_frame]`` (segment-bounded)."""
    start_frame = max(0, min(start_frame, length - 1))
    end_frame = max(start_frame, min(end_frame, length - 1))
    if num_clips == 1:
        anchors = [(start_frame + end_frame) // 2]
    else:
        anchors = np.linspace(start_frame, end_frame, num_clips).round().astype(int)
    half = (num_frames - 1) / 2.0
    offsets = (np.arange(num_frames) - half) * sampling_rate
    clips = []
    for anchor in anchors:
        idx = np.rint(anchor + offsets).astype(int)
        idx = np.clip(idx, start_frame, end_frame)  # keep clip inside the step
        clips.append(idx)
    return clips


def n_clips_for(duration: float, clip_seconds: float, max_clips: int) -> int:
    return int(max(1, min(max_clips, round(duration / max(0.5, clip_seconds)))))


def extract_recording(
    segments: list[CaptainCook4DSegment],
    video_file: Path,
    encoder: HuggingFaceVideoMAEv2Encoder,
    *,
    clip_seconds: float,
    max_clips: int,
    num_frames: int,
    sampling_rate: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    from decord import VideoReader

    reader = VideoReader(str(video_file))
    length = len(reader)
    fps = float(reader.get_avg_fps()) or 30.0

    all_clip_idx: list[np.ndarray] = []
    clip_counts: list[int] = []
    for seg in segments:
        k = n_clips_for(seg.duration, clip_seconds, max_clips)
        s = int(round(seg.start_time * fps))
        e = int(round(seg.end_time * fps))
        clips = segment_clip_frames(s, e, length, k, num_frames, sampling_rate)
        all_clip_idx.extend(clips)
        clip_counts.append(len(clips))

    # Encode all clips for the recording in batches; decode via decord (RGB).
    clip_feats = np.empty((len(all_clip_idx), encoder.feature_dim), dtype=np.float32)
    for start in range(0, len(all_clip_idx), batch_size):
        batch = all_clip_idx[start : start + batch_size]
        clips = []
        for idx in batch:
            frames_rgb = reader.get_batch(idx.tolist()).asnumpy()  # [T,H,W,3] RGB
            # Encoder flips channels (expects BGR); feed BGR so processor sees RGB.
            clips.append([f[..., ::-1] for f in frames_rgb])
        clip_feats[start : start + len(batch)] = encoder.encode(clips)

    counts = np.asarray(clip_counts, dtype=np.int64)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    seg_mean = np.stack(
        [clip_feats[bounds[i] : bounds[i + 1]].mean(0) for i in range(len(segments))]
    ).astype(np.float32)

    labels = np.asarray([s.label for s in segments], dtype=np.int64)
    cat = np.zeros((len(segments), len(ERROR_CATEGORIES)), dtype=np.int64)
    for i, s in enumerate(segments):
        for tag in s.error_tags:
            cat[i, CATEGORY_INDEX[tag]] = 1
    return {
        "seg_mean": seg_mean,
        "clip_feats": clip_feats,
        "clip_counts": counts,
        "labels": labels,
        "categories": cat,
        "step_id": np.asarray([s.step_id for s in segments], dtype=np.int64),
        "start_time": np.asarray([s.start_time for s in segments], dtype=np.float32),
        "end_time": np.asarray([s.end_time for s in segments], dtype=np.float32),
        "recording_id": np.asarray([s.recording_id for s in segments]),
        "person_id": np.asarray([s.person_id for s in segments]),
        "environment": np.asarray([s.environment for s in segments]),
        "activity_id": np.asarray([s.activity_id for s in segments], dtype=np.int64),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--clip-seconds", type=float, default=2.0)
    p.add_argument("--max-clips", type=int, default=16)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--sampling-rate", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--limit", type=int, default=None, help="max recordings (debug)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = load_segments(args.annotations_root)
    by_rec: dict[str, list[CaptainCook4DSegment]] = {}
    for seg in segments:
        by_rec.setdefault(seg.recording_id, []).append(seg)

    recording_ids = sorted(by_rec)
    encoder: HuggingFaceVideoMAEv2Encoder | None = None
    manifest: list[dict] = []
    done = skipped = 0
    for i, rid in enumerate(recording_ids, 1):
        if args.limit and done >= args.limit:
            break
        cache_path = out / f"{rid}.npz"
        video_file = video_path(args.data_root, rid)
        if cache_path.exists():
            skipped += 1
            continue
        if not video_file.exists():
            print(f"[{i}/{len(recording_ids)}] MISSING video {rid}, skip", flush=True)
            continue
        if encoder is None:
            encoder = HuggingFaceVideoMAEv2Encoder(precision=args.precision, pooling="model")
        try:
            payload = extract_recording(
                by_rec[rid], video_file, encoder,
                clip_seconds=args.clip_seconds, max_clips=args.max_clips,
                num_frames=args.num_frames, sampling_rate=args.sampling_rate,
                batch_size=args.batch_size,
            )
        except Exception as exc:  # pragma: no cover - robust to a few bad decodes
            print(f"[{i}/{len(recording_ids)}] FAILED {rid}: {exc}", flush=True)
            continue
        tmp = cache_path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as handle:  # handle avoids savez auto-appending .npz
            np.savez_compressed(handle, **payload)
        tmp.replace(cache_path)
        done += 1
        n = len(payload["labels"])
        print(f"[{i}/{len(recording_ids)}] {rid}: {n} segs, "
              f"{int(payload['labels'].sum())} err, {len(payload['clip_feats'])} clips", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "feature_dim": int(encoder.feature_dim) if encoder else None,
        "clip_seconds": args.clip_seconds, "max_clips": args.max_clips,
        "num_frames": args.num_frames, "sampling_rate": args.sampling_rate,
        "recordings_done": done, "recordings_skipped_existing": skipped,
    }, indent=2))
    print(f"DONE: extracted={done} skipped_existing={skipped}", flush=True)


if __name__ == "__main__":
    main()
