"""Extract InternVideo2-B14 per-segment features for CaptainCook4D.

Same npz schema as the VideoMAEv2 extractor (so the screen script is unchanged),
but with a stronger semantic+temporal backbone. 8-frame clips at ~1 clip/second
(matching the paper's 1-s sub-segment granularity). Resumable per-recording.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiops.data.captaincook4d import ERROR_CATEGORIES, load_segments, video_path
from internvideo2_b import InternVideo2Encoder
from extract_captaincook4d_features import segment_clip_frames, n_clips_for

CATEGORY_INDEX = {name: i for i, name in enumerate(ERROR_CATEGORIES)}
CKPT = ("/home/aiops/.cache/huggingface/hub/models--OpenGVLab--InternVideo2_distillation_models/"
        "snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac/stage1/B14/B14_ft_k710_f8/pytorch_model.bin")


def extract_recording(segments, video_file, encoder, *, clip_seconds, max_clips,
                      num_frames, sampling_rate, batch_size):
    from decord import VideoReader

    reader = VideoReader(str(video_file))
    length = len(reader)
    fps = float(reader.get_avg_fps()) or 30.0
    all_idx, counts = [], []
    for seg in segments:
        k = n_clips_for(seg.duration, clip_seconds, max_clips)
        s = int(round(seg.start_time * fps))
        e = int(round(seg.end_time * fps))
        clips = segment_clip_frames(s, e, length, k, num_frames, sampling_rate)
        all_idx.extend(clips)
        counts.append(len(clips))

    clip_feats = np.empty((len(all_idx), encoder.feature_dim), dtype=np.float32)
    for start in range(0, len(all_idx), batch_size):
        batch = all_idx[start:start + batch_size]
        clips = []
        for idx in batch:
            frames_rgb = reader.get_batch(idx.tolist()).asnumpy()  # RGB (InternVideo2 wants RGB)
            clips.append([f for f in frames_rgb])
        clip_feats[start:start + len(batch)] = encoder.encode(clips)

    counts = np.asarray(counts, dtype=np.int64)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    seg_mean = np.stack([clip_feats[bounds[i]:bounds[i + 1]].mean(0)
                         for i in range(len(segments))]).astype(np.float32)
    labels = np.asarray([s.label for s in segments], dtype=np.int64)
    cat = np.zeros((len(segments), len(ERROR_CATEGORIES)), dtype=np.int64)
    for i, s in enumerate(segments):
        for tag in s.error_tags:
            cat[i, CATEGORY_INDEX[tag]] = 1
    return {
        "seg_mean": seg_mean, "clip_feats": clip_feats, "clip_counts": counts,
        "labels": labels, "categories": cat,
        "step_id": np.asarray([s.step_id for s in segments], dtype=np.int64),
        "start_time": np.asarray([s.start_time for s in segments], dtype=np.float32),
        "end_time": np.asarray([s.end_time for s in segments], dtype=np.float32),
        "recording_id": np.asarray([s.recording_id for s in segments]),
        "person_id": np.asarray([s.person_id for s in segments]),
        "environment": np.asarray([s.environment for s in segments]),
        "activity_id": np.asarray([s.activity_id for s in segments], dtype=np.int64),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--checkpoint", default=CKPT)
    p.add_argument("--clip-seconds", type=float, default=1.0)
    p.add_argument("--max-clips", type=int, default=16)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--sampling-rate", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segments = load_segments(args.annotations_root)
    by_rec = {}
    for seg in segments:
        by_rec.setdefault(seg.recording_id, []).append(seg)
    recording_ids = sorted(by_rec)

    encoder = None
    done = skipped = 0
    for i, rid in enumerate(recording_ids, 1):
        cache_path = out / f"{rid}.npz"
        video_file = video_path(args.data_root, rid)
        if cache_path.exists():
            skipped += 1
            continue
        if not video_file.exists():
            continue
        if encoder is None:
            import torch
            encoder = InternVideo2Encoder(args.checkpoint, device="cuda", dtype=torch.float32)
            print("IV2 load:", {k: len(v) for k, v in encoder.load_report.items()}, flush=True)
        try:
            payload = extract_recording(by_rec[rid], video_file, encoder,
                                        clip_seconds=args.clip_seconds, max_clips=args.max_clips,
                                        num_frames=args.num_frames, sampling_rate=args.sampling_rate,
                                        batch_size=args.batch_size)
        except Exception as exc:
            print(f"[{i}/{len(recording_ids)}] FAILED {rid}: {exc}", flush=True)
            continue
        tmp = cache_path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as h:
            np.savez_compressed(h, **payload)
        tmp.replace(cache_path)
        done += 1
        print(f"[{i}/{len(recording_ids)}] {rid}: {len(payload['labels'])} segs, "
              f"{int(payload['labels'].sum())} err, {len(payload['clip_feats'])} clips", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "backbone": "InternVideo2-B14_ft_k710_f8", "feature_dim": 768,
        "clip_seconds": args.clip_seconds, "num_frames": args.num_frames,
        "sampling_rate": args.sampling_rate, "recordings_done": done, "skipped": skipped,
    }, indent=2))
    print(f"DONE extracted={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
