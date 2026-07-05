from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from aiops.data.temporal_manifest import TemporalSegment, TemporalVideo, write_temporal_manifest

STEP_ALIASES = {
    "activity_walking": "pick_part",
    "activity_carrying": "carry_part",
    "activity_standing": "position_part",
    "specialized_using_tool": "use_tool",
    "Pull": "pull_part",
    "Interacts": "align_part",
    "Transport_HeavyCarry": "move_heavy_part",
    "Entering": "enter_station",
    "Exiting": "exit_station",
}


def read_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clip_duration_s(video_path: str | Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def concatenate_videos(inputs: list[Path], output_path: Path, fps: float = 12.0, size: tuple[int, int] = (160, 160)) -> None:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for input_path in inputs:
        capture = cv2.VideoCapture(str(input_path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, size))
        capture.release()
    writer.release()


def build_sequences(
    source_manifest: str | Path,
    output_dir: str | Path,
    num_sequences: int,
    steps_per_sequence: int,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    source_rows = read_rows(source_manifest)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_label[row["label"]].append(row)

    usable_labels = [label for label, rows in by_label.items() if rows]
    output_root = Path(output_dir)
    video_dir = output_root / "videos"
    manifest_rows: list[TemporalVideo] = []

    for sequence_index in range(num_sequences):
        labels = [usable_labels[(sequence_index + offset) % len(usable_labels)] for offset in range(steps_per_sequence)]
        if sequence_index % 3 == 1:
            rng.shuffle(labels)
        elif sequence_index % 3 == 2 and len(labels) > 2:
            labels.insert(2, labels[1])

        selected_rows = [rng.choice(by_label[label]) for label in labels]
        input_paths = [Path(row["video_path"]) for row in selected_rows]
        output_video = video_dir / f"procedure_{sequence_index:03d}.mp4"
        concatenate_videos(input_paths, output_video)

        cursor = 0.0
        segments: list[TemporalSegment] = []
        for row, input_path in zip(selected_rows, input_paths):
            duration = clip_duration_s(input_path)
            step_id = STEP_ALIASES.get(row["label"], row["label"])
            segments.append(
                TemporalSegment(
                    step_id=step_id,
                    label=step_id.replace("_", " "),
                    start_s=round(cursor, 3),
                    end_s=round(cursor + duration, 3),
                )
            )
            cursor += duration
        manifest_rows.append(
            TemporalVideo(
                video_path=output_video,
                dataset="procedural_tinyvirat",
                procedure="synthetic_multi_step_assembly",
                segments=tuple(segments),
            )
        )

    manifest_path = output_root / "manifest.jsonl"
    write_temporal_manifest(manifest_rows, manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-step procedural videos from labeled clips.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", default="data/processed/procedural_tinyvirat")
    parser.add_argument("--num-sequences", type=int, default=48)
    parser.add_argument("--steps-per-sequence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print(
        build_sequences(
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
            num_sequences=args.num_sequences,
            steps_per_sequence=args.steps_per_sequence,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()

