from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Sample:
    video_path: Path
    label: str


def load_manifest(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                samples.append(Sample(video_path=Path(row["video_path"]), label=row["label"]))
    return samples


def extract_video_features(video_path: Path, max_frames: int = 24) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    last_gray: np.ndarray | None = None
    motion_scores: list[float] = []

    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        resized = cv2.resize(frame, (64, 64))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if last_gray is not None:
            motion_scores.append(float(np.mean(np.abs(gray - last_gray))))
        last_gray = gray
        frames.append(rgb)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    if not frames:
        return np.zeros(14, dtype=np.float32)

    stack = np.stack(frames)
    channel_mean = stack.mean(axis=(0, 1, 2))
    channel_std = stack.std(axis=(0, 1, 2))
    gray_stack = stack.mean(axis=3)
    brightness = gray_stack.mean(axis=(1, 2))

    feature = np.array(
        [
            *channel_mean.tolist(),
            *channel_std.tolist(),
            float(brightness.mean()),
            float(brightness.std()),
            float(brightness.min()),
            float(brightness.max()),
            float(np.mean(motion_scores)) if motion_scores else 0.0,
            float(np.std(motion_scores)) if motion_scores else 0.0,
            float(len(frames) / max_frames),
            fps,
        ],
        dtype=np.float32,
    )
    return feature


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    epochs: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    weights = np.zeros((x_train.shape[1], num_classes), dtype=np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    losses: list[float] = []

    for _ in range(epochs):
        probabilities = softmax(x_train @ weights + bias)
        target = np.eye(num_classes, dtype=np.float32)[y_train]
        loss = -np.mean(np.sum(target * np.log(probabilities + 1e-8), axis=1))
        losses.append(float(loss))

        gradient = (probabilities - target) / len(x_train)
        weights -= learning_rate * (x_train.T @ gradient)
        bias -= learning_rate * gradient.sum(axis=0)

    return weights, bias, losses


def accuracy(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    predictions = np.argmax(x @ weights + bias, axis=1)
    return float(np.mean(predictions == y))


def split_samples(samples: list[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for label_samples in grouped.values():
        rng.shuffle(label_samples)
        val_count = max(1, int(round(len(label_samples) * val_fraction))) if len(label_samples) > 1 else 0
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    return train, val


def vectorize(samples: list[Sample], label_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.zeros((0, 14), dtype=np.float32), np.zeros(0, dtype=np.int64)
    features = [extract_video_features(sample.video_path) for sample in samples]
    labels = [label_to_index[sample.label] for sample in samples]
    return np.stack(features), np.array(labels, dtype=np.int64)


def standardize(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_manifest(args.manifest)
    labels = sorted({sample.label for sample in samples})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    x_train, y_train = vectorize(train_samples, label_to_index)
    x_val, y_val = vectorize(val_samples, label_to_index)
    x_train, x_val, mean, std = standardize(x_train, x_val)

    weights, bias, losses = train_softmax_classifier(
        x_train=x_train,
        y_train=y_train,
        num_classes=len(labels),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    metrics = {
        "samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "classes": labels,
        "epochs": args.epochs,
        "final_loss": losses[-1],
        "train_accuracy": accuracy(x_train, y_train, weights, bias),
        "val_accuracy": accuracy(x_val, y_val, weights, bias),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "weights": weights,
                "bias": bias,
                "mean": mean,
                "std": std,
                "classes": labels,
            },
            handle,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight CPU video classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="runs/models/simple_video_classifier")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    metrics = run_training(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
