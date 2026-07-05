from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aiops.features import StatisticalClipFeatureExtractor
from aiops.models.mstcn import build_ms_tcn_plus_plus
from aiops.training.simple_video_classifier import Sample, load_manifest, split_samples


@dataclass(frozen=True)
class SequenceExample:
    features: np.ndarray
    labels: np.ndarray


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vectorize_sequences(
    samples: list[Sample],
    label_to_index: dict[str, int],
    extractor: StatisticalClipFeatureExtractor,
) -> list[SequenceExample]:
    examples: list[SequenceExample] = []
    for sample in samples:
        clips = extractor.extract(sample.video_path)
        if not clips:
            continue
        features = np.stack([clip.vector for clip in clips]).astype(np.float32)
        labels = np.full(len(clips), label_to_index[sample.label], dtype=np.int64)
        examples.append(SequenceExample(features=features, labels=labels))
    return examples


def feature_stats(examples: list[SequenceExample]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([example.features for example in examples], axis=0)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = (matrix.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def normalize_examples(examples: list[SequenceExample], mean: np.ndarray, std: np.ndarray) -> list[SequenceExample]:
    return [SequenceExample(features=(example.features - mean) / std, labels=example.labels) for example in examples]


def batch_accuracy(model: Any, examples: list[SequenceExample], device: Any) -> float:
    import torch

    if not examples:
        return 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for example in examples:
            x = torch.from_numpy(example.features.T[None, :, :]).to(device)
            y = torch.from_numpy(example.labels).to(device)
            logits = model(x)[-1][0].transpose(0, 1)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / total if total else 0.0


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    set_seed(args.seed)
    samples = load_manifest(args.manifest)
    classes = sorted({sample.label for sample in samples})
    label_to_index = {label: index for index, label in enumerate(classes)}
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)

    extractor = StatisticalClipFeatureExtractor(
        clip_duration_s=args.clip_duration_s,
        stride_s=args.clip_stride_s,
        frame_size=args.frame_size,
    )
    train_examples = vectorize_sequences(train_samples, label_to_index, extractor)
    val_examples = vectorize_sequences(val_samples, label_to_index, extractor)
    mean, std = feature_stats(train_examples)
    train_examples = normalize_examples(train_examples, mean, std)
    val_examples = normalize_examples(val_examples, mean, std)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = {
        "input_dim": extractor.output_dim,
        "hidden_dim": args.hidden_dim,
        "num_stages": args.num_stages,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    model = build_ms_tcn_plus_plus(num_classes=len(classes), **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_examples)
        model.train()
        losses: list[float] = []
        for example in train_examples:
            x = torch.from_numpy(example.features.T[None, :, :]).to(device)
            y = torch.from_numpy(example.labels[None, :]).to(device)
            outputs = model(x)
            loss = sum(functional.cross_entropy(stage_logits, y) for stage_logits in outputs) / len(outputs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "train_accuracy": batch_accuracy(model, train_examples, device),
                "val_accuracy": batch_accuracy(model, val_examples, device),
                "device": str(device),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    metrics = {
        "samples": len(samples),
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
        "classes": classes,
        "epochs": args.epochs,
        "train_accuracy": batch_accuracy(model, train_examples, device),
        "val_accuracy": batch_accuracy(model, val_examples, device),
        "device": str(device),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "classes": classes,
        "feature_mean": mean,
        "feature_std": std,
        "clip_duration_s": args.clip_duration_s,
        "clip_stride_s": args.clip_stride_s,
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AIOps MS-TCN temporal head.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="runs/models/mstcn_tinyvirat")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--clip-duration-s", type=float, default=1.0)
    parser.add_argument("--clip-stride-s", type=float, default=0.5)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
