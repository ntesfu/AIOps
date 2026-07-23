from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    pad_stategraph_batch,
    read_cache_index,
)
from aiops.models.stateverify_effect import (
    StateEffectLossConfig,
    StateEffectObserverConfig,
    build_state_effect_loss,
    build_state_effect_observer,
    state_effect_targets,
)
from aiops.training.monitoring import TrainingMonitor
from aiops.training.train_stategraph_psr import (
    OutcomeAwareBatchSampler,
    _infer_event_state_mapping,
)


ROI_ORDER = (
    "left_hand",
    "right_hand",
    "active_object",
    "interaction_context",
)


def _move_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _observer_inputs(batch, config: StateEffectObserverConfig):
    import torch

    auxiliary = batch["motion_aux"]
    visual_width = config.roi_count * config.roi_dim
    mask_end = visual_width + config.roi_count
    roi_features = auxiliary[..., :visual_width].reshape(
        *auxiliary.shape[:2], config.roi_count, config.roi_dim
    )
    roi_mask = auxiliary[..., visual_width:mask_end] > 0.5
    sensor_present = batch["modality_mask"][..., 2].bool()
    hands = batch["sensor"][..., : config.hand_dim]
    pose_start = config.hand_dim
    pose = batch["sensor"][..., pose_start : pose_start + config.pose_dim]
    hand_mask = sensor_present & (hands.abs().sum(dim=-1) > 0)
    pose_mask = sensor_present & (pose.abs().sum(dim=-1) > 0)
    return {
        "global_features": torch.cat(
            [batch["motion"], batch["appearance"]], dim=-1
        ),
        "roi_features": roi_features,
        "roi_mask": roi_mask,
        "hands": hands,
        "hand_mask": hand_mask,
        "pose": pose,
        "pose_mask": pose_mask,
        "valid_mask": batch["valid_mask"],
    }


def _validate_contract(metadata, records) -> int:
    if not records:
        raise ValueError("The cache index has no records.")
    sensor = metadata.get("sensor_contract", {})
    if sensor.get("gaze_excluded") is not True:
        raise ValueError(
            "StateVerify training requires a sensor_contract with gaze_excluded=true. "
            "Run scripts/rebuild_stategraph_sensors.py first."
        )
    if sensor.get("sources") != ["hands", "headset_pose"]:
        raise ValueError("Sensor sources must be exactly hands and headset_pose.")
    provenance = metadata.get("motion_aux_provenance", {})
    if int(provenance.get("format_version", 0)) != 2:
        raise ValueError(
            "StateVerify training requires gaze-free ROI format_version=2."
        )
    if tuple(provenance.get("roi_names", ())) != ROI_ORDER:
        raise ValueError(f"ROI order must be {ROI_ORDER}.")
    first = records[0]
    dimensions = {
        (
            record.motion_dim,
            record.appearance_dim,
            record.sensor_dim,
            record.motion_aux_dim,
            record.num_components,
            record.num_completion_components,
        )
        for record in records
    }
    if len(dimensions) != 1:
        raise ValueError("All cache records must have identical dimensions.")
    fixed_width = 5 * len(ROI_ORDER) + 3
    visual_width = first.motion_aux_dim - fixed_width
    if visual_width <= 0 or visual_width % len(ROI_ORDER):
        raise ValueError("The ROI cache does not match the v2 flat feature contract.")
    return visual_width // len(ROI_ORDER)


def _class_weights(records, event_state_indices):
    state_counts = np.ones(3, dtype=np.float64)
    effect_counts = np.ones(4, dtype=np.float64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            state = arrays["state"].astype(np.int64)
            state_mask = arrays["state_mask"].astype(bool)
            valid = np.ones(len(state), dtype=bool)
            mapped = np.full_like(state, -100)
            mapped[state == 1] = 0
            mapped[state == 2] = 1
            mapped[state == 0] = 2
            state_counts += np.bincount(
                mapped[state_mask], minlength=3
            )[:3]
            effect = np.full_like(state, -100)
            if len(state) > 1:
                pair = state_mask[1:] & state_mask[:-1]
                previous = mapped[:-1]
                current = mapped[1:]
                transition = np.zeros_like(current)
                transition[(current == 1) & (current != previous)] = 1
                transition[(current == 2) & (current != previous)] = 2
                transition[(current == 0) & (current != previous)] = 3
                effect[1:][pair] = transition[pair]
            outcomes = arrays["component_outcome"].astype(np.int64)
            for event_component, state_component in enumerate(event_state_indices):
                mask = outcomes[:, event_component] >= 0
                effect[:, state_component][mask] = (
                    outcomes[:, event_component][mask] + 1
                )
            effect_counts += np.bincount(
                effect[effect >= 0], minlength=4
            )[:4]

    def balance(counts):
        weights = np.sqrt(counts.sum() / (len(counts) * counts))
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.02, 8.0)
        return (weights / weights.mean()).astype(np.float32)

    return balance(state_counts), balance(effect_counts), state_counts, effect_counts


def _update_confusion(confusion, logits, target, mask):
    prediction = logits.argmax(dim=-1)
    actual = target[mask].detach().cpu().numpy()
    predicted = prediction[mask].detach().cpu().numpy()
    np.add.at(confusion, (actual, predicted), 1)


def _confusion_metrics(prefix: str, confusion: np.ndarray) -> dict[str, float]:
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros_like(support, dtype=np.float64),
        where=support > 0,
    )
    total = confusion.sum()
    metrics = {
        f"{prefix}_accuracy": float(np.trace(confusion) / total) if total else 0.0,
        f"{prefix}_macro_recall": float(recall[support > 0].mean())
        if (support > 0).any()
        else 0.0,
    }
    for index, value in enumerate(recall):
        metrics[f"{prefix}_recall_class_{index}"] = float(value)
        metrics[f"{prefix}_support_class_{index}"] = float(support[index])
    return metrics


def _evaluate(model, criterion, loader, device, config, event_state_indices, weights):
    import torch

    model.eval()
    totals: dict[str, list[float]] = {}
    state_confusion = np.zeros((3, 3), dtype=np.int64)
    effect_confusion = np.zeros((4, 4), dtype=np.int64)
    step_confusion = (
        np.zeros((config.num_steps, config.num_steps), dtype=np.int64)
        if config.num_steps > 0
        else None
    )
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            outputs = model(**_observer_inputs(batch, config))
            losses = criterion(
                outputs,
                batch,
                event_state_indices,
                state_class_weights=weights[0],
                effect_class_weights=weights[1],
            )
            typed = state_effect_targets(batch, event_state_indices)
            _update_confusion(
                state_confusion,
                outputs["state_logits"],
                typed["state"],
                typed["state_mask"],
            )
            _update_confusion(
                effect_confusion,
                outputs["effect_logits"],
                typed["effect"],
                typed["effect_mask"],
            )
            if step_confusion is not None:
                _update_confusion(
                    step_confusion,
                    outputs["step_logits"],
                    batch["step"],
                    batch["valid_mask"].bool() & (batch["step"] >= 0),
                )
            for name, value in losses.items():
                totals.setdefault(name, []).append(float(value.detach().cpu()))
    metrics = {f"loss_{name}": float(np.mean(values)) for name, values in totals.items()}
    metrics.update(_confusion_metrics("state", state_confusion))
    metrics.update(_confusion_metrics("effect", effect_confusion))
    if step_confusion is not None:
        metrics.update(_confusion_metrics("step", step_confusion))
    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    metadata, records = read_cache_index(args.cache_index)
    roi_dim = _validate_contract(metadata, records)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    if not train_records or not val_records:
        raise ValueError("StateVerify training requires train and val cache records.")
    first = train_records[0]
    mapping = _infer_event_state_mapping(
        train_records,
        first.num_completion_components,
        first.num_components,
        metadata,
    )
    event_state_indices = tuple(mapping["indices"])
    sensor_contract = metadata["sensor_contract"]
    config = StateEffectObserverConfig(
        global_dim=first.motion_dim + first.appearance_dim,
        roi_dim=roi_dim,
        num_components=first.num_components,
        num_steps=len(metadata.get("action_ids", [])),
        hand_dim=int(sensor_contract["hand_dim"]),
        pose_dim=int(sensor_contract["pose_dim"]),
        hidden_dim=args.hidden_dim,
        temporal_blocks=args.temporal_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        effect_lag=args.effect_lag,
    )
    loss_config = StateEffectLossConfig(
        state_weight=args.state_weight,
        effect_weight=args.effect_weight,
        step_weight=args.step_weight,
        transition_consistency_weight=args.transition_consistency_weight,
        focal_gamma=args.focal_gamma,
        effect_no_change_ratio=args.effect_no_change_ratio,
        effect_min_no_change=args.effect_min_no_change,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_state_effect_observer(config).to(device)
    criterion = build_state_effect_loss(loss_config).to(device)
    state_weights_np, effect_weights_np, state_counts, effect_counts = _class_weights(
        train_records, event_state_indices
    )
    state_weights = torch.from_numpy(state_weights_np).to(device)
    effect_weights = torch.from_numpy(effect_weights_np).to(device)
    train_dataset = StateGraphCacheDataset(
        train_records,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        training=True,
        seed=args.seed,
        preload=args.preload,
        event_centered_crops_per_event=args.event_centered_crops_per_event,
        event_centered_crop_radius=args.event_centered_crop_radius,
    )
    sampling = train_dataset.window_sampling_weights(
        rare_window_boost=(
            1.0 if args.rare_windows_per_batch > 0 else args.rare_window_boost
        )
    )
    batch_sampler = OutcomeAwareBatchSampler(
        len(train_dataset),
        args.batch_size,
        train_dataset.rare_window_indices(),
        rare_per_batch=args.rare_windows_per_batch,
        sampling_weights=sampling,
        seed=args.seed,
    )
    worker_options = dict(
        num_workers=args.workers,
        collate_fn=pad_stategraph_batch,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, **worker_options
    )
    val_loader = DataLoader(
        StateGraphCacheDataset(
            val_records,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_length,
            preload=args.preload,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **worker_options,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(output_dir, enabled=not args.no_monitor)
    history = []
    best_score = -float("inf")
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            train_dataset.set_epoch(epoch)
            model.train()
            running: dict[str, list[float]] = {}
            for batch_index, batch in enumerate(train_loader):
                if args.max_train_batches and batch_index >= args.max_train_batches:
                    break
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda",
                ):
                    outputs = model(**_observer_inputs(batch, config))
                    losses = criterion(
                        outputs,
                        batch,
                        event_state_indices,
                        state_class_weights=state_weights,
                        effect_class_weights=effect_weights,
                    )
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                for name, value in losses.items():
                    running.setdefault(name, []).append(float(value.detach().cpu()))
                global_step += 1
                if global_step % args.system_log_interval == 0:
                    monitor.log_system(global_step, device)
            scheduler.step()
            train_metrics = {
                f"loss_{name}": float(np.mean(values))
                for name, values in running.items()
            }
            val_metrics = _evaluate(
                model,
                criterion,
                val_loader,
                device,
                config,
                event_state_indices,
                (state_weights, effect_weights),
            )
            score = (
                val_metrics["state_macro_recall"]
                + val_metrics["effect_macro_recall"]
                + val_metrics["state_recall_class_2"]
                + 2.0 * val_metrics["effect_recall_class_2"]
                + 0.25 * val_metrics.get("step_accuracy", 0.0)
            )
            epoch_result = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": val_metrics,
                "selection_score": score,
            }
            history.append(epoch_result)
            monitor.log_scalars("train_epoch", train_metrics, epoch)
            monitor.log_scalars("validation", val_metrics, epoch)
            monitor.log_scalars(
                "optimizer",
                {"learning_rate": optimizer.param_groups[0]["lr"]},
                epoch,
            )
            print(json.dumps(epoch_result), flush=True)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(config),
                "loss_config": asdict(loss_config),
                "event_state_mapping": mapping,
                "metrics": val_metrics,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if score > best_score:
                best_score = score
                torch.save(checkpoint, output_dir / "best.pt")
    finally:
        monitor.close()

    result = {
        "cache_index": str(Path(args.cache_index).resolve()),
        "model_config": asdict(config),
        "loss_config": asdict(loss_config),
        "event_state_mapping": mapping,
        "state_counts": state_counts.tolist(),
        "effect_counts": effect_counts.tolist(),
        "state_class_weights": state_weights_np.tolist(),
        "effect_class_weights": effect_weights_np.tolist(),
        "best_selection_score": best_score,
        "history": history,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the gaze-free StateVerify component observer."
    )
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=192)
    parser.add_argument("--sequence-stride", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-blocks", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--effect-lag", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument("--step-weight", type=float, default=0.5)
    parser.add_argument("--transition-consistency-weight", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--effect-no-change-ratio", type=float, default=4.0)
    parser.add_argument("--effect-min-no-change", type=int, default=64)
    parser.add_argument("--rare-window-boost", type=float, default=6.0)
    parser.add_argument("--rare-windows-per-batch", type=int, default=1)
    parser.add_argument("--event-centered-crops-per-event", type=int, default=2)
    parser.add_argument("--event-centered-crop-radius", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--system-log-interval", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--no-monitor", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "best_selection_score": result["best_selection_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
