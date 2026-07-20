from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from aiops.data.stategraph_cache import (
    StateGraphCacheDataset,
    StateGraphCacheRecord,
    build_transition_matrix,
    pad_stategraph_batch,
    read_cache_index,
)
from aiops.evaluation.temporal_metrics import edit_score, segmental_f1
from aiops.models.stategraph_psr import (
    StateGraphLossConfig,
    StateGraphPSRConfig,
    build_stategraph_loss,
    build_stategraph_psr,
)
from aiops.data.procedure_schema import CACHE_SCHEMA_VERSION
from aiops.training.monitoring import TrainingMonitor


class OutcomeAwareBatchSampler:
    """Deterministic batches with guaranteed exposure to rare event windows."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        rare_indices: list[int],
        rare_per_batch: int = 1,
        seed: int = 7,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        if rare_per_batch < 0 or rare_per_batch > batch_size:
            raise ValueError("rare_per_batch must be between zero and batch_size")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rare_indices = np.asarray(sorted(set(rare_indices)), dtype=np.int64)
        self.rare_per_batch = rare_per_batch if len(self.rare_indices) else 0
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        all_indices = np.arange(self.dataset_size, dtype=np.int64)
        remaining = self.dataset_size
        for _ in range(len(self)):
            current_size = min(self.batch_size, remaining)
            rare_count = min(self.rare_per_batch, current_size)
            batch: list[int] = []
            if rare_count:
                batch.extend(
                    int(value)
                    for value in rng.choice(self.rare_indices, size=rare_count, replace=True)
                )
            fill = current_size - rare_count
            if fill:
                batch.extend(int(value) for value in rng.choice(all_indices, size=fill, replace=True))
            rng.shuffle(batch)
            yield batch
            remaining -= current_size


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    metadata, records = read_cache_index(args.cache_index)
    _validate_cache_schema(metadata, records)
    _validate_records(records)
    train_records = [record for record in records if record.split.lower() == "train"]
    all_train_records = train_records.copy()
    val_records = [record for record in records if record.split.lower() in {"val", "validation"}]
    test_records = [record for record in records if record.split.lower() == "test"]
    if not train_records:
        raise ValueError("The cache index contains no train recordings.")
    overfit_recordings = list(args.overfit_recording_id or [])
    if overfit_recordings:
        by_id = {record.recording_id: record for record in train_records}
        missing = sorted(set(overfit_recordings) - set(by_id))
        if missing:
            raise ValueError(f"Overfit recording IDs are not in the train split: {missing}")
        selected = [by_id[recording_id] for recording_id in overfit_recordings]
        train_records = selected
        val_records = selected
        test_records = []
    elif not val_records:
        train_records, val_records = _recording_level_split(train_records, args.val_fraction, args.seed)

    num_steps = train_records[0].num_steps
    factorization = _action_factorization(metadata, all_train_records, num_steps)
    event_state_mapping = _infer_event_state_mapping(
        all_train_records,
        train_records[0].num_completion_components,
        train_records[0].num_components,
    )
    active_action_mask = (
        _action_presence(train_records, num_steps)
        if overfit_recordings
        else [True] * num_steps
    )
    transition_matrix = build_transition_matrix(train_records, num_steps, args.transition_smoothing)
    model_config = StateGraphPSRConfig(
        motion_dim=train_records[0].motion_dim,
        appearance_dim=train_records[0].appearance_dim,
        sensor_dim=train_records[0].sensor_dim,
        num_steps=num_steps,
        num_action_verbs=len(factorization["verb_names"]),
        num_action_objects=len(factorization["object_names"]),
        action_verb_indices=tuple(factorization["verb_indices"]),
        action_object_indices=tuple(factorization["object_indices"]),
        seen_action_mask=tuple(factorization["seen_mask"]),
        active_action_mask=tuple(active_action_mask),
        event_state_indices=tuple(event_state_mapping["indices"]),
        num_completion_components=train_records[0].num_completion_components,
        num_event_outcomes=len(metadata["event_outcomes"]),
        num_components=train_records[0].num_components,
        hidden_dim=args.hidden_dim,
        num_temporal_blocks=args.num_temporal_blocks,
        attention_every=args.attention_every,
        num_heads=args.num_heads,
        dropout=args.dropout,
        graph_strength_init=args.graph_strength,
    )
    loss_config = StateGraphLossConfig(
        step_weight=args.step_weight,
        completion_weight=args.completion_weight,
        component_outcome_weight=args.component_outcome_weight,
        state_weight=args.state_weight,
        boundary_weight=args.boundary_weight,
        next_step_weight=args.next_step_weight,
        smoothing_weight=args.smoothing_weight,
        graph_weight=args.graph_weight,
        consistency_weight=args.consistency_weight,
        progress_weight=args.progress_weight,
        normality_weight=args.normality_weight,
        focal_gamma=args.focal_gamma,
        asl_negative_gamma=args.asl_negative_gamma,
        asl_clip=args.asl_clip,
        normality_error_weight=args.normality_error_weight,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_stategraph_psr(model_config, transition_matrix).to(device)
    criterion = build_stategraph_loss(loss_config).to(device)
    step_weights = torch.from_numpy(_class_weights(train_records, "step", num_steps)).to(device)
    completion_pos_weights = torch.from_numpy(
        _completion_pos_weights(
            train_records,
            model_config.num_completion_components,
            cap=args.completion_pos_weight_cap,
        )
    ).to(device)
    component_outcome_weights = torch.from_numpy(
        _class_weights(train_records, "component_outcome", model_config.num_event_outcomes)
    ).to(device)

    train_dataset = StateGraphCacheDataset(
        train_records,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        training=True,
        seed=args.seed,
    )
    sampling_weights = train_dataset.window_sampling_weights(
        incorrect_outcome_index=1,
        rare_window_boost=args.rare_window_boost,
    )
    batch_sampler = OutcomeAwareBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=args.batch_size,
        rare_indices=train_dataset.rare_window_indices(),
        rare_per_batch=args.rare_windows_per_batch,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_stategraph_batch,
    )
    val_loader = DataLoader(
        StateGraphCacheDataset(
            val_records,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_length,
            training=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_stategraph_batch,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    total_updates = max(1, math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs)
    warmup_updates = max(1, int(total_updates * args.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: _warmup_cosine(update, total_updates, warmup_updates)
    )
    use_amp = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_amp and args.precision == "fp16"
        )
    except (AttributeError, TypeError):  # torch 2.3 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.precision == "fp16")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    monitor = TrainingMonitor(output_dir, enabled=not args.disable_dashboard)
    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    patience_left = args.patience
    global_update = 0
    event_thresholds: list[float] | None = None

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running: dict[str, list[float]] = {}
            for batch_index, batch in enumerate(train_loader, start=1):
                batch = _move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        batch["motion"],
                        batch["appearance"],
                        batch["sensor"],
                        valid_mask=batch["valid_mask"],
                        modality_mask=batch["modality_mask"],
                    )
                    losses = criterion(
                        outputs,
                        batch,
                        transition_matrix=model.transition_matrix,
                        step_class_weights=step_weights,
                        completion_pos_weights=completion_pos_weights,
                        component_outcome_class_weights=component_outcome_weights,
                    )
                    scaled_loss = losses["total"] / args.accumulation_steps
                scaler.scale(scaled_loss).backward()
                for name, value in losses.items():
                    running.setdefault(name, []).append(float(value.detach().cpu()))
                if batch_index % args.accumulation_steps == 0 or batch_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_update += 1
                    update_values = {
                        name: float(values[-1]) for name, values in running.items()
                    }
                    update_values["learning_rate"] = optimizer.param_groups[0]["lr"]
                    update_values["gradient_norm"] = float(gradient_norm.detach().cpu())
                    monitor.log_scalars("train_update", update_values, global_update)
                    if global_update % args.gpu_log_interval == 0:
                        monitor.log_system(global_update, device)

            train_loss = {name: float(np.mean(values)) for name, values in running.items()}
            calibrate_events = epoch == 1 or epoch % args.calibration_interval == 0
            metrics = evaluate(
                model,
                val_loader,
                device,
                use_amp,
                amp_dtype,
                model_config.num_components,
                event_thresholds=event_thresholds,
                calibrate_events=calibrate_events,
            )
            event_thresholds = [
                metrics["correct_event_threshold"],
                metrics["incorrect_event_threshold"],
                metrics["remove_event_threshold"],
            ]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "graph_strength": float(torch.sigmoid(model.graph_strength_raw).detach().cpu()),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            monitor.log_scalars("train_epoch", train_loss, epoch)
            monitor.log_scalars("validation", metrics, epoch)
            monitor.log_scalars(
                "optimizer",
                {
                    "learning_rate": row["learning_rate"],
                    "graph_strength": row["graph_strength"],
                },
                epoch,
            )
            monitor.log_system(global_update, device)
            print(json.dumps(row), flush=True)

            score = (
                metrics["f1@50"]
                + 0.25 * metrics["edit"]
                + args.incorrect_selection_weight * metrics["incorrect_event_f1"]
            )
            if score > best_score:
                best_score = score
                best_metrics = metrics
                patience_left = args.patience
                _save_checkpoint(
                    output_dir / "best_checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    model_config,
                    loss_config,
                    metadata,
                    transition_matrix,
                    epoch,
                    metrics,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                    break
    finally:
        monitor.close()

    # Report and test the selected checkpoint, not the last (possibly worse)
    # early-stopping epoch.
    best_path = output_dir / "best_checkpoint.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    final_metrics = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        amp_dtype,
        model_config.num_components,
        calibrate_events=True,
    )
    test_metrics = None
    if test_records and args.evaluate_test:
        test_loader = DataLoader(
            StateGraphCacheDataset(
                test_records,
                sequence_length=args.sequence_length,
                sequence_stride=args.sequence_length,
                training=False,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_stategraph_batch,
        )
        test_metrics = evaluate(model, test_loader, device, use_amp, amp_dtype, model_config.num_components)
    summary = {
        "architecture": "StateGraph-PSR Lite Stage 1",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(train_dataset),
        "rare_train_windows": int((sampling_weights > 1.0).sum()),
        "rare_window_boost": args.rare_window_boost,
        "rare_windows_per_batch": args.rare_windows_per_batch,
        "train_recordings": len(train_records),
        "validation_recordings": len(val_records),
        "test_recordings": len(test_records),
        "overfit_recording_ids": overfit_recordings,
        "completion_pos_weight_cap": args.completion_pos_weight_cap,
        "action_factorization": factorization,
        "event_state_mapping": event_state_mapping,
        "model_config": model_config.to_dict(),
        "loss_config": asdict(loss_config),
        "best_validation": best_metrics,
        "final_validation": final_metrics,
        "test": test_metrics,
    }
    summary_payload = json.dumps(summary, indent=2) + "\n"
    (output_dir / "metrics.json").write_text(summary_payload, encoding="utf-8")
    (output_dir / "summary.json").write_text(summary_payload, encoding="utf-8")
    return summary


def evaluate(
    model,
    loader,
    device,
    use_amp: bool,
    amp_dtype,
    num_components: int,
    event_thresholds: list[float] | None = None,
    calibrate_events: bool = True,
) -> dict[str, float]:
    import torch

    model.eval()
    frame_correct = 0
    frame_total = 0
    raw_frame_correct = 0
    seen_action_correct = 0
    seen_action_total = 0
    unseen_action_correct = 0
    unseen_action_total = 0
    edits: list[float] = []
    f1_scores = {0.1: [], 0.25: [], 0.5: []}
    raw_f1_scores = {0.1: [], 0.25: [], 0.5: []}
    event_samples: list[tuple[list[tuple[int, int, int]], np.ndarray]] = []
    normality_scores: list[float] = []
    normality_targets: list[int] = []
    state_correct = 0
    state_total = 0
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    batch["motion"],
                    batch["appearance"],
                    batch["sensor"],
                    valid_mask=batch["valid_mask"],
                    modality_mask=batch["modality_mask"],
                )
            step_prediction = outputs["step_logits"].argmax(dim=-1)
            raw_step_prediction = outputs["raw_step_logits"].argmax(dim=-1)
            completion_score = torch.sigmoid(outputs["completion_logits"])
            component_outcome_probability = torch.softmax(
                outputs["component_outcome_logits"], dim=-1
            )
            state_prediction = outputs["state_logits"].argmax(dim=-1)
            normality_anomaly_score = torch.sigmoid(-outputs["normality_logits"])
            valid = batch["valid_mask"] & (batch["step"] >= 0)
            frame_correct += int((step_prediction[valid] == batch["step"][valid]).sum().item())
            raw_frame_correct += int(
                (raw_step_prediction[valid] == batch["step"][valid]).sum().item()
            )
            frame_total += int(valid.sum().item())
            seen_lookup = outputs["seen_action_mask"].bool()
            target_seen = seen_lookup[batch["step"].clamp_min(0)]
            seen_mask = valid & target_seen
            unseen_mask = valid & ~target_seen
            seen_action_correct += int(
                (step_prediction[seen_mask] == batch["step"][seen_mask]).sum().item()
            )
            seen_action_total += int(seen_mask.sum().item())
            unseen_action_correct += int(
                (step_prediction[unseen_mask] == batch["step"][unseen_mask]).sum().item()
            )
            unseen_action_total += int(unseen_mask.sum().item())
            state_valid = batch["state_mask"] & batch["valid_mask"].unsqueeze(-1)
            state_correct += int((state_prediction[state_valid] == batch["state"][state_valid]).sum().item())
            state_total += int(state_valid.sum().item())
            for sample_index in range(step_prediction.shape[0]):
                length = int(batch["valid_mask"][sample_index].sum().item())
                pred = step_prediction[sample_index, :length].detach().cpu().tolist()
                raw_pred = raw_step_prediction[sample_index, :length].detach().cpu().tolist()
                truth = batch["step"][sample_index, :length].detach().cpu().tolist()
                edits.append(edit_score(pred, truth))
                for overlap in f1_scores:
                    f1_scores[overlap].append(segmental_f1(pred, truth, overlap))
                    raw_f1_scores[overlap].append(segmental_f1(raw_pred, truth, overlap))
                truth_events = _target_events(
                    batch["component_outcome"][sample_index, :length].detach().cpu().numpy()
                )
                event_scores = (
                    completion_score[sample_index, :length].unsqueeze(-1)
                    * component_outcome_probability[sample_index, :length]
                )
                event_samples.append(
                    (truth_events, event_scores.detach().float().cpu().numpy())
                )
                outcome_targets = batch["component_outcome"][sample_index, :length]
                normality_mask = (outcome_targets == 0) | (outcome_targets == 1)
                normality_scores.extend(
                    normality_anomaly_score[sample_index, :length][normality_mask]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                normality_targets.extend(
                    (outcome_targets[normality_mask] == 1).detach().cpu().int().tolist()
                )
    if calibrate_events or event_thresholds is None:
        event_thresholds, calibrated_metrics, event_pr_auc = _calibrate_event_thresholds(
            event_samples, outcomes=3
        )
    else:
        calibrated_metrics = _event_metrics_from_samples(event_samples, event_thresholds)
        event_pr_auc = [float("nan")] * 3
    fixed_metrics = _event_metrics_from_samples(event_samples, [0.1, 0.1, 0.1])
    all_event_metrics = calibrated_metrics["all"]
    correct_metrics = calibrated_metrics[0]
    incorrect_metrics = calibrated_metrics[1]
    remove_metrics = calibrated_metrics[2]
    normality_ap = _binary_average_precision(normality_scores, normality_targets)
    correct_normality = [
        score for score, target in zip(normality_scores, normality_targets) if target == 0
    ]
    incorrect_normality = [
        score for score, target in zip(normality_scores, normality_targets) if target == 1
    ]
    return {
        "frame_accuracy": 100.0 * frame_correct / max(frame_total, 1),
        "raw_frame_accuracy": 100.0 * raw_frame_correct / max(frame_total, 1),
        "seen_action_accuracy": 100.0 * seen_action_correct / max(seen_action_total, 1),
        "unseen_composition_accuracy": 100.0 * unseen_action_correct / max(unseen_action_total, 1),
        "unseen_composition_frames": unseen_action_total,
        "edit": float(np.mean(edits)) if edits else 0.0,
        "f1@10": float(np.mean(f1_scores[0.1])) if f1_scores[0.1] else 0.0,
        "f1@25": float(np.mean(f1_scores[0.25])) if f1_scores[0.25] else 0.0,
        "f1@50": float(np.mean(f1_scores[0.5])) if f1_scores[0.5] else 0.0,
        "raw_f1@50": float(np.mean(raw_f1_scores[0.5])) if raw_f1_scores[0.5] else 0.0,
        "completion_event_precision": all_event_metrics["precision"],
        "completion_event_recall": all_event_metrics["recall"],
        "completion_event_f1": all_event_metrics["f1"],
        "correct_event_precision": correct_metrics["precision"],
        "correct_event_recall": correct_metrics["recall"],
        "correct_event_f1": correct_metrics["f1"],
        "incorrect_event_precision": incorrect_metrics["precision"],
        "incorrect_event_recall": incorrect_metrics["recall"],
        "incorrect_event_f1": incorrect_metrics["f1"],
        "remove_event_precision": remove_metrics["precision"],
        "remove_event_recall": remove_metrics["recall"],
        "remove_event_f1": remove_metrics["f1"],
        "correct_event_threshold": event_thresholds[0],
        "incorrect_event_threshold": event_thresholds[1],
        "remove_event_threshold": event_thresholds[2],
        "correct_event_pr_auc": event_pr_auc[0],
        "incorrect_event_pr_auc": event_pr_auc[1],
        "remove_event_pr_auc": event_pr_auc[2],
        "fixed_0.1_completion_event_f1": fixed_metrics["all"]["f1"],
        "fixed_0.1_incorrect_event_f1": fixed_metrics[1]["f1"],
        "normality_incorrect_average_precision": normality_ap,
        "normality_correct_mean_anomaly": float(np.mean(correct_normality))
        if correct_normality
        else 0.0,
        "normality_incorrect_mean_anomaly": float(np.mean(incorrect_normality))
        if incorrect_normality
        else 0.0,
        "state_accuracy": 100.0 * state_correct / max(state_total, 1),
    }


def _target_events(component_outcome: np.ndarray) -> list[tuple[int, int, int]]:
    rows, components = np.where(component_outcome >= 0)
    return [(int(row), int(component), int(component_outcome[row, component])) for row, component in zip(rows, components)]


def _predicted_events(
    completion_score: np.ndarray,
    component_outcome: np.ndarray,
    threshold: float = 0.5,
) -> list[tuple[int, int, int]]:
    events: list[tuple[int, int, int]] = []
    for component in range(completion_score.shape[1]):
        scores = completion_score[:, component]
        for row, score in enumerate(scores):
            previous = scores[row - 1] if row > 0 else -np.inf
            following = scores[row + 1] if row + 1 < len(scores) else -np.inf
            if score >= threshold and score >= previous and score > following:
                events.append((row, component, int(component_outcome[row, component])))
    return events


def _predicted_events_from_scores(
    event_scores: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
    minimum_distance: int = 4,
) -> list[tuple[int, int, int]]:
    """Propose each component completion once, then assign exactly one outcome."""

    events: list[tuple[int, int, int]] = []
    for component in range(event_scores.shape[1]):
        completion_scores = event_scores[:, component].sum(axis=-1)
        candidates: list[tuple[float, int, int]] = []
        for row, completion_score in enumerate(completion_scores):
            previous = completion_scores[row - 1] if row > 0 else -np.inf
            following = (
                completion_scores[row + 1]
                if row + 1 < len(completion_scores)
                else -np.inf
            )
            if completion_score < previous or completion_score <= following:
                continue
            outcome = int(event_scores[row, component].argmax())
            joint_score = float(event_scores[row, component, outcome])
            if joint_score >= float(thresholds[outcome]):
                candidates.append((joint_score, row, outcome))
        selected_rows: list[int] = []
        for _, row, outcome in sorted(candidates, reverse=True):
            if all(abs(row - selected) >= minimum_distance for selected in selected_rows):
                selected_rows.append(row)
                events.append((row, component, outcome))
    return sorted(events)


def _event_metrics_from_samples(
    samples: list[tuple[list[tuple[int, int, int]], np.ndarray]],
    thresholds: list[float] | tuple[float, ...],
) -> dict[Any, dict[str, float]]:
    outcomes = len(thresholds)
    counters: dict[Any, dict[str, int]] = {
        "all": {"true": 0, "predicted": 0, "tp": 0},
        **{
            outcome: {"true": 0, "predicted": 0, "tp": 0}
            for outcome in range(outcomes)
        },
    }
    for truth_events, event_scores in samples:
        predicted_events = _predicted_events_from_scores(event_scores, thresholds)
        matched = _match_event_counts(truth_events, predicted_events, tolerance=1)
        for key in counters["all"]:
            counters["all"][key] += matched[key]
        for outcome in range(outcomes):
            matched = _match_event_counts(
                [event for event in truth_events if event[2] == outcome],
                [event for event in predicted_events if event[2] == outcome],
                tolerance=1,
            )
            for key in counters[outcome]:
                counters[outcome][key] += matched[key]
    return {key: _precision_recall_f1(value) for key, value in counters.items()}


def _calibrate_event_thresholds(
    samples: list[tuple[list[tuple[int, int, int]], np.ndarray]],
    outcomes: int,
) -> tuple[list[float], dict[Any, dict[str, float]], list[float]]:
    """Select validation thresholds and event-level PR-AUC independently by outcome."""

    grid = np.unique(
        np.concatenate(
            [np.linspace(0.01, 0.2, 20), np.linspace(0.225, 0.9, 28)]
        )
    )
    thresholds: list[float] = []
    pr_auc: list[float] = []
    for outcome in range(outcomes):
        curve: list[tuple[float, float, float]] = []
        best_threshold = 0.1
        best_f1 = -1.0
        for threshold in grid:
            candidate = [0.999] * outcomes
            candidate[outcome] = float(threshold)
            metrics = _event_metrics_from_samples(samples, candidate)[outcome]
            curve.append((metrics["recall"], metrics["precision"], float(threshold)))
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
        points = sorted({(recall, precision) for recall, precision, _ in curve})
        area = 0.0
        for (left_recall, left_precision), (right_recall, right_precision) in zip(
            points[:-1], points[1:]
        ):
            area += (right_recall - left_recall) * (left_precision + right_precision) / 2.0
        pr_auc.append(area / 100.0)
    return thresholds, _event_metrics_from_samples(samples, thresholds), pr_auc


def _match_event_counts(
    truth: list[tuple[int, int, int]],
    predicted: list[tuple[int, int, int]],
    tolerance: int,
) -> dict[str, int]:
    unmatched = set(range(len(predicted)))
    true_positives = 0
    for true_event in truth:
        candidates = [
            index
            for index in unmatched
            if predicted[index][1] == true_event[1]
            and predicted[index][2] == true_event[2]
            and abs(predicted[index][0] - true_event[0]) <= tolerance
        ]
        if candidates:
            best = min(candidates, key=lambda index: abs(predicted[index][0] - true_event[0]))
            unmatched.remove(best)
            true_positives += 1
    return {"true": len(truth), "predicted": len(predicted), "tp": true_positives}


def _precision_recall_f1(counters: dict[str, int]) -> dict[str, float]:
    precision = counters["tp"] / max(counters["predicted"], 1)
    recall = counters["tp"] / max(counters["true"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
    }


def _binary_average_precision(scores: list[float], targets: list[int]) -> float:
    if not scores or not any(targets):
        return 0.0
    ranked = sorted(zip(scores, targets), key=lambda pair: pair[0], reverse=True)
    positives = sum(targets)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, target) in enumerate(ranked, start=1):
        if target:
            true_positives += 1
            precision_sum += true_positives / rank
    return 100.0 * precision_sum / positives


def _class_weights(records: list[StateGraphCacheRecord], field: str, classes: int) -> np.ndarray:
    counts = np.ones(classes, dtype=np.float64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            labels = arrays[field].reshape(-1)
        labels = labels[(labels >= 0) & (labels < classes)]
        counts += np.bincount(labels, minlength=classes)
    weights = counts.sum() / (classes * counts)
    return np.clip(weights, 0.25, 12.0).astype(np.float32)


def _action_factorization(
    metadata: dict[str, Any], records: list[StateGraphCacheRecord], num_steps: int
) -> dict[str, Any]:
    action_ids = list(metadata.get("action_ids", []))
    descriptions = list(metadata.get("action_descriptions", action_ids))
    if len(descriptions) != num_steps:
        descriptions = action_ids
    factors: list[tuple[str, str]] = []
    for description in descriptions:
        tokens = [token for token in re.split(r"[^a-z0-9]+", description.lower()) if token]
        verb = tokens[0] if tokens else "unknown"
        object_name = "_".join(tokens[1:]) if len(tokens) > 1 else "none"
        factors.append((verb, object_name))
    verb_names = sorted({verb for verb, _ in factors})
    object_names = sorted({object_name for _, object_name in factors})
    verb_to_index = {name: index for index, name in enumerate(verb_names)}
    object_to_index = {name: index for index, name in enumerate(object_names)}
    counts = np.zeros(num_steps, dtype=np.int64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            labels = arrays["step"].astype(np.int64)
        labels = labels[(labels >= 0) & (labels < num_steps)]
        counts += np.bincount(labels, minlength=num_steps)
    return {
        "verb_names": verb_names,
        "object_names": object_names,
        "verb_indices": [verb_to_index[verb] for verb, _ in factors],
        "object_indices": [object_to_index[object_name] for _, object_name in factors],
        "seen_mask": [bool(count) for count in counts],
        "unseen_train_actions": [
            action_ids[index] for index, count in enumerate(counts) if count == 0
        ],
    }


def _completion_pos_weights(
    records: list[StateGraphCacheRecord], components: int, cap: float = 100.0
) -> np.ndarray:
    positive = np.ones(components, dtype=np.float64)
    total_rows = 0
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            completion = arrays["completion"].astype(np.float64)
        positive += completion.sum(axis=0)
        total_rows += len(completion)
    negative = np.maximum(float(total_rows) - positive, 1.0)
    return np.clip(negative / positive, 1.0, cap).astype(np.float32)


def _infer_event_state_mapping(
    records: list[StateGraphCacheRecord], event_components: int, state_components: int
) -> dict[str, Any]:
    """Map each event component to the state column most consistent with its outcomes."""

    agreement = np.zeros((event_components, state_components), dtype=np.float64)
    observations = np.zeros((event_components, state_components), dtype=np.float64)
    # Event outcomes [correct, incorrect, remove] correspond most closely to
    # persistent state classes [correct=2, incorrect=0, pending=1].
    expected_state = np.asarray([2, 0, 1], dtype=np.int64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            outcomes = arrays["component_outcome"].astype(np.int64)
            state = arrays["state"].astype(np.int64)
            state_mask = arrays["state_mask"].astype(bool)
        rows, components = np.where(outcomes >= 0)
        for row, component in zip(rows, components):
            outcome = int(outcomes[row, component])
            valid = state_mask[row]
            observations[component, valid] += 1.0
            agreement[component, valid] += (
                state[row, valid] == expected_state[outcome]
            ).astype(np.float64)
    scores = agreement / np.maximum(observations, 1.0)
    indices = scores.argmax(axis=1).astype(np.int64)
    return {
        "indices": indices.tolist(),
        "agreement": [float(scores[index, state_index]) for index, state_index in enumerate(indices)],
        "observations": [
            int(observations[index, state_index]) for index, state_index in enumerate(indices)
        ],
    }


def _action_presence(
    records: list[StateGraphCacheRecord], num_steps: int
) -> list[bool]:
    counts = np.zeros(num_steps, dtype=np.int64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            labels = arrays["step"].astype(np.int64)
        labels = labels[(labels >= 0) & (labels < num_steps)]
        counts += np.bincount(labels, minlength=num_steps)
    return [bool(count) for count in counts]


def _move_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}


def _validate_records(records: list[StateGraphCacheRecord]) -> None:
    if not records:
        raise ValueError("The cache index is empty.")
    signature = (
        records[0].num_steps,
        records[0].motion_dim,
        records[0].appearance_dim,
        records[0].sensor_dim,
        records[0].num_components,
        records[0].num_completion_components,
    )
    for record in records[1:]:
        current = (
            record.num_steps,
            record.motion_dim,
            record.appearance_dim,
            record.sensor_dim,
            record.num_components,
            record.num_completion_components,
        )
        if current != signature:
            raise ValueError(f"Inconsistent cache dimensions: {record.recording_id} has {current}, expected {signature}")


def _validate_cache_schema(metadata: dict[str, Any], records: list[StateGraphCacheRecord]) -> None:
    version = int(metadata.get("schema_version", 1))
    if version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema v{version} cannot train the corrected Stage-1 objective. "
            f"Regenerate the cache with schema v{CACHE_SCHEMA_VERSION}; the old cache treats PSR completion IDs "
            "as mutually-exclusive steps and loses simultaneous events."
        )
    if metadata.get("label_contract") != "action_segmentation_plus_multicomponent_completion":
        raise ValueError("Cache is missing the corrected action/completion label contract.")
    if not records or records[0].num_completion_components <= 0:
        raise ValueError("Cache does not declare completion components.")
    if len(metadata.get("completion_components", [])) != records[0].num_completion_components:
        raise ValueError("Cache completion-component names do not match its tensor dimension.")
    if len(metadata.get("action_ids", [])) != records[0].num_steps:
        raise ValueError("Cache action IDs do not match its action-logit dimension.")
    required = {"motion", "appearance", "sensor", "step", "completion", "component_outcome"}
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{record.recording_id} is missing corrected cache arrays: {sorted(missing)}")


def _recording_level_split(
    records: list[StateGraphCacheRecord], validation_fraction: float, seed: int
) -> tuple[list[StateGraphCacheRecord], list[StateGraphCacheRecord]]:
    if len(records) < 2:
        raise ValueError(
            "At least two training recordings are required when the cache has no validation split. "
            "Add an official val recording to avoid window-level leakage."
        )
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * validation_fraction))))
    return shuffled[count:], shuffled[:count]


def _warmup_cosine(update: int, total: int, warmup: int) -> float:
    if update < warmup:
        return max((update + 1) / warmup, 1e-3)
    progress = min(1.0, (update - warmup) / max(total - warmup, 1))
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    model_config: StateGraphPSRConfig,
    loss_config: StateGraphLossConfig,
    metadata: dict[str, Any],
    transition_matrix: np.ndarray,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    import torch

    torch.save(
        {
            "architecture": "stategraph_psr_stage1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": model_config.to_dict(),
            "loss_config": asdict(loss_config),
            "dataset_metadata": metadata,
            "transition_matrix": transition_matrix,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StateGraph-PSR Lite on cached IndustReal features.")
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output-dir", default="runs/stategraph_psr_stage1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--sequence-stride", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-temporal-blocks", type=int, default=8)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph-strength", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--transition-smoothing", type=float, default=0.02)
    parser.add_argument(
        "--rare-window-boost",
        type=float,
        default=4.0,
        help="Sampling multiplier for train windows containing incorrect outcome/state labels.",
    )
    parser.add_argument(
        "--rare-windows-per-batch",
        type=int,
        default=1,
        help="Guarantee this many incorrect-event windows in each training batch.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--overfit-recording-id",
        action="append",
        default=None,
        help="Train and validate on this train-split recording for an intentional wiring test; repeat for 2-4 IDs.",
    )
    parser.add_argument("--step-weight", type=float, default=1.0)
    parser.add_argument("--completion-weight", type=float, default=0.45)
    parser.add_argument("--component-outcome-weight", type=float, default=0.7)
    parser.add_argument(
        "--completion-pos-weight-cap",
        type=float,
        default=100.0,
        help="Maximum per-component positive BCE weight for sparse completion events.",
    )
    parser.add_argument("--state-weight", type=float, default=0.8)
    parser.add_argument("--boundary-weight", type=float, default=0.25)
    parser.add_argument("--next-step-weight", type=float, default=0.3)
    parser.add_argument("--smoothing-weight", type=float, default=0.12)
    parser.add_argument("--graph-weight", type=float, default=0.15)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--normality-weight", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--asl-negative-gamma", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--normality-error-weight", type=float, default=8.0)
    parser.add_argument("--incorrect-selection-weight", type=float, default=0.75)
    parser.add_argument("--gpu-log-interval", type=int, default=10)
    parser.add_argument(
        "--calibration-interval",
        type=int,
        default=5,
        help="Run the expensive validation event-threshold sweep every N epochs.",
    )
    parser.add_argument("--disable-dashboard", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        parser.error("batch size and accumulation steps must be positive")
    if args.rare_window_boost < 1.0:
        parser.error("rare-window-boost must be at least 1")
    if args.completion_pos_weight_cap < 1.0:
        parser.error("completion-pos-weight-cap must be at least 1")
    if not 0.0 <= args.graph_strength <= 1.0:
        parser.error("graph-strength must be a probability in [0, 1]")
    if not 0 <= args.rare_windows_per_batch <= args.batch_size:
        parser.error("rare-windows-per-batch must be between zero and batch-size")
    if args.gpu_log_interval <= 0:
        parser.error("gpu-log-interval must be positive")
    if args.calibration_interval <= 0:
        parser.error("calibration-interval must be positive")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
