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
        sampling_weights: np.ndarray | None = None,
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
        if sampling_weights is None:
            sampling_weights = np.ones(dataset_size, dtype=np.float64)
        self.sampling_weights = np.asarray(sampling_weights, dtype=np.float64)
        if self.sampling_weights.shape != (dataset_size,):
            raise ValueError("sampling_weights must contain one value per dataset item")
        if not np.isfinite(self.sampling_weights).all() or (self.sampling_weights < 0).any():
            raise ValueError("sampling_weights must be finite and non-negative")
        weight_sum = float(self.sampling_weights.sum())
        if weight_sum <= 0:
            raise ValueError("sampling_weights must contain at least one positive value")
        self.sampling_probabilities = self.sampling_weights / weight_sum
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
                batch.extend(
                    int(value)
                    for value in rng.choice(
                        all_indices,
                        size=fill,
                        replace=True,
                        p=self.sampling_probabilities,
                    )
                )
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
        metadata,
    )
    active_action_mask = (
        _action_presence(train_records, num_steps)
        if overfit_recordings
        else [True] * num_steps
    )
    transition_matrix = build_transition_matrix(train_records, num_steps, args.transition_smoothing)
    model_config = StateGraphPSRConfig(
        motion_dim=train_records[0].motion_dim,
        motion_aux_dim=train_records[0].motion_aux_dim,
        appearance_dim=train_records[0].appearance_dim,
        sensor_dim=train_records[0].sensor_dim,
        num_steps=num_steps,
        num_action_verbs=len(factorization["verb_names"]),
        num_action_objects=len(factorization["object_names"]),
        action_verb_indices=tuple(factorization["verb_indices"]),
        action_object_indices=tuple(factorization["object_indices"]),
        seen_action_mask=tuple(factorization["seen_mask"]),
        active_action_mask=tuple(active_action_mask),
        mistake_action_mask=tuple(factorization["mistake_action_mask"]),
        action_event_component_indices=tuple(
            factorization["action_event_component_indices"]
        ),
        event_state_indices=tuple(event_state_mapping["indices"]),
        num_completion_components=train_records[0].num_completion_components,
        num_event_outcomes=len(metadata["event_outcomes"]),
        num_components=train_records[0].num_components,
        hidden_dim=args.hidden_dim,
        num_temporal_blocks=args.num_temporal_blocks,
        attention_every=args.attention_every,
        num_heads=args.num_heads,
        num_action_refinement_stages=args.num_action_refinement_stages,
        num_refinement_blocks=args.num_refinement_blocks,
        num_event_blocks=args.num_event_blocks,
        dropout=args.dropout,
        graph_strength_init=args.graph_strength,
        max_sequence_length=args.max_sequence_length,
        positional_dropout=args.positional_dropout,
    )
    loss_config = StateGraphLossConfig(
        step_weight=args.step_weight,
        completion_weight=args.completion_weight,
        component_outcome_weight=args.component_outcome_weight,
        incorrect_onset_weight=args.incorrect_onset_weight,
        state_weight=args.state_weight,
        boundary_weight=args.boundary_weight,
        next_step_weight=args.next_step_weight,
        smoothing_weight=args.smoothing_weight,
        graph_weight=args.graph_weight,
        consistency_weight=args.consistency_weight,
        progress_weight=args.progress_weight,
        normality_weight=args.normality_weight,
        refinement_weight=args.refinement_weight,
        focal_gamma=args.focal_gamma,
        asl_negative_gamma=args.asl_negative_gamma,
        asl_clip=args.asl_clip,
        normality_error_weight=args.normality_error_weight,
        event_label_horizon=args.event_label_horizon,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_stategraph_psr(model_config, transition_matrix).to(device)
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        if initialization.get("model_config") != model_config.to_dict():
            raise ValueError("Initialization checkpoint model configuration does not match.")
        model.load_state_dict(initialization["model_state"])
        print(f"Initialized model weights from {args.init_checkpoint}", flush=True)
    if args.freeze_action_backbone:
        trainable_prefixes = (
            "event_temporal_blocks.",
            "state_head.",
            "action_event_context.",
            "state_event_context.",
            "event_norm.",
            "completion_head.",
            "component_outcome_head.",
            "incorrect_onset_head.",
            "component_normal_prototypes",
            "normality_scale_raw",
            "normality_bias",
            "anomaly_strength_raw",
            "state_evidence_strength_raw",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
    criterion = build_stategraph_loss(loss_config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_training_vram_gib = _enforce_vram_limit(
        torch, device, args.max_vram_gib, phase="initialization"
    )
    step_weights = torch.from_numpy(
        _class_weights(
            train_records,
            "step",
            num_steps,
            power=args.step_class_weight_power,
        )
    ).to(device)
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
    state_class_weights = torch.from_numpy(
        _state_class_weights(
            train_records,
            model_config.num_components,
            cap=args.state_class_weight_cap,
            power=args.state_class_weight_power,
        )
    ).to(device)
    incorrect_pos_weights = torch.from_numpy(
        _outcome_pos_weights(
            train_records,
            model_config.num_completion_components,
            outcome_index=1,
            cap=args.incorrect_pos_weight_cap,
        )
    ).to(device)

    train_dataset = StateGraphCacheDataset(
        train_records,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        training=True,
        seed=args.seed,
        preload=args.preload_cache,
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
        sampling_weights=sampling_weights,
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
            preload=args.preload_cache,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pad_stategraph_batch,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
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
    if not args.resume:
        history_path.write_text("", encoding="utf-8")
    monitor = TrainingMonitor(output_dir, enabled=not args.disable_dashboard)
    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    patience_left = args.patience
    global_update = 0
    event_thresholds: list[float] | None = None
    state_incorrect_threshold: float | None = None
    fps = float(metadata.get("fps", 10.0))
    stride_frames = float(metadata.get("stride_frames", 5.0))
    seconds_per_step = stride_frames / fps if fps > 0 and stride_frames > 0 else 0.5
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.parent.resolve() != output_dir.resolve():
            raise ValueError("--resume must point to last_checkpoint.pt inside --output-dir.")
        if not (output_dir / "best_checkpoint.pt").exists():
            raise ValueError("Resume output directory is missing best_checkpoint.pt.")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != model_config.to_dict():
            raise ValueError("Resume checkpoint model configuration does not match this run.")
        if checkpoint.get("loss_config") != asdict(loss_config):
            raise ValueError("Resume checkpoint loss configuration does not match this run.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        training_state = checkpoint.get("training_state") or {}
        start_epoch = int(checkpoint["epoch"]) + 1
        checkpoint_metrics = checkpoint.get("metrics")
        if training_state:
            best_score = float(training_state.get("best_score", best_score))
            best_metrics = training_state.get("best_metrics", best_metrics)
        elif checkpoint_metrics:
            # A best checkpoint intentionally omits mutable training state. It
            # can still recover an interrupted run: its own calibrated metrics
            # are authoritative for selection, while optimizer/scheduler state
            # is already stored alongside the weights.
            best_metrics = checkpoint_metrics
            best_score = _validation_selection_score(
                checkpoint_metrics, mistake_weight=args.incorrect_selection_weight
            )
        patience_left = int(training_state.get("patience_left", patience_left))
        global_update = int(
            training_state.get("global_update", max(0, scheduler.last_epoch))
        )
        fallback_event_thresholds = (
            [
                checkpoint_metrics["correct_event_threshold"],
                checkpoint_metrics["incorrect_event_threshold"],
                checkpoint_metrics["remove_event_threshold"],
            ]
            if checkpoint_metrics
            else event_thresholds
        )
        event_thresholds = training_state.get(
            "event_thresholds", fallback_event_thresholds
        )
        state_incorrect_threshold = training_state.get(
            "state_incorrect_threshold",
            checkpoint_metrics.get("state_incorrect_threshold")
            if checkpoint_metrics
            else state_incorrect_threshold,
        )
        batch_sampler.epoch = int(
            training_state.get("batch_sampler_epoch", checkpoint["epoch"])
        )
        _restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resuming {resume_path} at epoch {start_epoch}", flush=True)
    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint already completed epoch {start_epoch - 1}, "
            f"which is not below --epochs {args.epochs}."
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_dataset.set_epoch(epoch)
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
                        motion_aux=batch.get("motion_aux"),
                    )
                    losses = criterion(
                        outputs,
                        batch,
                        transition_matrix=model.transition_matrix,
                        step_class_weights=step_weights,
                        completion_pos_weights=completion_pos_weights,
                        component_outcome_class_weights=component_outcome_weights,
                        state_class_weights=state_class_weights,
                        incorrect_pos_weights=incorrect_pos_weights,
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
                    peak_training_vram_gib = max(
                        peak_training_vram_gib,
                        _enforce_vram_limit(
                            torch, device, args.max_vram_gib, phase="training"
                        ),
                    )
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
                seconds_per_step=seconds_per_step,
                event_thresholds=event_thresholds,
                calibrate_events=calibrate_events,
                state_incorrect_threshold=state_incorrect_threshold,
                calibrate_state=calibrate_events,
            )
            event_thresholds = [
                metrics["correct_event_threshold"],
                metrics["incorrect_event_threshold"],
                metrics["remove_event_threshold"],
            ]
            state_incorrect_threshold = metrics["state_incorrect_threshold"]
            peak_training_vram_gib = max(
                peak_training_vram_gib,
                _enforce_vram_limit(
                    torch, device, args.max_vram_gib, phase="validation"
                ),
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "graph_strength": float(torch.sigmoid(model.graph_strength_raw).detach().cpu()),
            }
            score = _validation_selection_score(
                metrics, mistake_weight=args.incorrect_selection_weight
            )
            row["selection_score"] = score
            row["selection_eligible"] = calibrate_events
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

            if calibrate_events and score > best_score:
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
                should_stop = False
            else:
                patience_left -= 1
                should_stop = patience_left <= 0
            training_state = {
                "best_score": best_score,
                "best_metrics": best_metrics,
                "patience_left": patience_left,
                "global_update": global_update,
                "event_thresholds": event_thresholds,
                "state_incorrect_threshold": state_incorrect_threshold,
                "batch_sampler_epoch": batch_sampler.epoch,
            }
            _save_checkpoint(
                output_dir / "last_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                model_config,
                loss_config,
                metadata,
                transition_matrix,
                epoch,
                metrics,
                training_state=training_state,
                rng_state=_capture_rng_state(),
            )
            if should_stop:
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
        seconds_per_step=seconds_per_step,
        calibrate_events=True,
        calibrate_state=True,
    )
    validation_event_thresholds = [
        final_metrics["correct_event_threshold"],
        final_metrics["incorrect_event_threshold"],
        final_metrics["remove_event_threshold"],
    ]
    validation_state_incorrect_threshold = final_metrics[
        "state_incorrect_threshold"
    ]
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
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            use_amp,
            amp_dtype,
            model_config.num_components,
            seconds_per_step=seconds_per_step,
            event_thresholds=validation_event_thresholds,
            calibrate_events=False,
            state_incorrect_threshold=validation_state_incorrect_threshold,
            calibrate_state=False,
        )
    export_path = (
        Path(args.export_checkpoint)
        if args.export_checkpoint
        else output_dir / "inference_checkpoint.pt"
    )
    _save_inference_checkpoint(
        export_path,
        model,
        model_config,
        metadata,
        transition_matrix,
        validation_event_thresholds,
        final_metrics,
        test_metrics,
    )
    summary = {
        "architecture": "StateGraph-PSR Lite Stage 1",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "initialization_checkpoint": args.init_checkpoint,
        "freeze_action_backbone": args.freeze_action_backbone,
        "peak_training_vram_gib": peak_training_vram_gib,
        "maximum_allowed_vram_gib": args.max_vram_gib,
        "train_windows": len(train_dataset),
        "rare_train_windows": int((sampling_weights > 1.0).sum()),
        "rare_window_boost": args.rare_window_boost,
        "rare_windows_per_batch": args.rare_windows_per_batch,
        "train_recordings": len(train_records),
        "validation_recordings": len(val_records),
        "test_recordings": len(test_records),
        "overfit_recording_ids": overfit_recordings,
        "completion_pos_weight_cap": args.completion_pos_weight_cap,
        "step_class_weight_power": args.step_class_weight_power,
        "state_class_weight_power": args.state_class_weight_power,
        "state_class_weight_cap": args.state_class_weight_cap,
        "action_factorization": factorization,
        "event_state_mapping": event_state_mapping,
        "model_config": model_config.to_dict(),
        "loss_config": asdict(loss_config),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation": best_metrics,
        "final_validation": final_metrics,
        "test": test_metrics,
        "inference_checkpoint": str(export_path),
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
    seconds_per_step: float = 0.5,
    event_thresholds: list[float] | None = None,
    calibrate_events: bool = True,
    calibrate_latency: bool = False,
    state_incorrect_threshold: float | None = None,
    calibrate_state: bool = True,
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
    state_score_chunks: list[np.ndarray] = []
    state_truth_chunks: list[np.ndarray] = []
    state_other_prediction_chunks: list[np.ndarray] = []
    evaluated_steps = 0
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
                    motion_aux=batch.get("motion_aux"),
                )
            step_prediction = outputs["step_logits"].argmax(dim=-1)
            raw_step_prediction = outputs["raw_step_logits"].argmax(dim=-1)
            completion_score = torch.sigmoid(outputs["completion_logits"])
            component_outcome_probability = torch.softmax(
                outputs["component_outcome_logits"], dim=-1
            )
            state_probabilities = torch.softmax(outputs["state_logits"], dim=-1)
            normality_anomaly_score = torch.sigmoid(-outputs["normality_logits"])
            if "mistake_action_probability" in outputs:
                normality_anomaly_score = torch.maximum(
                    normality_anomaly_score, outputs["mistake_action_probability"]
                )
            state_incorrect_probability = outputs["state_outcome_probabilities"][..., 1]
            previous_state_incorrect = torch.nn.functional.pad(
                state_incorrect_probability[:, :-1], (0, 0, 1, 0), value=0.0
            )
            # A probabilistic onset is a positive transition, not p_t*(1-p_t-1):
            # the latter remains non-zero in a steady state and creates repeated
            # false alerts.  Positive differencing is causal and becomes quiet
            # once the predicted incorrect state stabilises.
            state_incorrect_onset = (
                state_incorrect_probability - previous_state_incorrect
            ).clamp_min(0.0)
            prototype_state_fault_score = torch.sqrt(
                (state_incorrect_onset * normality_anomaly_score).clamp_min(0.0)
            )
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
            if state_valid.any():
                state_score = state_probabilities[..., 0]
                # An explicit failed-action class is strong component-specific
                # evidence for the otherwise extremely rare incorrect state.
                if "mistake_action_probability" in outputs:
                    state_score = torch.maximum(
                        state_score, outputs["mistake_action_probability"]
                    )
                state_score_chunks.append(
                    state_score[state_valid].detach().float().cpu().numpy()
                )
                state_truth_chunks.append(
                    batch["state"][state_valid].detach().cpu().numpy()
                )
                state_other_prediction_chunks.append(
                    (state_probabilities[..., 1:].argmax(dim=-1) + 1)[state_valid]
                    .detach()
                    .cpu()
                    .numpy()
                )
            for sample_index in range(step_prediction.shape[0]):
                length = int(batch["valid_mask"][sample_index].sum().item())
                evaluated_steps += length
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
                event_scores = event_scores.clone()
                event_scores[..., 1] = torch.maximum(
                    event_scores[..., 1],
                    prototype_state_fault_score[sample_index, :length],
                )
                event_scores[..., 1] = torch.maximum(
                    event_scores[..., 1],
                    torch.sigmoid(outputs["incorrect_onset_logits"])[
                        sample_index, :length
                    ]
                    * normality_anomaly_score[sample_index, :length],
                )
                if "mistake_action_onset_score" in outputs:
                    event_scores[..., 1] = torch.maximum(
                        event_scores[..., 1],
                        outputs["mistake_action_onset_score"][sample_index, :length],
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
    primary_tolerance = max(1, round(1.0 / seconds_per_step))
    if calibrate_events or event_thresholds is None:
        event_thresholds, calibrated_metrics, event_pr_auc = _calibrate_event_thresholds(
            event_samples,
            outcomes=3,
            tolerance=primary_tolerance,
            seconds_per_step=seconds_per_step,
        )
    else:
        calibrated_metrics = _event_metrics_from_samples(
            event_samples,
            event_thresholds,
            tolerance=primary_tolerance,
            seconds_per_step=seconds_per_step,
        )
        event_pr_auc = [float("nan")] * 3
    fixed_metrics = _event_metrics_from_samples(
        event_samples,
        [0.1, 0.1, 0.1],
        tolerance=primary_tolerance,
        seconds_per_step=seconds_per_step,
    )
    # Keep the primary event metric at +/-1 second regardless of cache sampling
    # rate. Wider 2 s/4 s windows diagnose latency; calibrating them independently
    # prevents a strict-window threshold from hiding delayed but useful evidence.
    latency_metrics: dict[int, tuple[list[float], dict[Any, dict[str, float]]]] = {}
    for tolerance_seconds in (2, 4):
        tolerance = max(1, round(tolerance_seconds / seconds_per_step))
        if calibrate_events and calibrate_latency:
            latency_thresholds, metrics_at_tolerance, _ = _calibrate_event_thresholds(
                event_samples,
                outcomes=3,
                tolerance=tolerance,
                seconds_per_step=seconds_per_step,
            )
        else:
            latency_thresholds = list(event_thresholds)
            metrics_at_tolerance = _event_metrics_from_samples(
                event_samples,
                latency_thresholds,
                tolerance=tolerance,
                seconds_per_step=seconds_per_step,
            )
        latency_metrics[tolerance_seconds] = (latency_thresholds, metrics_at_tolerance)
    all_event_metrics = calibrated_metrics["all"]
    correct_metrics = calibrated_metrics[0]
    incorrect_metrics = calibrated_metrics[1]
    remove_metrics = calibrated_metrics[2]
    normality_ap = _binary_average_precision(normality_scores, normality_targets)
    incorrect_diagnostics = _event_timing_diagnostics(
        event_samples,
        event_thresholds,
        outcome=1,
        total_duration_seconds=evaluated_steps * seconds_per_step,
        seconds_per_step=seconds_per_step,
        match_tolerance=primary_tolerance,
    )
    correct_normality = [
        score for score, target in zip(normality_scores, normality_targets) if target == 0
    ]
    incorrect_normality = [
        score for score, target in zip(normality_scores, normality_targets) if target == 1
    ]
    state_scores = np.concatenate(state_score_chunks) if state_score_chunks else np.empty(0)
    state_truth = np.concatenate(state_truth_chunks) if state_truth_chunks else np.empty(0, dtype=np.int64)
    state_other_prediction = (
        np.concatenate(state_other_prediction_chunks)
        if state_other_prediction_chunks
        else np.empty(0, dtype=np.int64)
    )
    if calibrate_state or state_incorrect_threshold is None:
        state_incorrect_threshold, state_confusion = _calibrate_incorrect_state_threshold(
            state_scores, state_truth, state_other_prediction
        )
    else:
        state_confusion = _state_confusion_at_threshold(
            state_scores,
            state_truth,
            state_other_prediction,
            state_incorrect_threshold,
        )
    state_total = int(state_confusion.sum())
    state_correct = int(np.trace(state_confusion))
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
        "event_match_tolerance_seconds": primary_tolerance * seconds_per_step,
        "incorrect_event_f1_tol2s": latency_metrics[2][1][1]["f1"],
        "incorrect_event_recall_tol2s": latency_metrics[2][1][1]["recall"],
        "incorrect_event_threshold_tol2s": latency_metrics[2][0][1],
        "incorrect_event_f1_tol4s": latency_metrics[4][1][1]["f1"],
        "incorrect_event_recall_tol4s": latency_metrics[4][1][1]["recall"],
        "incorrect_event_threshold_tol4s": latency_metrics[4][0][1],
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
        "state_macro_f1": _macro_f1_from_confusion(state_confusion),
        "state_incorrect_f1": _class_f1_from_confusion(state_confusion, 0),
        "state_incorrect_threshold": float(state_incorrect_threshold),
        "incorrect_detection_delay_steps": incorrect_diagnostics["mean_delay_steps"],
        "incorrect_detection_delay_seconds": incorrect_diagnostics["mean_delay_steps"]
        * seconds_per_step,
        "incorrect_delayed_detections": incorrect_diagnostics["matched_detections"],
        "incorrect_false_alerts_per_minute": incorrect_diagnostics[
            "false_alerts_per_minute"
        ],
    }


def _target_events(component_outcome: np.ndarray) -> list[tuple[int, int, int]]:
    rows, components = np.where(component_outcome >= 0)
    return [(int(row), int(component), int(component_outcome[row, component])) for row, component in zip(rows, components)]


def _class_f1_from_confusion(confusion: np.ndarray, class_index: int) -> float:
    true_positive = int(confusion[class_index, class_index])
    false_positive = int(confusion[:, class_index].sum()) - true_positive
    false_negative = int(confusion[class_index, :].sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * (2 * true_positive / denominator) if denominator else 0.0


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    supported = [index for index in range(confusion.shape[0]) if confusion[index].sum() > 0]
    if not supported:
        return 0.0
    return float(np.mean([_class_f1_from_confusion(confusion, index) for index in supported]))


def _state_confusion_at_threshold(
    incorrect_scores: np.ndarray,
    truth: np.ndarray,
    other_prediction: np.ndarray,
    threshold: float,
) -> np.ndarray:
    prediction = np.where(incorrect_scores >= threshold, 0, other_prediction)
    return np.bincount(3 * truth + prediction, minlength=9).reshape(3, 3)


def _calibrate_incorrect_state_threshold(
    incorrect_scores: np.ndarray,
    truth: np.ndarray,
    other_prediction: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Tune the rare incorrect-state decision on validation only.

    Argmax is poorly calibrated when incorrect states occupy under one percent
    of component rows. A compact fixed grid is deterministic, inexpensive, and
    yields a threshold that can be frozen for actor-held-out testing.
    """

    if not len(truth):
        return 0.5, np.zeros((3, 3), dtype=np.int64)
    best_threshold = 0.5
    best_confusion = _state_confusion_at_threshold(
        incorrect_scores, truth, other_prediction, best_threshold
    )
    best_f1 = _class_f1_from_confusion(best_confusion, 0)
    for threshold in np.linspace(0.01, 0.99, 99):
        confusion = _state_confusion_at_threshold(
            incorrect_scores, truth, other_prediction, float(threshold)
        )
        f1 = _class_f1_from_confusion(confusion, 0)
        # Prefer the higher threshold on ties to reduce false incorrect states.
        if f1 > best_f1 + 1e-12 or (
            abs(f1 - best_f1) <= 1e-12 and threshold > best_threshold
        ):
            best_f1 = f1
            best_threshold = float(threshold)
            best_confusion = confusion
    return best_threshold, best_confusion


def _event_timing_diagnostics(
    samples: list[tuple[list[tuple[int, int, int]], np.ndarray]],
    thresholds: list[float] | tuple[float, ...],
    outcome: int,
    total_duration_seconds: float,
    seconds_per_step: float = 0.5,
    match_tolerance: int = 1,
) -> dict[str, float]:
    """Report causal alert delay and strict false-alert rate for one outcome."""

    delays: list[int] = []
    predicted_total = 0
    strict_true_positives = 0
    for truth_events, event_scores in samples:
        truth = [event for event in truth_events if event[2] == outcome]
        predicted = [
            event
            for event in _predicted_events_from_scores(
                event_scores, thresholds, seconds_per_step=seconds_per_step
            )
            if event[2] == outcome
        ]
        predicted_total += len(predicted)
        strict_true_positives += _match_event_counts(
            truth, predicted, tolerance=match_tolerance
        )["tp"]
        unmatched = set(range(len(predicted)))
        for true_row, true_component, _ in sorted(truth):
            candidates = [
                index
                for index in unmatched
                if predicted[index][1] == true_component and predicted[index][0] >= true_row
            ]
            if candidates:
                best = min(candidates, key=lambda index: predicted[index][0])
                unmatched.remove(best)
                delays.append(predicted[best][0] - true_row)
    false_alerts = max(0, predicted_total - strict_true_positives)
    return {
        "mean_delay_steps": float(np.mean(delays)) if delays else -1.0,
        "matched_detections": float(len(delays)),
        "false_alerts_per_minute": 60.0 * false_alerts / max(total_duration_seconds, 1e-6),
    }


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
    minimum_distance: int | None = None,
    install_minimum_distance: int | None = None,
    seconds_per_step: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Propose and suppress peaks independently for each event outcome.

    A peak in the sum over outcomes can be controlled by the majority ``correct``
    curve and hide a smaller, temporally sharper fault peak.  We therefore form
    candidates per outcome. NMS is deliberately outcome-specific: Assembly101
    commonly records an incorrect installation attempt followed shortly by a
    correct retry, so suppressing nearby peaks *across* outcomes erases exactly
    the mistake events we need to detect. ``install_minimum_distance`` remains
    accepted for checkpoint/tool compatibility but is no longer used for
    cross-outcome suppression. Durations are converted into cache steps so
    changing visual sampling rate cannot change decoder timing.
    """

    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    minimum_distance = minimum_distance or max(1, round(4.0 / seconds_per_step))
    events: list[tuple[int, int, int]] = []
    for component in range(event_scores.shape[1]):
        candidates: list[tuple[float, float, int, int]] = []
        for outcome, threshold in enumerate(thresholds):
            scores = event_scores[:, component, outcome]
            previous = np.concatenate((np.asarray([-np.inf]), scores[:-1]))
            following = np.concatenate((scores[1:], np.asarray([-np.inf])))
            peak_rows = np.flatnonzero(
                (scores >= previous)
                & (scores > following)
                & (scores >= float(threshold))
            )
            for row in peak_rows:
                score = float(scores[row])
                calibrated_confidence = float(score) / max(float(threshold), 1e-6)
                candidates.append((calibrated_confidence, score, int(row), outcome))
        selected_mask = np.zeros((event_scores.shape[2], event_scores.shape[0]), dtype=np.bool_)
        for _, _, row, outcome in sorted(candidates, reverse=True):
            near_start = max(0, row - minimum_distance + 1)
            near_end = min(event_scores.shape[0], row + minimum_distance)
            conflicts = bool(selected_mask[outcome, near_start:near_end].any())
            # At a single timestamp/component only one mutually exclusive
            # outcome can occur. Nearby timestamps, however, may be a failed
            # attempt and its successful retry and must both survive.
            conflicts = conflicts or bool(selected_mask[:, row].any())
            if not conflicts:
                selected_mask[outcome, row] = True
                events.append((row, component, outcome))
    return sorted(events)


def _event_metrics_from_samples(
    samples: list[tuple[list[tuple[int, int, int]], np.ndarray]],
    thresholds: list[float] | tuple[float, ...],
    tolerance: int = 1,
    seconds_per_step: float = 0.5,
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
        predicted_events = _predicted_events_from_scores(
            event_scores, thresholds, seconds_per_step=seconds_per_step
        )
        matched = _match_event_counts(truth_events, predicted_events, tolerance=tolerance)
        for key in counters["all"]:
            counters["all"][key] += matched[key]
        for outcome in range(outcomes):
            matched = _match_event_counts(
                [event for event in truth_events if event[2] == outcome],
                [event for event in predicted_events if event[2] == outcome],
                tolerance=tolerance,
            )
            for key in counters[outcome]:
                counters[outcome][key] += matched[key]
    return {key: _precision_recall_f1(value) for key, value in counters.items()}


def _calibrate_event_thresholds(
    samples: list[tuple[list[tuple[int, int, int]], np.ndarray]],
    outcomes: int,
    tolerance: int = 1,
    seconds_per_step: float = 0.5,
) -> tuple[list[float], dict[Any, dict[str, float]], list[float]]:
    """Select validation thresholds and event-level PR-AUC independently by outcome."""

    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.2, 20),
                np.linspace(0.225, 0.9, 28),
                np.asarray([0.925, 0.95, 0.975, 0.99, 0.995, 0.999]),
            ]
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
            metrics = _event_metrics_from_samples(
                samples,
                candidate,
                tolerance=tolerance,
                seconds_per_step=seconds_per_step,
            )[outcome]
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
    return (
        thresholds,
        _event_metrics_from_samples(
            samples,
            thresholds,
            tolerance=tolerance,
            seconds_per_step=seconds_per_step,
        ),
        pr_auc,
    )


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


def _validation_selection_score(
    metrics: dict[str, float], mistake_weight: float = 1.0
) -> float:
    """Balanced validation-only checkpoint criterion for the full task contract."""
    general = (
        0.25 * metrics["f1@50"]
        + 0.10 * metrics["edit"]
        + 0.10 * metrics["frame_accuracy"]
        + 0.15 * metrics["state_macro_f1"]
    )
    mistake = (
        0.15 * metrics["state_incorrect_f1"]
        + 0.15 * metrics["incorrect_event_f1"]
        + 0.10 * metrics["normality_incorrect_average_precision"]
        - 0.25 * min(metrics["incorrect_false_alerts_per_minute"], 20.0)
    )
    return general + mistake_weight * mistake


def _class_weights(
    records: list[StateGraphCacheRecord],
    field: str,
    classes: int,
    power: float = 1.0,
) -> np.ndarray:
    counts = np.ones(classes, dtype=np.float64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            labels = arrays[field].reshape(-1)
        labels = labels[(labels >= 0) & (labels < classes)]
        counts += np.bincount(labels, minlength=classes)
    weights = (counts.sum() / (classes * counts)) ** power
    return np.clip(weights, 0.25, 12.0).astype(np.float32)


def _action_factorization(
    metadata: dict[str, Any], records: list[StateGraphCacheRecord], num_steps: int
) -> dict[str, Any]:
    action_ids = list(metadata.get("action_ids", []))
    descriptions = list(metadata.get("action_descriptions", action_ids))
    if len(descriptions) != num_steps:
        descriptions = action_ids
    taxonomy = metadata.get("action_taxonomy", [])
    official_factors = {
        str(row["action_cls"]).strip().lower(): (
            str(row["verb_cls"]).strip().lower().replace(" ", "_"),
            str(row["noun_cls"]).strip().lower().replace(" ", "_"),
        )
        for row in taxonomy
        if all(key in row for key in ("action_cls", "verb_cls", "noun_cls"))
    }
    factors: list[tuple[str, str]] = []
    taxonomy_hits = 0
    for action_id, description in zip(action_ids, descriptions):
        if action_id.lower() in official_factors:
            factors.append(official_factors[action_id.lower()])
            taxonomy_hits += 1
            continue
        tokens = [token for token in re.split(r"[^a-z0-9]+", description.lower()) if token]
        verb = tokens[0] if tokens else "unknown"
        object_name = "_".join(tokens[1:]) if len(tokens) > 1 else "none"
        factors.append((verb, object_name))
    verb_names = sorted({verb for verb, _ in factors})
    object_names = sorted({object_name for _, object_name in factors})
    verb_to_index = {name: index for index, name in enumerate(verb_names)}
    object_to_index = {name: index for index, name in enumerate(object_names)}
    completion_names = [
        str(name).strip().lower().replace(" ", "_")
        for name in metadata.get("completion_components", [])
    ]
    completion_lookup = {name: index for index, name in enumerate(completion_names)}
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
        "mistake_action_mask": [verb.startswith("attempt_to_") for verb, _ in factors],
        "action_event_component_indices": [
            completion_lookup.get(object_name, -1) for _, object_name in factors
        ],
        "seen_mask": [bool(count) for count in counts],
        "factor_source": (
            "official_action_taxonomy" if taxonomy_hits == max(0, num_steps - 1) else "mixed"
        ),
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


def _state_class_weights(
    records: list[StateGraphCacheRecord],
    components: int,
    cap: float = 12.0,
    power: float = 0.5,
) -> np.ndarray:
    """Per-component state balancing without letting rare faults explode gradients."""
    counts = np.ones((components, 3), dtype=np.float64)
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            state = arrays["state"].astype(np.int64)
            mask = arrays["state_mask"].astype(np.bool_)
        for component in range(components):
            labels = state[:, component][mask[:, component]]
            counts[component] += np.bincount(labels, minlength=3)
    # Inverse square root is more stable than full inverse frequency for the
    # very scarce incorrect state, while still countering pending-state scale.
    weights = (counts.sum(axis=1, keepdims=True) / (3.0 * counts)) ** power
    weights = np.clip(weights, 0.25, cap)
    weights /= weights.mean(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _outcome_pos_weights(
    records: list[StateGraphCacheRecord],
    components: int,
    outcome_index: int,
    cap: float = 100.0,
) -> np.ndarray:
    positive = np.ones(components, dtype=np.float64)
    total_rows = 0
    for record in records:
        with np.load(record.path, allow_pickle=False) as arrays:
            outcomes = arrays["component_outcome"].astype(np.int64)
        positive += (outcomes == outcome_index).sum(axis=0)
        total_rows += len(outcomes)
    negative = np.maximum(float(total_rows) - positive, 1.0)
    return np.clip(negative / positive, 1.0, cap).astype(np.float32)


def _infer_event_state_mapping(
    records: list[StateGraphCacheRecord],
    event_components: int,
    state_components: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map each event component to the state column most consistent with its outcomes."""

    # Dataset adapters should declare semantic component names whenever
    # possible. Exact name alignment is stronger and more stable than inferring
    # a mapping from correlated assembly states (several parts are often
    # installed together, making argmax agreement ambiguous).
    if metadata is not None:
        event_names = list(metadata.get("completion_components", []))
        state_names = list(metadata.get("state_components", []))
        if len(event_names) == event_components and len(state_names) == state_components:
            state_lookup = {name: index for index, name in enumerate(state_names)}
            if len(state_lookup) == len(state_names) and all(name in state_lookup for name in event_names):
                return {
                    "indices": [state_lookup[name] for name in event_names],
                    "agreement": [1.0] * event_components,
                    "observations": [0] * event_components,
                    "per_outcome_agreement": [[1.0, 1.0, 1.0] for _ in event_names],
                    "source": "metadata_component_names",
                }

    agreement = np.zeros((event_components, state_components, 3), dtype=np.float64)
    observations = np.zeros((event_components, state_components, 3), dtype=np.float64)
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
            end = min(len(state), row + 3)
            valid = state_mask[row:end].any(axis=0)
            observations[component, valid, outcome] += 1.0
            matches = (
                (state[row:end] == expected_state[outcome]) & state_mask[row:end]
            ).any(axis=0)
            agreement[component, valid, outcome] += matches[valid].astype(np.float64)
    per_outcome_scores = agreement / np.maximum(observations, 1.0)
    outcome_weights = np.asarray([0.3, 0.6, 0.1], dtype=np.float64)
    available = observations > 0
    weighted = per_outcome_scores * outcome_weights
    normalizer = (available * outcome_weights).sum(axis=-1)
    scores = weighted.sum(axis=-1) / np.maximum(normalizer, 1e-12)
    indices = scores.argmax(axis=1).astype(np.int64)
    return {
        "indices": indices.tolist(),
        "agreement": [float(scores[index, state_index]) for index, state_index in enumerate(indices)],
        "observations": [
            int(observations[index, state_index].sum())
            for index, state_index in enumerate(indices)
        ],
        "per_outcome_agreement": [
            per_outcome_scores[index, state_index].tolist()
            for index, state_index in enumerate(indices)
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
        records[0].motion_aux_dim,
        records[0].appearance_dim,
        records[0].sensor_dim,
        records[0].num_components,
        records[0].num_completion_components,
    )
    for record in records[1:]:
        current = (
            record.num_steps,
            record.motion_dim,
            record.motion_aux_dim,
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
    accepted_contracts = {
        "action_segmentation_plus_multicomponent_completion",
        "complete_coarse_action_plus_part_to_part_mistake_and_state",
    }
    if metadata.get("label_contract") not in accepted_contracts:
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
    training_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
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
            "training_state": training_state,
            "rng_state": rng_state,
        },
        path,
    )


def _save_inference_checkpoint(
    path: Path,
    model,
    model_config: StateGraphPSRConfig,
    metadata: dict[str, Any],
    transition_matrix: np.ndarray,
    event_thresholds: list[float],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> None:
    """Export compact optimizer-free BF16 weights for reproducible inference."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: (
            value.detach().cpu().to(torch.bfloat16)
            if value.is_floating_point()
            else value.detach().cpu()
        )
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "format_version": 1,
            "architecture": "stategraph_psr_stage1",
            "model_state": state,
            "model_config": model_config.to_dict(),
            "dataset_metadata": metadata,
            "transition_matrix": transition_matrix,
            "event_thresholds": event_thresholds,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


def _capture_rng_state() -> dict[str, Any]:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _enforce_vram_limit(torch, device, limit_gib: float, phase: str) -> float:
    if device.type != "cuda":
        return 0.0
    peak = max(
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    ) / 2**30
    if peak >= limit_gib:
        raise RuntimeError(
            f"{phase} exceeded the process VRAM limit: {peak:.2f} GiB >= {limit_gib:.2f} GiB"
        )
    return float(peak)


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location=device)`` also maps RNG byte tensors to
    # CUDA.  The default CPU generator only accepts a CPU ByteTensor, so move
    # checkpointed generator states back before restoring an exact resume.
    torch.random.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StateGraph-PSR Lite on cached IndustReal features.")
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output-dir", default="runs/stategraph_psr_stage1")
    parser.add_argument(
        "--export-checkpoint",
        default=None,
        help="Optional path for a compact BF16 inference checkpoint without optimizer state.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume exactly from a last_checkpoint.pt produced by this run.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights from a compatible training or inference checkpoint.",
    )
    parser.add_argument(
        "--freeze-action-backbone",
        action="store_true",
        help="Train only event/state modules after checkpoint initialization.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-vram-gib", type=float, default=23.0)
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
    parser.add_argument("--num-action-refinement-stages", type=int, default=0)
    parser.add_argument("--num-refinement-blocks", type=int, default=4)
    parser.add_argument("--num-event-blocks", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="Maximum cached temporal length supported by positional encoding.",
    )
    parser.add_argument(
        "--positional-dropout",
        type=float,
        default=0.0,
        help="Dropout applied to fixed temporal position features.",
    )
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
    parser.add_argument(
        "--preload-cache",
        action="store_true",
        help="Load compressed feature records once before worker fork to avoid repeated NFS decompression.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--overfit-recording-id",
        action="append",
        default=None,
        help="Train and validate on this train-split recording for an intentional wiring test; repeat for 2-4 IDs.",
    )
    parser.add_argument("--step-weight", type=float, default=1.0)
    parser.add_argument(
        "--step-class-weight-power",
        type=float,
        default=1.0,
        help="Exponent on inverse-frequency action weights; 0.5 is a softer square-root balance.",
    )
    parser.add_argument("--completion-weight", type=float, default=0.45)
    parser.add_argument("--component-outcome-weight", type=float, default=0.7)
    parser.add_argument("--incorrect-onset-weight", type=float, default=0.7)
    parser.add_argument(
        "--completion-pos-weight-cap",
        type=float,
        default=100.0,
        help="Maximum per-component positive BCE weight for sparse completion events.",
    )
    parser.add_argument("--state-weight", type=float, default=0.8)
    parser.add_argument(
        "--state-class-weight-power",
        type=float,
        default=0.5,
        help="Exponent on per-component inverse state frequency; 1.0 emphasizes rare faults.",
    )
    parser.add_argument(
        "--state-class-weight-cap",
        type=float,
        default=12.0,
        help="Cap before normalizing per-component inverse state weights.",
    )
    parser.add_argument("--boundary-weight", type=float, default=0.25)
    parser.add_argument("--next-step-weight", type=float, default=0.3)
    parser.add_argument("--smoothing-weight", type=float, default=0.12)
    parser.add_argument("--graph-weight", type=float, default=0.15)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--normality-weight", type=float, default=0.3)
    parser.add_argument("--refinement-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--asl-negative-gamma", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--normality-error-weight", type=float, default=8.0)
    parser.add_argument("--incorrect-pos-weight-cap", type=float, default=100.0)
    parser.add_argument(
        "--event-label-horizon",
        type=int,
        default=2,
        help="Causally repeat sparse event labels over this many following feature rows for training only.",
    )
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
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.freeze_action_backbone and not args.init_checkpoint:
        parser.error("--freeze-action-backbone requires --init-checkpoint")
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        parser.error("batch size and accumulation steps must be positive")
    if args.rare_window_boost < 1.0:
        parser.error("rare-window-boost must be at least 1")
    if args.completion_pos_weight_cap < 1.0:
        parser.error("completion-pos-weight-cap must be at least 1")
    if args.incorrect_pos_weight_cap < 1.0:
        parser.error("incorrect-pos-weight-cap must be at least 1")
    if args.state_class_weight_cap < 1.0:
        parser.error("state-class-weight-cap must be at least 1")
    if args.max_vram_gib <= 0:
        parser.error("max-vram-gib must be positive")
    if not 0.0 <= args.step_class_weight_power <= 1.0:
        parser.error("step-class-weight-power must be in [0, 1]")
    if not 0.0 <= args.state_class_weight_power <= 1.0:
        parser.error("state-class-weight-power must be in [0, 1]")
    if not 0.0 <= args.graph_strength <= 1.0:
        parser.error("graph-strength must be a probability in [0, 1]")
    if not 0 <= args.rare_windows_per_batch <= args.batch_size:
        parser.error("rare-windows-per-batch must be between zero and batch-size")
    if args.gpu_log_interval <= 0:
        parser.error("gpu-log-interval must be positive")
    if args.calibration_interval <= 0:
        parser.error("calibration-interval must be positive")
    if args.event_label_horizon < 0:
        parser.error("event-label-horizon cannot be negative")
    if min(
        args.num_action_refinement_stages,
        args.num_refinement_blocks,
        args.num_event_blocks,
    ) < 0:
        parser.error("refinement stage/block counts cannot be negative")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
