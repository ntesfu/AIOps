from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class TorchNotAvailableError(RuntimeError):
    pass


def _load_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised on training machines
        raise TorchNotAvailableError(
            "StateGraph-PSR requires PyTorch. Install the 'ml' extra in the CUDA environment."
        ) from exc
    return torch, nn, functional


@dataclass(frozen=True)
class StateGraphPSRConfig:
    motion_dim: int
    appearance_dim: int
    sensor_dim: int
    num_steps: int
    motion_aux_dim: int = 0
    num_action_verbs: int = 0
    num_action_objects: int = 0
    action_verb_indices: tuple[int, ...] = ()
    action_object_indices: tuple[int, ...] = ()
    seen_action_mask: tuple[bool, ...] = ()
    active_action_mask: tuple[bool, ...] = ()
    mistake_action_mask: tuple[bool, ...] = ()
    action_event_component_indices: tuple[int, ...] = ()
    event_state_indices: tuple[int, ...] = ()
    num_completion_components: int = 1
    num_event_outcomes: int = 3
    num_components: int = 11
    hidden_dim: int = 192
    num_temporal_blocks: int = 8
    attention_every: int = 2
    num_heads: int = 4
    num_action_refinement_stages: int = 0
    num_refinement_blocks: int = 4
    num_event_blocks: int = 0
    dropout: float = 0.2
    graph_strength_init: float = 0.12
    composition_strength_init: float = 1.0
    max_dilation: int = 16
    # Explicit order signal for attention layers.  The temporal convolutions are
    # translation equivariant, so without this signal two windows containing the
    # same actions at different offsets are indistinguishable to the attention
    # branch.  A fixed sinusoid keeps the model causal and adds no trainable
    # parameters (important when using long cached windows on a 24 GB GPU).
    max_sequence_length: int = 4096
    positional_dropout: float = 0.0
    # Feed explicit transition/next-action disagreement into the event branch.
    # This is optional so cached-feature ablations can isolate its contribution.
    procedural_event_context: bool = False
    learned_event_fusion: bool = False
    # Decompose sparse component onsets into a shared temporal detector and a
    # component selector. Disabled by default so legacy checkpoints retain an
    # identical parameter set and output behaviour.
    factorized_mistake_detection: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateGraphLossConfig:
    step_weight: float = 1.0
    completion_weight: float = 0.45
    component_outcome_weight: float = 0.7
    incorrect_onset_weight: float = 0.7
    state_weight: float = 0.8
    boundary_weight: float = 0.25
    next_step_weight: float = 0.3
    smoothing_weight: float = 0.12
    graph_weight: float = 0.15
    consistency_weight: float = 0.15
    progress_weight: float = 0.25
    normality_weight: float = 0.3
    refinement_weight: float = 0.5
    focal_gamma: float = 1.5
    asl_negative_gamma: float = 4.0
    asl_clip: float = 0.05
    normality_error_weight: float = 8.0
    event_label_horizon: int = 2
    # Zero preserves dense-negative BCE. A positive value keeps every mistake
    # target and only this many highest-scoring negatives per positive.
    incorrect_hard_negative_ratio: float = 0.0
    any_mistake_weight: float = 1.0
    mistake_component_weight: float = 1.0


def build_stategraph_psr(config: StateGraphPSRConfig, transition_matrix: Any | None = None):
    """Build the low-VRAM Stage-1 StateGraph-PSR model.

    Inputs are cached per-window features with shape ``[batch, time, dim]``.
    The visual backbones are deliberately outside this trainable graph.
    """

    if config.num_steps <= 0 or config.num_completion_components <= 0 or config.num_components <= 0:
        raise ValueError("Action, completion-component, and state-component counts must be positive.")
    if config.num_event_outcomes != 3:
        raise ValueError("Stage 1 expects three event outcomes: correct, incorrect, remove.")
    if config.hidden_dim % config.num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads.")
    if min(
        config.num_action_refinement_stages,
        config.num_refinement_blocks,
        config.num_event_blocks,
    ) < 0:
        raise ValueError("Refinement stage/block counts cannot be negative.")

    if config.action_verb_indices:
        if len(config.action_verb_indices) != config.num_steps:
            raise ValueError("action_verb_indices must contain one entry per action.")
        if len(config.action_object_indices) != config.num_steps:
            raise ValueError("action_object_indices must contain one entry per action.")
        if len(config.seen_action_mask) != config.num_steps:
            raise ValueError("seen_action_mask must contain one entry per action.")
        if config.active_action_mask and len(config.active_action_mask) != config.num_steps:
            raise ValueError("active_action_mask must contain one entry per action.")
        if config.mistake_action_mask and len(config.mistake_action_mask) != config.num_steps:
            raise ValueError("mistake_action_mask must contain one entry per action.")
        if config.action_event_component_indices:
            if len(config.action_event_component_indices) != config.num_steps:
                raise ValueError(
                    "action_event_component_indices must contain one entry per action."
                )
            if any(
                index < -1 or index >= config.num_completion_components
                for index in config.action_event_component_indices
            ):
                raise ValueError("action_event_component_indices contains an invalid component.")
    if config.event_state_indices:
        if len(config.event_state_indices) != config.num_completion_components:
            raise ValueError("event_state_indices must contain one state index per event component.")
        if any(index < 0 or index >= config.num_components for index in config.event_state_indices):
            raise ValueError("event_state_indices contains an out-of-range state component.")

    torch, nn, functional = _load_torch()

    class ProjectionStem(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.available = input_dim > 0
            if self.available:
                self.net = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )

        def forward(self, x):
            if not self.available:
                return None
            return self.net(x)

    class SinusoidalPositionEncoding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            length = max(1, int(config.max_sequence_length))
            position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, config.hidden_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / config.hidden_dim)
            )
            encoding = torch.zeros(length, config.hidden_dim, dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(position * div_term)
            # Odd hidden dimensions have one fewer cosine channel.
            encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
            self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
            self.dropout = nn.Dropout(config.positional_dropout)

        def forward(self, x):
            if x.shape[1] > self.encoding.shape[1]:
                raise ValueError(
                    f"Sequence length {x.shape[1]} exceeds max_sequence_length "
                    f"{self.encoding.shape[1]}"
                )
            return x + self.dropout(self.encoding[:, : x.shape[1]].to(dtype=x.dtype, device=x.device))

    class CausalDepthwiseBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.dilation = dilation
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.depthwise = nn.Conv1d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=3,
                dilation=dilation,
                groups=config.hidden_dim,
            )
            self.pointwise = nn.Conv1d(config.hidden_dim, 2 * config.hidden_dim, kernel_size=1)
            self.output = nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=1)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            residual = x
            out = self.norm(x).transpose(1, 2)
            out = functional.pad(out, (2 * self.dilation, 0))
            out = self.depthwise(out)
            value, gate = self.pointwise(out).chunk(2, dim=1)
            out = value * torch.sigmoid(gate)
            out = self.dropout(self.output(out).transpose(1, 2))
            return residual + out

    class CausalAttentionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.attention = nn.MultiheadAttention(
                config.hidden_dim,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(config.hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.hidden_dim, config.hidden_dim),
                nn.Dropout(config.dropout),
            )

        def forward(self, x, valid_mask):
            length = x.shape[1]
            causal_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
            )
            normalized = self.norm(x)
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=causal_mask,
                key_padding_mask=~valid_mask if valid_mask is not None else None,
                need_weights=False,
            )
            x = x + attended
            return x + self.ffn(self.ffn_norm(x))

    class TemporalBlock(nn.Module):
        def __init__(self, dilation: int, with_attention: bool) -> None:
            super().__init__()
            self.conv = CausalDepthwiseBlock(dilation)
            self.attention = CausalAttentionBlock() if with_attention else None

        def forward(self, x, valid_mask):
            x = self.conv(x)
            if self.attention is not None:
                x = self.attention(x, valid_mask)
            return x

    class CausalActionRefinementStage(nn.Module):
        """Refine class trajectories without looking at future frames."""

        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Sequential(
                nn.LayerNorm(config.num_steps),
                nn.Linear(config.num_steps, config.hidden_dim),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList(
                TemporalBlock(
                    min(2 ** (index % 5), config.max_dilation),
                    config.attention_every > 0
                    and (index + 1) % config.attention_every == 0,
                )
                for index in range(config.num_refinement_blocks)
            )
            self.output = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.num_steps),
            )

        def forward(self, logits, valid_mask):
            features = self.input(torch.softmax(logits, dim=-1))
            for block in self.blocks:
                features = block(features, valid_mask)
            return logits + self.output(features)

    class StateGraphPSRLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.motion_stem = ProjectionStem(config.motion_dim)
            self.motion_aux_stem = ProjectionStem(config.motion_aux_dim)
            self.appearance_stem = ProjectionStem(config.appearance_dim)
            self.sensor_stem = ProjectionStem(config.sensor_dim)
            modality_count = 4 if config.motion_aux_dim > 0 else 3
            self.modality_gate = nn.Sequential(
                nn.LayerNorm(config.hidden_dim * modality_count),
                nn.Linear(config.hidden_dim * modality_count, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, modality_count),
            )
            # A weighted average is useful for action recognition but can hide
            # precisely the cross-encoder mismatch that signals a wrong part,
            # tool, or state. Preserve that residual for the event-only branch.
            self.disagreement_stem = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.fusion_norm = nn.LayerNorm(config.hidden_dim)
            self.position_encoding = SinusoidalPositionEncoding()
            blocks = []
            for index in range(config.num_temporal_blocks):
                dilation = min(2 ** (index % 5), config.max_dilation)
                with_attention = config.attention_every > 0 and (index + 1) % config.attention_every == 0
                blocks.append(TemporalBlock(dilation, with_attention))
            self.temporal_blocks = nn.ModuleList(blocks)
            self.action_refinement_stages = nn.ModuleList(
                CausalActionRefinementStage()
                for _ in range(config.num_action_refinement_stages)
            )
            self.event_temporal_blocks = nn.ModuleList(
                TemporalBlock(
                    min(2 ** (index % 5), config.max_dilation),
                    config.attention_every > 0
                    and (index + 1) % config.attention_every == 0,
                )
                for index in range(config.num_event_blocks)
            )
            factorized = bool(config.action_verb_indices)
            num_verbs = config.num_action_verbs if factorized else config.num_steps
            num_objects = config.num_action_objects if factorized else 1
            verb_indices = (
                torch.as_tensor(config.action_verb_indices, dtype=torch.long)
                if factorized
                else torch.arange(config.num_steps, dtype=torch.long)
            )
            object_indices = (
                torch.as_tensor(config.action_object_indices, dtype=torch.long)
                if factorized
                else torch.zeros(config.num_steps, dtype=torch.long)
            )
            seen_mask = (
                torch.as_tensor(config.seen_action_mask, dtype=torch.float32)
                if factorized
                else torch.ones(config.num_steps, dtype=torch.float32)
            )
            active_mask = (
                torch.as_tensor(config.active_action_mask, dtype=torch.float32)
                if config.active_action_mask
                else torch.ones(config.num_steps, dtype=torch.float32)
            )
            self.register_buffer("action_verb_indices", verb_indices)
            self.register_buffer("action_object_indices", object_indices)
            self.register_buffer("seen_action_mask", seen_mask)
            self.register_buffer("active_action_mask", active_mask)
            mistake_action_mask = (
                torch.as_tensor(config.mistake_action_mask, dtype=torch.float32)
                if config.mistake_action_mask
                else torch.zeros(config.num_steps, dtype=torch.float32)
            )
            action_event_component_indices = (
                torch.as_tensor(config.action_event_component_indices, dtype=torch.long)
                if config.action_event_component_indices
                else torch.full((config.num_steps,), -1, dtype=torch.long)
            )
            self.register_buffer("mistake_action_mask", mistake_action_mask)
            self.register_buffer(
                "action_event_component_indices", action_event_component_indices
            )
            event_state_indices = (
                torch.as_tensor(config.event_state_indices, dtype=torch.long)
                if config.event_state_indices
                else torch.arange(config.num_completion_components, dtype=torch.long)
                % config.num_components
            )
            self.register_buffer("event_state_indices", event_state_indices)
            self.action_residual_prototypes = nn.Parameter(
                torch.empty(config.num_steps, config.hidden_dim)
            )
            self.verb_prototypes = nn.Parameter(torch.empty(num_verbs, config.hidden_dim))
            self.object_prototypes = nn.Parameter(torch.empty(num_objects, config.hidden_dim))
            nn.init.trunc_normal_(self.action_residual_prototypes, std=0.02)
            nn.init.trunc_normal_(self.verb_prototypes, std=0.02)
            nn.init.trunc_normal_(self.object_prototypes, std=0.02)
            self.prototype_attention = nn.MultiheadAttention(
                config.hidden_dim,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.prototype_norm = nn.LayerNorm(config.hidden_dim)
            self.step_classifier = nn.Linear(config.hidden_dim, config.num_steps)
            self.verb_classifier = nn.Linear(config.hidden_dim, num_verbs)
            self.object_classifier = nn.Linear(config.hidden_dim, num_objects)
            self.state_head = nn.Linear(config.hidden_dim, config.num_components * 3)
            self.action_event_context = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.state_event_context = nn.Linear(config.num_components * 3, config.hidden_dim)
            if config.procedural_event_context:
                self.procedure_event_context = nn.Sequential(
                    nn.LayerNorm(4 * config.num_completion_components + 2),
                    nn.Linear(
                        4 * config.num_completion_components + 2,
                        config.hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            self.event_norm = nn.LayerNorm(config.hidden_dim)
            self.completion_head = nn.Linear(config.hidden_dim, config.num_completion_components)
            self.component_outcome_head = nn.Linear(
                config.hidden_dim, config.num_completion_components * config.num_event_outcomes
            )
            self.incorrect_onset_head = nn.Linear(
                config.hidden_dim, config.num_completion_components
            )
            if config.factorized_mistake_detection:
                self.any_mistake_onset_head = nn.Linear(config.hidden_dim, 1)
            if config.learned_event_fusion:
                # Shared across components so every rare mistake teaches the
                # same calibration rule. Inputs remain interpretable and are
                # exposed separately for ablation/reporting.
                self.event_fusion_head = nn.Sequential(
                    nn.LayerNorm(5),
                    nn.Linear(5, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                )
            self.component_normal_prototypes = nn.Parameter(
                torch.empty(config.num_completion_components, config.hidden_dim)
            )
            nn.init.trunc_normal_(self.component_normal_prototypes, std=0.02)
            self.normality_scale_raw = nn.Parameter(torch.tensor(2.0))
            self.normality_bias = nn.Parameter(torch.tensor(0.0))
            self.anomaly_strength_raw = nn.Parameter(torch.tensor(-1.0))
            self.state_evidence_strength_raw = nn.Parameter(torch.tensor(0.0))
            self.boundary_head = nn.Linear(config.hidden_dim, 1)
            self.progress_head = nn.Linear(config.hidden_dim, 1)
            self.next_step_head = nn.Linear(config.hidden_dim, config.num_steps)
            graph_gate = min(max(float(config.graph_strength_init), 1e-4), 1.0 - 1e-4)
            self.graph_strength_raw = nn.Parameter(
                torch.tensor(math.log(graph_gate / (1.0 - graph_gate)))
            )
            self.composition_strength_raw = nn.Parameter(
                torch.tensor(float(config.composition_strength_init))
            )

            if transition_matrix is None:
                matrix = torch.ones(config.num_steps, config.num_steps, dtype=torch.float32)
            else:
                matrix = torch.as_tensor(transition_matrix, dtype=torch.float32)
                if tuple(matrix.shape) != (config.num_steps, config.num_steps):
                    raise ValueError(
                        f"transition_matrix must be [{config.num_steps}, {config.num_steps}], got {tuple(matrix.shape)}"
                    )
            matrix = matrix.clamp_min(0.0)
            matrix = matrix + torch.eye(config.num_steps, dtype=matrix.dtype) * 0.05
            matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1e-6)
            self.register_buffer("transition_matrix", matrix)

        def _project_modalities(self, motion, appearance, sensor, modality_mask, motion_aux=None):
            batch, length = motion.shape[:2]
            zeros = motion.new_zeros(batch, length, config.hidden_dim)
            projected = [
                self.motion_stem(motion) if config.motion_dim > 0 else zeros,
            ]
            if config.motion_aux_dim > 0:
                projected.append(
                    self.motion_aux_stem(motion_aux) if motion_aux is not None else zeros
                )
            projected.extend(
                [
                    self.appearance_stem(appearance) if config.appearance_dim > 0 else zeros,
                    self.sensor_stem(sensor) if config.sensor_dim > 0 else zeros,
                ]
            )
            stack = torch.stack(projected, dim=2)
            gate_input = torch.cat(projected, dim=-1)
            gate_logits = self.modality_gate(gate_input)
            if modality_mask is not None:
                if config.motion_aux_dim > 0 and modality_mask.shape[-1] == 3:
                    aux_mask = torch.full_like(modality_mask[..., :1], motion_aux is not None)
                    modality_mask = torch.cat(
                        [modality_mask[..., :1], aux_mask, modality_mask[..., 1:]], dim=-1
                    )
                gate_logits = gate_logits.masked_fill(~modality_mask.bool(), -1e4)
            gates = torch.softmax(gate_logits, dim=-1)
            fused = (stack * gates.unsqueeze(-1)).sum(dim=2)
            disagreement = ((stack - fused.unsqueeze(2)).square() * gates.unsqueeze(-1)).sum(dim=2)
            return self.fusion_norm(fused), gates, self.disagreement_stem(disagreement)

        def _apply_graph_filter(self, raw_logits, valid_mask):
            batch = raw_logits.shape[0]
            strength = torch.sigmoid(self.graph_strength_raw)
            evidence = torch.softmax(raw_logits, dim=-1)
            initial = raw_logits.new_full((batch, 1, config.num_steps), 1.0 / config.num_steps)
            # A first-order causal prior is parallel across time during training.
            # The previous *visual evidence* (never a future row) predicts the
            # allowed next state. This avoids hundreds of serialized CUDA kernel
            # launches from a Python recurrence on every sequence.
            previous = torch.cat([initial, evidence[:, :-1]], dim=1)
            prior = previous @ self.transition_matrix
            graph_prior = torch.where(
                self.seen_action_mask.view(1, 1, -1) > 0.5,
                prior,
                evidence,
            )
            posterior = (1.0 - strength) * evidence + strength * graph_prior
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if valid_mask is not None:
                posterior = torch.where(valid_mask.unsqueeze(-1), posterior, evidence)
            return torch.log(posterior.clamp_min(1e-7)), posterior

        def forward(
            self,
            motion,
            appearance,
            sensor,
            valid_mask=None,
            modality_mask=None,
            motion_aux=None,
        ):
            if valid_mask is None:
                valid_mask = torch.ones(motion.shape[:2], dtype=torch.bool, device=motion.device)
            fused, modality_gates, modality_disagreement = self._project_modalities(
                motion, appearance, sensor, modality_mask, motion_aux
            )
            temporal = self.position_encoding(fused)
            for block in self.temporal_blocks:
                temporal = block(temporal, valid_mask)
            prototype_residual = self.action_residual_prototypes * self.seen_action_mask.unsqueeze(-1)
            prototype_total = (
                prototype_residual
                + self.verb_prototypes[self.action_verb_indices]
                + self.object_prototypes[self.action_object_indices]
            )
            prototypes = prototype_total / (2.0 + self.seen_action_mask.unsqueeze(-1))
            prototypes = prototypes.unsqueeze(0).expand(temporal.shape[0], -1, -1)
            proto_context, proto_weights = self.prototype_attention(
                self.prototype_norm(temporal),
                prototypes,
                prototypes,
                key_padding_mask=(self.active_action_mask < 0.5)
                .unsqueeze(0)
                .expand(temporal.shape[0], -1),
                need_weights=True,
            )
            temporal = temporal + proto_context
            atomic_logits = self.step_classifier(temporal) * self.seen_action_mask
            verb_logits = self.verb_classifier(temporal)
            object_logits = self.object_classifier(temporal)
            composed_logits = (
                verb_logits[..., self.action_verb_indices]
                + object_logits[..., self.action_object_indices]
            ) / 2.0
            composition_strength = functional.softplus(self.composition_strength_raw)
            raw_step_logits = atomic_logits + composition_strength * composed_logits
            raw_step_logits = raw_step_logits.masked_fill(
                self.active_action_mask.view(1, 1, -1) < 0.5, -1e4
            )
            refinement_step_logits = []
            for stage in self.action_refinement_stages:
                raw_step_logits = stage(raw_step_logits, valid_mask)
                raw_step_logits = raw_step_logits.masked_fill(
                    self.active_action_mask.view(1, 1, -1) < 0.5, -1e4
                )
                refinement_step_logits.append(raw_step_logits)
            graph_step_logits, step_probabilities = self._apply_graph_filter(raw_step_logits, valid_mask)
            next_step_logits = self.next_step_head(temporal)
            # Assembly101 explicitly labels failed actions as ``attempt to ...``.
            # Convert that dense action supervision into a component-aware,
            # causal mistake-onset cue instead of asking a sparse 140-event head
            # to relearn the same semantics independently.
            mapped_actions = self.action_event_component_indices >= 0
            mistake_action_probability = step_probabilities.new_zeros(
                *step_probabilities.shape[:-1], config.num_completion_components
            )
            if bool(mapped_actions.any()):
                mistake_mass = (
                    step_probabilities[..., mapped_actions]
                    * self.mistake_action_mask[mapped_actions]
                )
                scatter_indices = self.action_event_component_indices[mapped_actions]
                scatter_indices = scatter_indices.view(1, 1, -1).expand_as(mistake_mass)
                mistake_action_probability.scatter_add_(
                    -1, scatter_indices, mistake_mass
                )
            previous_mistake_probability = functional.pad(
                mistake_action_probability[:, :-1], (0, 0, 1, 0), value=0.0
            )
            mistake_action_onset_score = (
                mistake_action_probability - previous_mistake_probability
            ).clamp_min(0.0)
            # Compare observed actions with both the empirical procedure graph
            # and the learned next-action prediction.  These signals answer a
            # different question from visual anomaly: is this action plausible
            # *at this point in the procedure*?  Everything is shifted or based
            # on current/past evidence, preserving causal inference.
            initial_action = step_probabilities.new_full(
                (step_probabilities.shape[0], 1, config.num_steps),
                1.0 / config.num_steps,
            )
            previous_action = torch.cat(
                [initial_action, step_probabilities[:, :-1]], dim=1
            )
            graph_expected = previous_action @ self.transition_matrix
            previous_next = torch.cat(
                [initial_action, torch.softmax(next_step_logits[:, :-1], dim=-1)], dim=1
            )
            graph_compatibility = (graph_expected * step_probabilities).sum(dim=-1)
            next_compatibility = (previous_next * step_probabilities).sum(dim=-1)
            procedure_violation_score = 0.5 * (
                (1.0 - graph_compatibility) + (1.0 - next_compatibility)
            )
            component_action_mass = step_probabilities.new_zeros(
                *step_probabilities.shape[:-1], config.num_completion_components
            )
            component_expected_mass = component_action_mass.clone()
            if bool(mapped_actions.any()):
                component_indices = self.action_event_component_indices[mapped_actions]
                scatter_indices = component_indices.view(1, 1, -1).expand(
                    *step_probabilities.shape[:-1], -1
                )
                component_action_mass.scatter_add_(
                    -1, scatter_indices, step_probabilities[..., mapped_actions]
                )
                component_expected_mass.scatter_add_(
                    -1,
                    scatter_indices,
                    graph_expected[..., mapped_actions].to(component_expected_mass.dtype),
                )
            event_temporal = temporal + modality_disagreement
            for block in self.event_temporal_blocks:
                event_temporal = block(event_temporal, valid_mask)
            state_logits = self.state_head(event_temporal).view(
                temporal.shape[0], temporal.shape[1], config.num_components, 3
            )
            action_context = step_probabilities @ prototype_total
            state_probabilities = torch.softmax(state_logits, dim=-1)
            event_features = self.event_norm(
                event_temporal
                + self.action_event_context(action_context)
                + self.state_event_context(state_probabilities.flatten(start_dim=-2))
            )
            if config.procedural_event_context:
                procedure_features = torch.cat(
                    [
                        component_action_mass,
                        component_expected_mass,
                        (component_action_mass - component_expected_mass).clamp_min(0.0),
                        mistake_action_probability,
                        graph_compatibility.unsqueeze(-1),
                        next_compatibility.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                event_features = self.event_norm(
                    event_features + self.procedure_event_context(procedure_features)
                )
            normal_reference = action_context.unsqueeze(-2) + self.component_normal_prototypes
            normality_logits = (
                functional.softplus(self.normality_scale_raw)
                * functional.cosine_similarity(event_features.unsqueeze(-2), normal_reference, dim=-1)
                + self.normality_bias
            )
            component_outcome_logits = self.component_outcome_head(event_features).view(
                temporal.shape[0],
                temporal.shape[1],
                config.num_completion_components,
                config.num_event_outcomes,
            )
            incorrect_index = 1
            anomaly_strength = functional.softplus(self.anomaly_strength_raw)
            incorrect_boost = anomaly_strength * (-normality_logits)
            component_outcome_logits = component_outcome_logits.clone()
            component_outcome_logits[..., incorrect_index] = (
                component_outcome_logits[..., incorrect_index] + incorrect_boost
            )
            component_state_probabilities = state_probabilities[..., self.event_state_indices, :]
            # State classes are [incorrect, pending, correct]; event outcomes are
            # [correct, incorrect, remove]. Pending is the closest observable
            # state evidence for a removal event.
            state_outcome_probabilities = component_state_probabilities[..., [2, 0, 1]]
            state_evidence_strength = functional.softplus(self.state_evidence_strength_raw)
            component_outcome_logits = component_outcome_logits + state_evidence_strength * torch.log(
                state_outcome_probabilities.clamp_min(1e-6)
            )
            incorrect_onset_logits = self.incorrect_onset_head(event_features)
            fused_incorrect_logits = None
            if config.learned_event_fusion:
                outcome_incorrect_logit = (
                    component_outcome_logits[..., 1]
                    - torch.logsumexp(
                        component_outcome_logits[..., [0, 2]], dim=-1
                    )
                )
                state_incorrect_logit = torch.logit(
                    state_outcome_probabilities[..., 1].clamp(1e-5, 1.0 - 1e-5)
                )
                fusion_inputs = torch.stack(
                    [
                        incorrect_onset_logits,
                        -normality_logits,
                        outcome_incorrect_logit,
                        state_incorrect_logit,
                        torch.logit(
                            mistake_action_onset_score.clamp(1e-5, 1.0 - 1e-5)
                        ),
                    ],
                    dim=-1,
                )
                fused_incorrect_logits = self.event_fusion_head(fusion_inputs).squeeze(-1)
            any_mistake_onset_logits = None
            incorrect_component_logits = None
            incorrect_component_probabilities = None
            factorized_incorrect_probabilities = None
            factorized_incorrect_logits = None
            if config.factorized_mistake_detection:
                any_mistake_onset_logits = self.any_mistake_onset_head(
                    event_features
                ).squeeze(-1)
                # Learned fusion, when enabled, is component evidence rather
                # than an independent onset decision. The softmax makes its
                # conditional meaning explicit.
                incorrect_component_logits = (
                    fused_incorrect_logits
                    if fused_incorrect_logits is not None
                    else incorrect_onset_logits
                )
                incorrect_component_probabilities = torch.softmax(
                    incorrect_component_logits, dim=-1
                )
                factorized_incorrect_probabilities = (
                    torch.sigmoid(any_mistake_onset_logits).unsqueeze(-1)
                    * incorrect_component_probabilities
                )
                factorized_incorrect_logits = torch.logit(
                    factorized_incorrect_probabilities.clamp(1e-6, 1.0 - 1e-6)
                )
            entropy = -(step_probabilities.clamp_min(1e-7).log() * step_probabilities).sum(dim=-1)
            # Entropy is bounded by log(K); this keeps uncertainty in [0, 1].
            normalized_entropy = entropy / math.log(max(config.num_steps, 2))
            energy = -torch.logsumexp(raw_step_logits, dim=-1)
            return {
                "features": temporal,
                "raw_step_logits": raw_step_logits,
                "atomic_step_logits": atomic_logits,
                "verb_logits": verb_logits,
                "object_logits": object_logits,
                "step_logits": graph_step_logits,
                "step_probabilities": step_probabilities,
                "refinement_step_logits": refinement_step_logits,
                "event_features": event_features,
                "completion_logits": self.completion_head(event_features),
                "component_outcome_logits": component_outcome_logits,
                "incorrect_onset_logits": incorrect_onset_logits,
                "fused_incorrect_logits": fused_incorrect_logits,
                "any_mistake_onset_logits": any_mistake_onset_logits,
                "incorrect_component_logits": incorrect_component_logits,
                "incorrect_component_probabilities": incorrect_component_probabilities,
                "factorized_incorrect_probabilities": factorized_incorrect_probabilities,
                "factorized_incorrect_logits": factorized_incorrect_logits,
                "mistake_action_probability": mistake_action_probability,
                "mistake_action_onset_score": mistake_action_onset_score,
                "normality_logits": normality_logits,
                "state_outcome_probabilities": state_outcome_probabilities,
                "event_state_indices": self.event_state_indices,
                "state_logits": state_logits,
                "boundary_logits": self.boundary_head(temporal).squeeze(-1),
                "progress_logits": self.progress_head(temporal).squeeze(-1),
                "next_step_logits": next_step_logits,
                "procedure_violation_score": procedure_violation_score,
                "uncertainty": normalized_entropy,
                "energy": energy,
                "modality_gates": modality_gates,
                "modality_disagreement": modality_disagreement,
                "prototype_weights": proto_weights,
                "seen_action_mask": self.seen_action_mask,
            }

    return StateGraphPSRLite()


def build_stategraph_loss(config: StateGraphLossConfig):
    torch, nn, functional = _load_torch()
    if config.any_mistake_weight < 0 or config.mistake_component_weight < 0:
        raise ValueError("Factorized mistake loss weights cannot be negative.")

    class StateGraphMultiTaskLoss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config

        @staticmethod
        def _focal_ce(logits, targets, gamma: float, class_weights=None):
            valid = targets != -100
            if not valid.any():
                return logits.sum() * 0.0
            selected_logits = logits[valid]
            selected_targets = targets[valid]
            ce = functional.cross_entropy(
                selected_logits, selected_targets, weight=class_weights, reduction="none"
            )
            probability = torch.softmax(selected_logits, dim=-1).gather(1, selected_targets[:, None]).squeeze(1)
            return (((1.0 - probability).clamp_min(0.0) ** gamma) * ce).mean()

        @staticmethod
        def _focal_bce(logits, targets, valid_mask, gamma: float, pos_weight=None):
            if not valid_mask.any():
                return logits.sum() * 0.0
            selected_logits = logits[valid_mask]
            selected_targets = targets[valid_mask]
            bce = functional.binary_cross_entropy_with_logits(
                selected_logits, selected_targets, pos_weight=pos_weight, reduction="none"
            )
            probabilities = torch.sigmoid(selected_logits)
            pt = torch.where(selected_targets > 0.5, probabilities, 1.0 - probabilities)
            return (((1.0 - pt).clamp_min(0.0) ** gamma) * bce).mean()

        @staticmethod
        def _asymmetric_bce(
            logits,
            targets,
            valid_mask,
            positive_gamma: float,
            negative_gamma: float,
            clip: float,
            pos_weight=None,
        ):
            if not valid_mask.any():
                return logits.sum() * 0.0
            expanded_mask = valid_mask
            while expanded_mask.ndim < logits.ndim:
                expanded_mask = expanded_mask.unsqueeze(-1)
            expanded_mask = expanded_mask.expand_as(logits)
            probabilities = torch.sigmoid(logits)
            positive_probability = probabilities.clamp(1e-8, 1.0 - 1e-8)
            negative_probability = (1.0 - probabilities + clip).clamp(max=1.0)
            positive_loss = targets * torch.log(positive_probability)
            negative_loss = (1.0 - targets) * torch.log(
                negative_probability.clamp_min(1e-8)
            )
            if pos_weight is not None:
                weight_shape = [1] * (logits.ndim - 1) + [pos_weight.shape[0]]
                positive_loss = positive_loss * pos_weight.view(*weight_shape)
            positive_focus = (1.0 - positive_probability) ** positive_gamma
            negative_focus = (1.0 - negative_probability) ** negative_gamma
            loss = -(positive_focus * positive_loss + negative_focus * negative_loss)
            return loss[expanded_mask].mean()

        @staticmethod
        def _hard_negative_mask(logits, targets, valid_mask, ratio: float):
            """Keep all positives and the hardest declared number of negatives."""

            expanded_valid = valid_mask.unsqueeze(-1).expand_as(targets)
            if ratio <= 0:
                return expanded_valid
            positive = expanded_valid & (targets > 0.5)
            negative = expanded_valid & ~positive
            positive_count = int(positive.sum().item())
            negative_count = int(negative.sum().item())
            if negative_count == 0:
                return positive
            keep_count = min(
                negative_count,
                max(1, int(math.ceil(max(positive_count, 1) * ratio))),
            )
            negative_indices = negative.nonzero(as_tuple=False)
            negative_scores = logits.detach()[negative]
            hardest = torch.topk(negative_scores, keep_count, sorted=False).indices
            selected = positive.clone()
            chosen = negative_indices[hardest]
            selected[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
            return selected

        def forward(
            self,
            outputs,
            targets,
            transition_matrix=None,
            step_class_weights=None,
            completion_pos_weights=None,
            component_outcome_class_weights=None,
            state_class_weights=None,
            incorrect_pos_weights=None,
        ):
            valid_mask = targets["valid_mask"].bool()
            step_targets = targets["step"]
            completion_targets, outcome_targets = _expand_causal_event_targets(
                targets["completion"],
                targets["component_outcome"],
                valid_mask,
                config.event_label_horizon,
            )
            losses = {}
            losses["step"] = self._focal_ce(
                outputs["step_logits"], step_targets, config.focal_gamma, step_class_weights
            )
            refinement_logits = outputs.get("refinement_step_logits", [])
            if refinement_logits:
                refinement_losses = [
                    self._focal_ce(logits, step_targets, config.focal_gamma, step_class_weights)
                    for logits in refinement_logits
                ]
                losses["refinement"] = torch.stack(refinement_losses).mean()
            else:
                losses["refinement"] = outputs["step_logits"].sum() * 0.0
            losses["completion"] = self._asymmetric_bce(
                outputs["completion_logits"],
                completion_targets.float(),
                valid_mask,
                positive_gamma=0.0,
                negative_gamma=config.asl_negative_gamma,
                clip=config.asl_clip,
                pos_weight=completion_pos_weights,
            )
            losses["component_outcome"] = self._focal_ce(
                outputs["component_outcome_logits"],
                outcome_targets,
                config.focal_gamma,
                component_outcome_class_weights,
            )
            incorrect_targets = (outcome_targets == 1).to(
                outputs["incorrect_onset_logits"].dtype
            )
            any_mistake_logits = outputs.get("any_mistake_onset_logits")
            component_selector_logits = outputs.get("incorrect_component_logits")
            if any_mistake_logits is not None and component_selector_logits is not None:
                component_positive = incorrect_targets > 0.5
                any_mistake_targets = component_positive.any(dim=-1).to(
                    any_mistake_logits.dtype
                )
                losses["any_mistake_onset"] = self._asymmetric_bce(
                    any_mistake_logits,
                    any_mistake_targets,
                    valid_mask,
                    positive_gamma=0.0,
                    negative_gamma=config.asl_negative_gamma,
                    clip=config.asl_clip,
                )
                selector_mask = valid_mask & component_positive.any(dim=-1)
                if selector_mask.any():
                    # Exact Assembly101 onsets have one component. Expanded
                    # horizons can overlap rarely, so use a normalized soft
                    # target instead of silently discarding a component.
                    selector_targets = component_positive.to(
                        component_selector_logits.dtype
                    )
                    selector_targets = selector_targets / selector_targets.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0)
                    losses["mistake_component"] = -(
                        selector_targets[selector_mask]
                        * functional.log_softmax(
                            component_selector_logits[selector_mask], dim=-1
                        )
                    ).sum(dim=-1).mean()
                else:
                    # Preserve a differentiable zero for batches without a rare
                    # event; the shared temporal detector still learns negatives.
                    losses["mistake_component"] = component_selector_logits.sum() * 0.0
                losses["incorrect_onset"] = (
                    config.any_mistake_weight * losses["any_mistake_onset"]
                    + config.mistake_component_weight * losses["mistake_component"]
                )
            else:
                learned_event_logits = outputs.get("fused_incorrect_logits")
                onset_logits = (
                    learned_event_logits
                    if learned_event_logits is not None
                    else outputs["incorrect_onset_logits"]
                )
                losses["incorrect_onset"] = self._asymmetric_bce(
                    onset_logits,
                    incorrect_targets,
                    self._hard_negative_mask(
                        onset_logits,
                        incorrect_targets,
                        valid_mask,
                        config.incorrect_hard_negative_ratio,
                    ),
                    positive_gamma=0.0,
                    negative_gamma=config.asl_negative_gamma,
                    clip=config.asl_clip,
                    pos_weight=incorrect_pos_weights,
                )
                losses["any_mistake_onset"] = onset_logits.sum() * 0.0
                losses["mistake_component"] = onset_logits.sum() * 0.0

            state_targets = targets["state"]
            state_mask = targets["state_mask"].bool() & valid_mask.unsqueeze(-1)
            if state_mask.any():
                selected_logits = outputs["state_logits"][state_mask]
                selected_targets = state_targets[state_mask]
                state_ce = functional.cross_entropy(
                    selected_logits, selected_targets, reduction="none"
                )
                if state_class_weights is not None:
                    component_ids = torch.arange(
                        state_targets.shape[-1], device=state_targets.device
                    ).view(1, 1, -1).expand_as(state_targets)[state_mask]
                    selected_weights = state_class_weights[component_ids, selected_targets]
                    state_ce = state_ce * selected_weights
                probabilities = torch.softmax(selected_logits, dim=-1).gather(
                    1, selected_targets[:, None]
                ).squeeze(1)
                losses["state"] = (
                    ((1.0 - probabilities).clamp_min(0.0) ** config.focal_gamma) * state_ce
                ).mean()
            else:
                losses["state"] = outputs["state_logits"].sum() * 0.0

            # IndustReal provides dense component states in addition to sparse
            # completion events. Use every observed installed state to train the
            # normality prototypes: state classes are [incorrect, pending,
            # correct]. Pending is deliberately excluded because it says
            # nothing about installation correctness.
            event_state_indices = outputs["event_state_indices"]
            mapped_states = state_targets[..., event_state_indices]
            mapped_state_mask = state_mask[..., event_state_indices]
            dense_normality_mask = mapped_state_mask & (
                (mapped_states == 0) | (mapped_states == 2)
            )
            dense_normality_binary = (mapped_states == 2).float()

            event_normality_mask = (outcome_targets == 0) | (outcome_targets == 1)
            event_normality_binary = (outcome_targets == 0).float()
            # Exact completion labels take precedence at the same component and
            # row, avoiding double-counting contradictory annotations.
            dense_normality_mask = dense_normality_mask & ~event_normality_mask
            normality_mask = dense_normality_mask | event_normality_mask
            normality_binary = torch.where(
                event_normality_mask,
                event_normality_binary,
                dense_normality_binary,
            )
            if normality_mask.any():
                normality_loss = functional.binary_cross_entropy_with_logits(
                    outputs["normality_logits"][normality_mask],
                    normality_binary[normality_mask],
                    reduction="none",
                )
                selected_normality = normality_binary[normality_mask]
                normality_weights = torch.where(
                    selected_normality > 0.5,
                    torch.ones_like(selected_normality),
                    torch.full_like(selected_normality, config.normality_error_weight),
                )
                losses["normality"] = (normality_loss * normality_weights).mean()
            else:
                losses["normality"] = outputs["normality_logits"].sum() * 0.0

            boundary_target = targets["boundary"].float()
            if valid_mask.any():
                positive = boundary_target[valid_mask].sum()
                negative = valid_mask.sum() - positive
                pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=20.0)
                losses["boundary"] = functional.binary_cross_entropy_with_logits(
                    outputs["boundary_logits"][valid_mask],
                    boundary_target[valid_mask],
                    pos_weight=pos_weight,
                )
            else:
                losses["boundary"] = outputs["boundary_logits"].sum() * 0.0

            losses["next_step"] = self._focal_ce(
                outputs["next_step_logits"], targets["next_step"], config.focal_gamma, step_class_weights
            )

            progress_target, progress_mask = _action_progress_targets(step_targets, valid_mask)
            if progress_mask.any():
                losses["progress"] = functional.smooth_l1_loss(
                    torch.sigmoid(outputs["progress_logits"])[progress_mask],
                    progress_target[progress_mask],
                )
            else:
                losses["progress"] = outputs["progress_logits"].sum() * 0.0

            log_probs = functional.log_softmax(outputs["raw_step_logits"], dim=-1)
            pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            if pair_mask.any():
                delta = log_probs[:, 1:] - log_probs[:, :-1].detach()
                losses["smoothing"] = (delta.square().mean(dim=-1)[pair_mask]).mean()
            else:
                losses["smoothing"] = log_probs.sum() * 0.0

            matrix = transition_matrix
            if matrix is None:
                matrix = outputs["step_probabilities"].new_ones(
                    outputs["step_probabilities"].shape[-1], outputs["step_probabilities"].shape[-1]
                )
            allowed = (matrix > 0).to(outputs["step_probabilities"].dtype)
            unseen = outputs.get("seen_action_mask")
            if unseen is not None:
                unseen = unseen < 0.5
                allowed[:, unseen] = 1.0
                allowed[unseen, :] = 1.0
            previous = outputs["step_probabilities"][:, :-1]
            current = outputs["step_probabilities"][:, 1:]
            allowed_mass = torch.einsum("bti,ij,btj->bt", previous, allowed, current)
            if pair_mask.any():
                losses["graph"] = (-torch.log(allowed_mass.clamp_min(1e-6))[pair_mask]).mean()
            else:
                losses["graph"] = allowed_mass.sum() * 0.0

            incorrect_index = 1
            completion_probability = torch.sigmoid(outputs["completion_logits"])
            outcome_incorrect = torch.softmax(outputs["component_outcome_logits"], dim=-1)[
                ..., incorrect_index
            ]
            event_incorrect = 1.0 - torch.prod(
                1.0 - completion_probability * outcome_incorrect, dim=-1
            )
            component_incorrect = torch.softmax(outputs["state_logits"], dim=-1)[..., 0]
            any_component_incorrect = 1.0 - torch.prod(1.0 - component_incorrect, dim=-1)
            consistency_mask = (
                valid_mask
                & state_mask.any(dim=-1)
                & completion_targets.bool().any(dim=-1)
                & (outcome_targets == incorrect_index).any(dim=-1)
            )
            if consistency_mask.any():
                losses["consistency"] = functional.mse_loss(
                    event_incorrect[consistency_mask], any_component_incorrect[consistency_mask]
                )
            else:
                losses["consistency"] = event_incorrect.sum() * 0.0

            total = (
                config.step_weight * losses["step"]
                + config.completion_weight * losses["completion"]
                + config.component_outcome_weight * losses["component_outcome"]
                + config.incorrect_onset_weight * losses["incorrect_onset"]
                + config.state_weight * losses["state"]
                + config.boundary_weight * losses["boundary"]
                + config.next_step_weight * losses["next_step"]
                + config.smoothing_weight * losses["smoothing"]
                + config.graph_weight * losses["graph"]
                + config.consistency_weight * losses["consistency"]
                + config.progress_weight * losses["progress"]
                + config.normality_weight * losses["normality"]
                + config.refinement_weight * losses["refinement"]
            )
            losses["total"] = total
            return losses

    return StateGraphMultiTaskLoss()


def _expand_causal_event_targets(completion, outcomes, valid_mask, horizon: int):
    """Repeat rare outcome labels into a short post-event training horizon.

    Only future rows are filled, so this does not leak future evidence into the
    causal online model. Completion targets remain exact point events to retain
    immediate alert timing; evaluation also uses the original point events.
    """

    torch, _, _ = _load_torch()
    if horizon <= 0:
        return completion, outcomes
    expanded_outcomes = outcomes.clone()
    for offset in range(1, horizon + 1):
        source_completion = completion[:, :-offset].bool()
        source_outcomes = outcomes[:, :-offset]
        destination_valid = valid_mask[:, offset:].unsqueeze(-1)
        empty_destination = expanded_outcomes[:, offset:] == -100
        propagate = (
            source_completion
            & (source_outcomes != -100)
            & destination_valid
            & empty_destination
        )
        expanded_outcomes[:, offset:] = torch.where(
            propagate,
            source_outcomes,
            expanded_outcomes[:, offset:],
        )
    return completion, expanded_outcomes


def _action_progress_targets(step_targets, valid_mask):
    """Create causal-compatible 0..1 progress targets for each action segment."""

    torch, _, _ = _load_torch()
    progress = torch.zeros_like(step_targets, dtype=torch.float32)
    mask = valid_mask.bool() & (step_targets >= 0)
    for batch_index in range(step_targets.shape[0]):
        length = int(mask[batch_index].sum().item())
        if length == 0:
            continue
        labels = step_targets[batch_index, :length]
        start = 0
        while start < length:
            end = start + 1
            while end < length and labels[end] == labels[start]:
                end += 1
            segment_length = end - start
            if segment_length == 1:
                progress[batch_index, start] = 1.0
            else:
                progress[batch_index, start:end] = torch.linspace(
                    0.0, 1.0, segment_length, device=step_targets.device
                )
            start = end
    return progress, mask
