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
    num_action_verbs: int = 0
    num_action_objects: int = 0
    action_verb_indices: tuple[int, ...] = ()
    action_object_indices: tuple[int, ...] = ()
    seen_action_mask: tuple[bool, ...] = ()
    num_completion_components: int = 1
    num_event_outcomes: int = 3
    num_components: int = 11
    hidden_dim: int = 192
    num_temporal_blocks: int = 8
    attention_every: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    graph_strength_init: float = 1.5
    composition_strength_init: float = 1.0
    max_dilation: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateGraphLossConfig:
    step_weight: float = 1.0
    completion_weight: float = 0.45
    component_outcome_weight: float = 0.7
    state_weight: float = 0.8
    boundary_weight: float = 0.25
    next_step_weight: float = 0.3
    smoothing_weight: float = 0.12
    graph_weight: float = 0.15
    consistency_weight: float = 0.15
    focal_gamma: float = 1.5


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

    if config.action_verb_indices:
        if len(config.action_verb_indices) != config.num_steps:
            raise ValueError("action_verb_indices must contain one entry per action.")
        if len(config.action_object_indices) != config.num_steps:
            raise ValueError("action_object_indices must contain one entry per action.")
        if len(config.seen_action_mask) != config.num_steps:
            raise ValueError("seen_action_mask must contain one entry per action.")

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

    class StateGraphPSRLite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.motion_stem = ProjectionStem(config.motion_dim)
            self.appearance_stem = ProjectionStem(config.appearance_dim)
            self.sensor_stem = ProjectionStem(config.sensor_dim)
            self.modality_gate = nn.Sequential(
                nn.LayerNorm(config.hidden_dim * 3),
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 3),
            )
            self.fusion_norm = nn.LayerNorm(config.hidden_dim)
            blocks = []
            for index in range(config.num_temporal_blocks):
                dilation = min(2 ** (index % 5), config.max_dilation)
                with_attention = config.attention_every > 0 and (index + 1) % config.attention_every == 0
                blocks.append(TemporalBlock(dilation, with_attention))
            self.temporal_blocks = nn.ModuleList(blocks)
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
            self.register_buffer("action_verb_indices", verb_indices)
            self.register_buffer("action_object_indices", object_indices)
            self.register_buffer("seen_action_mask", seen_mask)
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
            self.completion_head = nn.Linear(config.hidden_dim, config.num_completion_components)
            self.component_outcome_head = nn.Linear(
                config.hidden_dim, config.num_completion_components * config.num_event_outcomes
            )
            self.state_head = nn.Linear(config.hidden_dim, config.num_components * 3)
            self.boundary_head = nn.Linear(config.hidden_dim, 1)
            self.next_step_head = nn.Linear(config.hidden_dim, config.num_steps)
            self.graph_strength_raw = nn.Parameter(torch.tensor(float(config.graph_strength_init)))
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

        def _project_modalities(self, motion, appearance, sensor, modality_mask):
            batch, length = motion.shape[:2]
            zeros = motion.new_zeros(batch, length, config.hidden_dim)
            projected = [
                self.motion_stem(motion) if config.motion_dim > 0 else zeros,
                self.appearance_stem(appearance) if config.appearance_dim > 0 else zeros,
                self.sensor_stem(sensor) if config.sensor_dim > 0 else zeros,
            ]
            stack = torch.stack(projected, dim=2)
            gate_input = torch.cat(projected, dim=-1)
            gate_logits = self.modality_gate(gate_input)
            if modality_mask is not None:
                gate_logits = gate_logits.masked_fill(~modality_mask.bool(), -1e4)
            gates = torch.softmax(gate_logits, dim=-1)
            fused = (stack * gates.unsqueeze(-1)).sum(dim=2)
            return self.fusion_norm(fused), gates

        def _apply_graph_filter(self, raw_logits, valid_mask):
            batch, length, _ = raw_logits.shape
            posterior_rows = []
            adjusted_rows = []
            previous = raw_logits.new_full((batch, config.num_steps), 1.0 / config.num_steps)
            strength = functional.softplus(self.graph_strength_raw)
            for index in range(length):
                prior = previous @ self.transition_matrix
                graph_bias = strength * torch.log(prior.clamp_min(1e-6))
                graph_bias = graph_bias * self.seen_action_mask.unsqueeze(0)
                adjusted = raw_logits[:, index] + graph_bias
                posterior = torch.softmax(adjusted, dim=-1)
                if valid_mask is not None:
                    active = valid_mask[:, index].unsqueeze(-1)
                    posterior = torch.where(active, posterior, previous)
                adjusted_rows.append(adjusted)
                posterior_rows.append(posterior)
                previous = posterior
            return torch.stack(adjusted_rows, dim=1), torch.stack(posterior_rows, dim=1)

        def forward(
            self,
            motion,
            appearance,
            sensor,
            valid_mask=None,
            modality_mask=None,
        ):
            if valid_mask is None:
                valid_mask = torch.ones(motion.shape[:2], dtype=torch.bool, device=motion.device)
            fused, modality_gates = self._project_modalities(motion, appearance, sensor, modality_mask)
            temporal = fused
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
                self.prototype_norm(temporal), prototypes, prototypes, need_weights=True
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
            graph_step_logits, step_probabilities = self._apply_graph_filter(raw_step_logits, valid_mask)
            state_logits = self.state_head(temporal).view(
                temporal.shape[0], temporal.shape[1], config.num_components, 3
            )
            component_outcome_logits = self.component_outcome_head(temporal).view(
                temporal.shape[0],
                temporal.shape[1],
                config.num_completion_components,
                config.num_event_outcomes,
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
                "completion_logits": self.completion_head(temporal),
                "component_outcome_logits": component_outcome_logits,
                "state_logits": state_logits,
                "boundary_logits": self.boundary_head(temporal).squeeze(-1),
                "next_step_logits": self.next_step_head(temporal),
                "uncertainty": normalized_entropy,
                "energy": energy,
                "modality_gates": modality_gates,
                "prototype_weights": proto_weights,
                "seen_action_mask": self.seen_action_mask,
            }

    return StateGraphPSRLite()


def build_stategraph_loss(config: StateGraphLossConfig):
    torch, nn, functional = _load_torch()

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

        def forward(
            self,
            outputs,
            targets,
            transition_matrix=None,
            step_class_weights=None,
            completion_pos_weights=None,
            component_outcome_class_weights=None,
        ):
            valid_mask = targets["valid_mask"].bool()
            step_targets = targets["step"]
            losses = {}
            losses["step"] = self._focal_ce(
                outputs["step_logits"], step_targets, config.focal_gamma, step_class_weights
            )
            losses["completion"] = self._focal_bce(
                outputs["completion_logits"],
                targets["completion"].float(),
                valid_mask,
                config.focal_gamma,
                completion_pos_weights,
            )
            losses["component_outcome"] = self._focal_ce(
                outputs["component_outcome_logits"],
                targets["component_outcome"],
                config.focal_gamma,
                component_outcome_class_weights,
            )

            state_targets = targets["state"]
            state_mask = targets["state_mask"].bool() & valid_mask.unsqueeze(-1)
            if state_mask.any():
                losses["state"] = functional.cross_entropy(
                    outputs["state_logits"][state_mask], state_targets[state_mask]
                )
            else:
                losses["state"] = outputs["state_logits"].sum() * 0.0

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
                & targets["completion"].bool().any(dim=-1)
                & (targets["component_outcome"] == incorrect_index).any(dim=-1)
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
                + config.state_weight * losses["state"]
                + config.boundary_weight * losses["boundary"]
                + config.next_step_weight * losses["next_step"]
                + config.smoothing_weight * losses["smoothing"]
                + config.graph_weight * losses["graph"]
                + config.consistency_weight * losses["consistency"]
            )
            losses["total"] = total
            return losses

    return StateGraphMultiTaskLoss()
