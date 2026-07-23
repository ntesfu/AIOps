from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StateEffectObserverConfig:
    """Compact gaze-free component state/effect observer."""

    global_dim: int
    roi_dim: int
    num_components: int
    num_steps: int = 0
    roi_count: int = 4
    hand_dim: int = 0
    pose_dim: int = 0
    depth_dim: int = 0
    hidden_dim: int = 256
    temporal_blocks: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    effect_lag: int = 4

    def validate(self) -> None:
        if self.global_dim <= 0 or self.roi_dim <= 0:
            raise ValueError("global_dim and roi_dim must be positive.")
        if self.num_components <= 0 or self.roi_count <= 0:
            raise ValueError("num_components and roi_count must be positive.")
        if self.num_steps < 0:
            raise ValueError("num_steps cannot be negative.")
        if min(self.hand_dim, self.pose_dim, self.depth_dim) < 0:
            raise ValueError("Optional input dimensions cannot be negative.")
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads.")
        if self.temporal_blocks <= 0 or self.effect_lag <= 0:
            raise ValueError("temporal_blocks and effect_lag must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateEffectLossConfig:
    """Supervision weights for persistent state and typed effects."""

    state_weight: float = 1.0
    effect_weight: float = 1.0
    step_weight: float = 0.5
    transition_consistency_weight: float = 0.25
    focal_gamma: float = 1.5
    effect_no_change_ratio: float = 4.0
    effect_min_no_change: int = 64

    def validate(self) -> None:
        if min(
            self.state_weight,
            self.effect_weight,
            self.step_weight,
            self.transition_consistency_weight,
            self.focal_gamma,
            self.effect_no_change_ratio,
            self.effect_min_no_change,
        ) < 0:
            raise ValueError("State/effect loss weights and focal_gamma cannot be negative.")


def state_effect_targets(targets, event_state_indices):
    """Convert legacy IndustReal labels into the StateVerify typed contract.

    Raw persistent-state order is ``incorrect, pending, correct``. StateVerify
    uses ``not_completed, installed_correct, installed_incorrect``. Sparse
    completion outcomes are ``correct, incorrect, remove`` and map to effect
    classes 1..3; class 0 is a dense observed no-change transition.
    """

    state = targets["state"]
    state_mask = targets["state_mask"].bool()
    valid_mask = targets["valid_mask"].bool()
    if state.shape != state_mask.shape:
        raise ValueError("state and state_mask must have identical shapes.")
    if state.shape[:2] != valid_mask.shape:
        raise ValueError("valid_mask must match the batch/time state axes.")

    typed_state = state.new_full(state.shape, -100)
    typed_state[state == 1] = 0
    typed_state[state == 2] = 1
    typed_state[state == 0] = 2
    typed_state_mask = state_mask & valid_mask.unsqueeze(-1)

    effect = state.new_full(state.shape, -100)
    if state.shape[1] > 1:
        previous = typed_state[:, :-1]
        current = typed_state[:, 1:]
        pair_mask = (
            typed_state_mask[:, :-1]
            & typed_state_mask[:, 1:]
            & valid_mask[:, :-1].unsqueeze(-1)
            & valid_mask[:, 1:].unsqueeze(-1)
        )
        transition = current.new_zeros(current.shape)
        transition[(current == 1) & (current != previous)] = 1
        transition[(current == 2) & (current != previous)] = 2
        transition[(current == 0) & (current != previous)] = 3
        effect[:, 1:][pair_mask] = transition[pair_mask]

    outcomes = targets["component_outcome"]
    if outcomes.shape[:2] != state.shape[:2]:
        raise ValueError("component_outcome must match the batch/time state axes.")
    if len(event_state_indices) != outcomes.shape[-1]:
        raise ValueError(
            "event_state_indices must contain one state component for each "
            "component_outcome column."
        )
    for event_component, state_component in enumerate(event_state_indices):
        if not 0 <= int(state_component) < state.shape[-1]:
            raise ValueError("event_state_indices contains an invalid state component.")
        event = outcomes[..., event_component]
        event_mask = (event >= 0) & valid_mask
        effect[..., int(state_component)][event_mask] = event[event_mask] + 1

    return {
        "state": typed_state,
        "state_mask": typed_state_mask,
        "effect": effect,
        "effect_mask": (effect >= 0) & valid_mask.unsqueeze(-1),
    }

def build_state_effect_loss(config: StateEffectLossConfig):
    """Build the typed StateVerify loss with probabilistic transition coupling."""

    config.validate()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("Install the ml dependencies to build the observer loss.") from exc

    class StateEffectLoss(nn.Module):
        @staticmethod
        def _effect_training_mask(logits, labels, mask):
            positive = mask & (labels != 0)
            negative = mask & (labels == 0)
            negative_count = int(negative.sum())
            if negative_count == 0:
                return positive
            keep = max(
                config.effect_min_no_change,
                int(config.effect_no_change_ratio * int(positive.sum())),
            )
            keep = min(keep, negative_count)
            selected = positive.clone()
            if keep:
                anomaly_score = 1.0 - torch.softmax(
                    logits.detach(), dim=-1
                )[..., 0]
                negative_positions = negative.nonzero(as_tuple=False)
                chosen = torch.topk(
                    anomaly_score[negative], keep, sorted=False
                ).indices
                chosen_positions = negative_positions[chosen]
                selected[tuple(chosen_positions.transpose(0, 1))] = True
            return selected

        @staticmethod
        def _focal_cross_entropy(logits, labels, mask, class_weights=None):
            if not mask.any():
                return logits.sum() * 0.0
            selected_logits = logits[mask]
            selected_labels = labels[mask]
            cross_entropy = functional.cross_entropy(
                selected_logits,
                selected_labels,
                reduction="none",
                weight=class_weights,
            )
            probability = torch.softmax(selected_logits, dim=-1).gather(
                1, selected_labels.unsqueeze(-1)
            ).squeeze(-1)
            return (
                (1.0 - probability).clamp_min(0.0).pow(config.focal_gamma)
                * cross_entropy
            ).mean()

        def forward(
            self,
            outputs,
            targets,
            event_state_indices,
            state_class_weights=None,
            effect_class_weights=None,
        ):
            typed = state_effect_targets(targets, event_state_indices)
            effect_training_mask = self._effect_training_mask(
                outputs["effect_logits"], typed["effect"], typed["effect_mask"]
            )
            losses = {
                "state": self._focal_cross_entropy(
                    outputs["state_logits"],
                    typed["state"],
                    typed["state_mask"],
                    state_class_weights,
                ),
                "effect": self._focal_cross_entropy(
                    outputs["effect_logits"],
                    typed["effect"],
                    effect_training_mask,
                    effect_class_weights,
                ),
            }
            if outputs.get("step_logits") is not None:
                step_mask = targets["valid_mask"].bool() & (targets["step"] >= 0)
                losses["step"] = self._focal_cross_entropy(
                    outputs["step_logits"],
                    targets["step"],
                    step_mask,
                )

            state_probability = torch.softmax(outputs["state_logits"], dim=-1)
            effect_probability = torch.softmax(outputs["effect_logits"], dim=-1)
            pair_mask = (
                targets["valid_mask"][:, 1:].bool()
                & targets["valid_mask"][:, :-1].bool()
            ).unsqueeze(-1).unsqueeze(-1)
            if pair_mask.any():
                previous = state_probability[:, :-1]
                current = state_probability[:, 1:]
                no_change = (previous * current).sum(dim=-1)
                complete_correct = current[..., 1] * (1.0 - previous[..., 1])
                complete_incorrect = current[..., 2] * (1.0 - previous[..., 2])
                remove = current[..., 0] * (1.0 - previous[..., 0])
                implied_effect = torch.stack(
                    [no_change, complete_correct, complete_incorrect, remove], dim=-1
                )
                implied_effect = implied_effect / implied_effect.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                consistency_mask = pair_mask.expand_as(implied_effect)
                losses["transition_consistency"] = functional.smooth_l1_loss(
                    effect_probability[:, 1:][consistency_mask],
                    implied_effect[consistency_mask],
                )
            else:
                losses["transition_consistency"] = effect_probability.sum() * 0.0
            losses["total"] = (
                config.state_weight * losses["state"]
                + config.effect_weight * losses["effect"]
                + config.transition_consistency_weight
                * losses["transition_consistency"]
            )
            if "step" in losses:
                losses["total"] = losses["total"] + config.step_weight * losses["step"]
            return losses

    return StateEffectLoss()


def build_state_effect_observer(config: StateEffectObserverConfig):
    """Build a causal observer without importing PyTorch at package import time."""

    config.validate()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("Install the ml dependencies to build the observer.") from exc

    class Projection(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.input_dim = input_dim
            self.network = (
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.GELU(),
                )
                if input_dim > 0
                else None
            )

        def forward(self, values, reference, mask=None):
            if self.network is None or values is None:
                return reference.new_zeros(reference.shape)
            projected = self.network(values)
            if mask is not None:
                projected = projected * mask.to(projected.dtype).unsqueeze(-1)
            return projected

    class CausalComponentBlock(nn.Module):
        def __init__(self, dilation: int):
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
            self.pointwise = nn.Sequential(
                nn.GELU(),
                nn.Conv1d(config.hidden_dim, 2 * config.hidden_dim, kernel_size=1),
                nn.GLU(dim=1),
                nn.Dropout(config.dropout),
            )

        def forward(self, values, valid_mask):
            batch, time, components, hidden = values.shape
            flattened = (
                self.norm(values)
                .permute(0, 2, 3, 1)
                .reshape(batch * components, hidden, time)
            )
            flattened = functional.pad(flattened, (2 * self.dilation, 0))
            update = self.pointwise(self.depthwise(flattened))
            update = (
                update.reshape(batch, components, hidden, time)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if valid_mask is not None:
                update = update * valid_mask[:, :, None, None].to(update.dtype)
            return values + update

    class StateEffectObserver(nn.Module):
        state_order = (
            "not_completed",
            "installed_correct",
            "installed_incorrect",
        )
        effect_order = (
            "no_change",
            "complete_correct",
            "complete_incorrect",
            "remove",
        )
        roi_order = (
            "left_hand",
            "right_hand",
            "active_object",
            "interaction_context",
        )

        def __init__(self):
            super().__init__()
            self.config = config
            self.global_stem = Projection(config.global_dim)
            self.roi_stem = nn.Sequential(
                nn.LayerNorm(config.roi_dim),
                nn.Linear(config.roi_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.hand_stem = Projection(config.hand_dim)
            self.pose_stem = Projection(config.pose_dim)
            self.depth_stem = Projection(config.depth_dim)
            self.roi_type_embedding = nn.Parameter(
                torch.empty(config.roi_count, config.hidden_dim)
            )
            self.component_queries = nn.Parameter(
                torch.empty(config.num_components, config.hidden_dim)
            )
            nn.init.trunc_normal_(self.roi_type_embedding, std=0.02)
            nn.init.trunc_normal_(self.component_queries, std=0.02)
            self.roi_attention = nn.MultiheadAttention(
                config.hidden_dim,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.temporal_blocks = nn.ModuleList(
                [
                    CausalComponentBlock(2 ** (index % 4))
                    for index in range(config.temporal_blocks)
                ]
            )
            self.state_head = nn.Linear(config.hidden_dim, 3)
            self.step_head = (
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.num_steps),
                )
                if config.num_steps > 0
                else None
            )
            self.step_embedding = (
                nn.Parameter(torch.empty(config.num_steps, config.hidden_dim))
                if config.num_steps > 0
                else None
            )
            self.step_context_gate = (
                nn.Parameter(torch.tensor(0.0))
                if config.num_steps > 0
                else None
            )
            if self.step_embedding is not None:
                nn.init.trunc_normal_(self.step_embedding, std=0.02)
            self.effect_head = nn.Sequential(
                nn.LayerNorm(2 * config.hidden_dim),
                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 4),
            )

        def forward(
            self,
            global_features,
            roi_features,
            *,
            roi_mask=None,
            hands=None,
            hand_mask=None,
            pose=None,
            pose_mask=None,
            depth=None,
            depth_mask=None,
            valid_mask=None,
            expected_effect=None,
        ):
            if global_features.ndim != 3:
                raise ValueError("global_features must have shape [batch, time, dim].")
            batch, time = global_features.shape[:2]
            expected_roi_shape = (batch, time, config.roi_count, config.roi_dim)
            if tuple(roi_features.shape) != expected_roi_shape:
                raise ValueError(
                    f"roi_features must have shape {expected_roi_shape}; "
                    f"received {tuple(roi_features.shape)}."
                )
            if valid_mask is None:
                valid_mask = torch.ones(
                    batch, time, dtype=torch.bool, device=global_features.device
                )
            if roi_mask is None:
                roi_mask = torch.ones(
                    batch,
                    time,
                    config.roi_count,
                    dtype=torch.bool,
                    device=global_features.device,
                )
            if tuple(roi_mask.shape) != (batch, time, config.roi_count):
                raise ValueError("roi_mask must have shape [batch, time, roi_count].")

            global_context = self.global_stem(
                global_features, global_features.new_zeros(batch, time, config.hidden_dim)
            )
            masked_roi_features = roi_features * roi_mask.unsqueeze(-1).to(
                dtype=roi_features.dtype
            )
            roi_tokens = self.roi_stem(masked_roi_features)
            roi_tokens = (
                roi_tokens
                + self.roi_type_embedding.view(
                    1, 1, config.roi_count, config.hidden_dim
                )
            )
            flattened_mask = ~roi_mask.reshape(batch * time, config.roi_count).bool()
            all_missing = flattened_mask.all(dim=-1)
            safe_mask = flattened_mask.clone()
            safe_mask[all_missing, 0] = False
            flattened_tokens = roi_tokens.reshape(
                batch * time, config.roi_count, config.hidden_dim
            )
            queries = self.component_queries.view(
                1, config.num_components, config.hidden_dim
            ).expand(batch * time, -1, -1)
            attended, attention = self.roi_attention(
                queries,
                flattened_tokens,
                flattened_tokens,
                key_padding_mask=safe_mask,
                need_weights=True,
            )
            component_roi = attended.reshape(
                batch, time, config.num_components, config.hidden_dim
            )
            attention = attention.reshape(
                batch, time, config.num_components, config.roi_count
            )

            auxiliary = self.hand_stem(hands, global_context, hand_mask)
            auxiliary = auxiliary + self.pose_stem(pose, global_context, pose_mask)
            auxiliary = auxiliary + self.depth_stem(depth, global_context, depth_mask)
            component_features = self.fusion(
                global_context.unsqueeze(-2)
                + component_roi
                + auxiliary.unsqueeze(-2)
                + self.component_queries.view(
                    1, 1, config.num_components, config.hidden_dim
                )
            )
            component_features = component_features * valid_mask[
                :, :, None, None
            ].to(component_features.dtype)
            for block in self.temporal_blocks:
                component_features = block(component_features, valid_mask)

            step_logits = (
                self.step_head(component_features.mean(dim=-2))
                if self.step_head is not None
                else None
            )
            if step_logits is not None:
                step_probabilities = torch.softmax(step_logits, dim=-1)
                step_context = step_probabilities @ self.step_embedding
                component_features = component_features + torch.tanh(
                    self.step_context_gate
                ) * step_context.unsqueeze(-2)
            else:
                step_probabilities = None
            state_logits = self.state_head(component_features)
            indices = torch.arange(time, device=component_features.device)
            reference = (indices - config.effect_lag).clamp_min(0)
            previous_features = component_features[:, reference]
            delta_features = component_features - previous_features
            effect_logits = self.effect_head(
                torch.cat([component_features, delta_features], dim=-1)
            )
            observed_effect = torch.softmax(effect_logits, dim=-1)
            effect_residual = None
            if expected_effect is not None:
                if tuple(expected_effect.shape) != (
                    batch,
                    time,
                    config.num_components,
                    4,
                ):
                    raise ValueError(
                        "expected_effect must have shape "
                        "[batch, time, component, 4]."
                    )
                expected = expected_effect.clamp_min(0)
                expected = expected / expected.sum(dim=-1, keepdim=True).clamp_min(
                    1e-6
                )
                effect_residual = 1.0 - torch.sqrt(
                    observed_effect.clamp_min(1e-8)
                    * expected.clamp_min(1e-8)
                ).sum(dim=-1)
                effect_residual = effect_residual.clamp(0.0, 1.0)
            return {
                "component_features": component_features,
                "step_logits": step_logits,
                "step_probabilities": step_probabilities,
                "state_logits": state_logits,
                "state_probabilities": torch.softmax(state_logits, dim=-1),
                "effect_logits": effect_logits,
                "effect_probabilities": observed_effect,
                "effect_residual": effect_residual,
                "component_roi_attention": attention,
            }

    return StateEffectObserver()
