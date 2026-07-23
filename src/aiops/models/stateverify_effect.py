from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StateEffectObserverConfig:
    """Compact gaze-free component state/effect observer."""

    global_dim: int
    roi_dim: int
    num_components: int
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
                "state_logits": state_logits,
                "state_probabilities": torch.softmax(state_logits, dim=-1),
                "effect_logits": effect_logits,
                "effect_probabilities": observed_effect,
                "effect_residual": effect_residual,
                "component_roi_attention": attention,
            }

    return StateEffectObserver()
