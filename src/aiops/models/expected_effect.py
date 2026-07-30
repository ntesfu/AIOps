"""Expected-effect predictor: verify assembly by predicted outcome.

The core insight behind the plan is that correctness becomes visible in the
*resulting state*, and that the *normal* effect of an action is learnable from
abundant correct data even though real errors are scarce.

This module learns, from **pre-action (past) context only**, the effect that a
component *should* undergo — a distribution over
``no_change / complete_correct / complete_incorrect / remove``.  Trained on
correct executions, it predicts ``complete_correct`` where a correct install is
underway.  The effect **residual** is then the divergence between the observer's
*observed* effect (which sees the after-state) and this expected effect: when an
execution goes wrong, the observed distribution moves toward
``complete_incorrect`` while the expected distribution stays on
``complete_correct``, so the residual rises.

This replaces the conservative proxy residual (positive onset of the incorrect
state probability) documented in ``docs/stateverify_psr_core.md`` with a learned
reference model, and it is deliberately kept causal: each output reads only the
current step context and a *lagged* component feature, so future frames can never
change a past prediction.

PyTorch is imported lazily (inside the build/loss functions) so importing this
module — and its config dataclasses — never requires the ML stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


EFFECT_ORDER = ("no_change", "complete_correct", "complete_incorrect", "remove")


@dataclass(frozen=True)
class ExpectedEffectPredictorConfig:
    """Compact, causal, per-(time, component) effect predictor."""

    hidden_dim: int
    num_components: int
    num_steps: int = 0
    effect_lag: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_components <= 0:
            raise ValueError("num_components must be positive.")
        if self.num_steps < 0:
            raise ValueError("num_steps cannot be negative.")
        if self.effect_lag <= 0:
            raise ValueError("effect_lag must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_expected_effect_predictor(config: ExpectedEffectPredictorConfig):
    """Build the causal expected-effect predictor without importing torch eagerly."""

    config.validate()
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError(
            "Install the ml dependencies to build the expected-effect predictor."
        ) from exc

    class ExpectedEffectPredictor(nn.Module):
        effect_order = EFFECT_ORDER

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.component_embedding = nn.Parameter(
                torch.empty(config.num_components, config.hidden_dim)
            )
            nn.init.trunc_normal_(self.component_embedding, std=0.02)
            self.step_embedding = (
                nn.Parameter(torch.empty(config.num_steps, config.hidden_dim))
                if config.num_steps > 0
                else None
            )
            if self.step_embedding is not None:
                nn.init.trunc_normal_(self.step_embedding, std=0.02)
            input_dim = 2 * config.hidden_dim + (
                config.hidden_dim if config.num_steps > 0 else 0
            )
            self.head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, len(EFFECT_ORDER)),
            )

        def forward(
            self,
            component_features,
            step_probabilities=None,
            valid_mask=None,
        ):
            if component_features.ndim != 4:
                raise ValueError(
                    "component_features must have shape [batch, time, component, hidden]."
                )
            batch, time, components, hidden = component_features.shape
            if components != config.num_components or hidden != config.hidden_dim:
                raise ValueError(
                    "component_features must match num_components and hidden_dim."
                )
            # Causal pre-action context: read a strictly non-future lagged frame.
            indices = torch.arange(time, device=component_features.device)
            reference = (indices - config.effect_lag).clamp_min(0)
            lagged = component_features[:, reference]
            component = self.component_embedding.view(
                1, 1, components, hidden
            ).expand(batch, time, -1, -1)
            parts = [lagged, component]
            if self.step_embedding is not None and step_probabilities is not None:
                step_context = step_probabilities @ self.step_embedding
                parts.append(step_context.unsqueeze(-2).expand(-1, -1, components, -1))
            features = torch.cat(parts, dim=-1)
            logits = self.head(features)
            if valid_mask is not None:
                logits = logits * valid_mask[:, :, None, None].to(logits.dtype)
            return {
                "expected_effect_logits": logits,
                "expected_effect": torch.softmax(logits, dim=-1),
            }

    return ExpectedEffectPredictor()


def expected_effect_loss(
    expected_logits,
    typed_effect,
    effect_mask,
    *,
    exclude_labels: tuple[int, ...] = (2,),
):
    """Cross-entropy that teaches the predictor the *normal* effect only.

    ``typed_effect`` / ``effect_mask`` come from
    :func:`aiops.models.stateverify_effect.state_effect_targets`.  Frames whose
    true effect is an execution error (``complete_incorrect`` by default) are
    excluded, so the predictor learns what *should* happen rather than what did.
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("Install the ml dependencies to compute the loss.") from exc

    mask = effect_mask.clone()
    for label in exclude_labels:
        mask = mask & (typed_effect != label)
    if not bool(mask.any()):
        return expected_logits.sum() * 0.0
    return functional.cross_entropy(expected_logits[mask], typed_effect[mask])


def expected_effect_residual(observed_effect, expected_effect):
    """Bounded ``[0, 1]`` divergence between observed and expected effect.

    Uses the Bhattacharyya coefficient, matching the residual already computed in
    :mod:`aiops.models.stateverify_effect`, so the tracker sees a consistent
    effect-channel scale.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("Install the ml dependencies to compute the residual.") from exc

    coefficient = torch.sqrt(
        observed_effect.clamp_min(1e-8) * expected_effect.clamp_min(1e-8)
    ).sum(dim=-1)
    return (1.0 - coefficient).clamp(0.0, 1.0)
