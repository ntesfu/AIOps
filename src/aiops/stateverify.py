from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Iterable

import numpy as np


class ComponentState(IntEnum):
    """Canonical persistent component states used by StateVerify-PSR.

    This order is intentionally independent of the historical StateGraph head,
    whose logits are ordered ``incorrect, pending, correct``. Call
    :func:`canonicalize_stategraph_emissions` at that boundary.
    """

    NOT_COMPLETED = 0
    INSTALLED_CORRECT = 1
    INSTALLED_INCORRECT = 2


@dataclass(frozen=True)
class TripleResidualEvidence:
    """Calibrated residual evidence on a common ``[time, component]`` axis.

    A value close to one means strong evidence of a mismatch. A value close to
    zero supports normal execution. Missing evidence is represented by a false
    entry in the corresponding mask and is neutral during fusion.
    """

    procedure: np.ndarray
    execution: np.ndarray
    effect: np.ndarray
    procedure_mask: np.ndarray | None = None
    execution_mask: np.ndarray | None = None
    effect_mask: np.ndarray | None = None

    @classmethod
    def from_arrays(
        cls,
        procedure: np.ndarray,
        execution: np.ndarray,
        effect: np.ndarray,
        *,
        num_components: int | None = None,
        procedure_mask: np.ndarray | None = None,
        execution_mask: np.ndarray | None = None,
        effect_mask: np.ndarray | None = None,
    ) -> "TripleResidualEvidence":
        execution_array = _as_time_component(execution, num_components, "execution")
        components = execution_array.shape[1]
        procedure_array = _as_time_component(procedure, components, "procedure")
        effect_array = _as_time_component(effect, components, "effect")
        if not (
            procedure_array.shape == execution_array.shape == effect_array.shape
        ):
            raise ValueError(
                "Procedure, execution, and effect residuals must share the same "
                f"[time, component] shape; received {procedure_array.shape}, "
                f"{execution_array.shape}, and {effect_array.shape}."
            )
        shape = execution_array.shape
        return cls(
            procedure=_probabilities(procedure_array, "procedure"),
            execution=_probabilities(execution_array, "execution"),
            effect=_probabilities(effect_array, "effect"),
            procedure_mask=_mask_or_finite(procedure_mask, procedure_array, shape),
            execution_mask=_mask_or_finite(execution_mask, execution_array, shape),
            effect_mask=_mask_or_finite(effect_mask, effect_array, shape),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.execution.shape

    def at(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack(
            [self.procedure[index], self.execution[index], self.effect[index]], axis=-1
        )
        masks = np.stack(
            [
                _resolved_mask(self.procedure_mask, self.procedure)[index],
                _resolved_mask(self.execution_mask, self.execution)[index],
                _resolved_mask(self.effect_mask, self.effect)[index],
            ],
            axis=-1,
        )
        return values, masks


@dataclass(frozen=True)
class BeliefTrackerConfig:
    """Configuration for the factorized causal component filter."""

    transition_matrix: tuple[tuple[float, float, float], ...] = (
        (0.96, 0.03, 0.01),
        (0.01, 0.985, 0.005),
        (0.01, 0.005, 0.985),
    )
    residual_weights: tuple[float, float, float] = (0.35, 0.75, 1.0)
    residual_strength: float = 0.75
    emission_temperature: float = 1.0
    alert_threshold: float = 0.65
    recovery_threshold: float = 0.70
    confirmation_steps: int = 2
    recovery_confirmation_steps: int = 2
    epsilon: float = 1e-6

    def validate(self) -> None:
        transition = np.asarray(self.transition_matrix, dtype=np.float64)
        if transition.shape != (3, 3):
            raise ValueError("transition_matrix must have shape [3, 3].")
        if not np.isfinite(transition).all() or (transition < 0).any():
            raise ValueError("transition_matrix must contain finite non-negative values.")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Every transition_matrix row must sum to one.")
        weights = np.asarray(self.residual_weights, dtype=np.float64)
        if weights.shape != (3,) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("residual_weights must be three non-negative values.")
        if self.residual_strength < 0:
            raise ValueError("residual_strength cannot be negative.")
        if self.emission_temperature <= 0:
            raise ValueError("emission_temperature must be positive.")
        for name, value in (
            ("alert_threshold", self.alert_threshold),
            ("recovery_threshold", self.recovery_threshold),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one.")
        if self.confirmation_steps <= 0 or self.recovery_confirmation_steps <= 0:
            raise ValueError("Confirmation windows must be positive.")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be between zero and 0.5.")


@dataclass(frozen=True)
class BeliefEvent:
    step: int
    component: int
    kind: str
    previous_state: str
    state: str
    confidence: float
    attribution: str
    residuals: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeliefTrace:
    posterior: np.ndarray
    state: np.ndarray
    alert_active: np.ndarray
    fused_fault_probability: np.ndarray
    residual_contribution: np.ndarray
    events: tuple[BeliefEvent, ...]

    def to_dict(self, component_names: Iterable[str] | None = None) -> dict[str, Any]:
        names = (
            list(component_names)
            if component_names is not None
            else [str(index) for index in range(self.posterior.shape[1])]
        )
        if len(names) != self.posterior.shape[1]:
            raise ValueError("component_names length does not match the trace.")
        events = []
        for event in self.events:
            row = event.to_dict()
            row["component_name"] = names[event.component]
            events.append(row)
        return {
            "state_order": [state.name.lower() for state in ComponentState],
            "component_names": names,
            "posterior": self.posterior.tolist(),
            "state": self.state.tolist(),
            "alert_active": self.alert_active.tolist(),
            "fused_fault_probability": self.fused_fault_probability.tolist(),
            "residual_contribution": self.residual_contribution.tolist(),
            "events": events,
        }


class TypedComponentBeliefTracker:
    """Causal factorized Bayesian filter with auditable alert hysteresis.

    The tracker never couples components through a Cartesian global state. Each
    component has three persistent states, while procedure constraints enter as
    an explicit residual evidence channel. This keeps inference linear in the
    number of components and lets downstream systems explain every alert.
    """

    def __init__(self, num_components: int, config: BeliefTrackerConfig | None = None):
        if num_components <= 0:
            raise ValueError("num_components must be positive.")
        self.num_components = num_components
        self.config = config or BeliefTrackerConfig()
        self.config.validate()
        self._transition = np.asarray(
            self.config.transition_matrix, dtype=np.float64
        )
        self._weights = np.asarray(self.config.residual_weights, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._belief = np.zeros((self.num_components, 3), dtype=np.float64)
        self._belief[:, ComponentState.NOT_COMPLETED] = 1.0
        self._alert_active = np.zeros(self.num_components, dtype=np.bool_)
        self._alert_streak = np.zeros(self.num_components, dtype=np.int64)
        self._recovery_streak = np.zeros(self.num_components, dtype=np.int64)
        self._last_residuals = np.full((self.num_components, 3), np.nan)
        self._last_masks = np.zeros((self.num_components, 3), dtype=np.bool_)
        self._step = 0

    def run(
        self,
        state_emissions: np.ndarray,
        residuals: TripleResidualEvidence,
    ) -> BeliefTrace:
        emissions = canonical_state_emissions(state_emissions)
        if emissions.shape[1] != self.num_components:
            raise ValueError(
                f"Tracker has {self.num_components} components but emissions have "
                f"{emissions.shape[1]}."
            )
        if residuals.shape != emissions.shape[:2]:
            raise ValueError(
                f"Residual shape {residuals.shape} does not match emissions "
                f"{emissions.shape[:2]}."
            )
        self.reset()
        posteriors = []
        states = []
        alerts = []
        fault_probabilities = []
        contributions = []
        events: list[BeliefEvent] = []
        for index in range(emissions.shape[0]):
            values, masks = residuals.at(index)
            step = self.update(emissions[index], values, masks)
            posteriors.append(step["posterior"])
            states.append(step["state"])
            alerts.append(step["alert_active"])
            fault_probabilities.append(step["fused_fault_probability"])
            contributions.append(step["residual_contribution"])
            events.extend(step["events"])
        return BeliefTrace(
            posterior=np.asarray(posteriors),
            state=np.asarray(states),
            alert_active=np.asarray(alerts),
            fused_fault_probability=np.asarray(fault_probabilities),
            residual_contribution=np.asarray(contributions),
            events=tuple(events),
        )

    def update(
        self,
        state_emission: np.ndarray,
        residual_values: np.ndarray,
        residual_masks: np.ndarray | None = None,
    ) -> dict[str, Any]:
        emission = canonical_state_emissions(state_emission)
        if emission.shape != (self.num_components, 3):
            raise ValueError(
                f"state_emission must have shape [{self.num_components}, 3]."
            )
        values = np.asarray(residual_values, dtype=np.float64)
        if values.shape != (self.num_components, 3):
            raise ValueError(
                f"residual_values must have shape [{self.num_components}, 3]."
            )
        masks = (
            np.isfinite(values)
            if residual_masks is None
            else np.asarray(residual_masks, dtype=np.bool_)
        )
        if masks.shape != values.shape:
            raise ValueError("residual_masks must match residual_values.")
        if ((values[masks] < 0) | (values[masks] > 1)).any():
            raise ValueError("Observed residual values must lie in [0, 1].")

        safe_values = np.where(masks, values, 0.5)
        fused_fault, contribution = self._fuse_residuals(safe_values, masks)
        prior = self._belief @ self._transition
        tempered_emission = np.power(
            np.clip(emission, self.config.epsilon, 1.0),
            1.0 / self.config.emission_temperature,
        )
        residual_odds = fused_fault / np.clip(
            1.0 - fused_fault, self.config.epsilon, None
        )
        tempered_emission[:, ComponentState.INSTALLED_INCORRECT] *= np.power(
            residual_odds, self.config.residual_strength
        )
        posterior = prior * tempered_emission
        posterior /= posterior.sum(axis=-1, keepdims=True).clip(
            min=self.config.epsilon
        )

        previous_state = self._belief.argmax(axis=-1)
        current_state = posterior.argmax(axis=-1)
        self._belief = posterior
        self._last_residuals = values.copy()
        self._last_masks = masks.copy()

        events: list[BeliefEvent] = []
        incorrect_probability = posterior[:, ComponentState.INSTALLED_INCORRECT]
        recovered_probability = np.maximum(
            posterior[:, ComponentState.NOT_COMPLETED],
            posterior[:, ComponentState.INSTALLED_CORRECT],
        )
        for component in range(self.num_components):
            if not self._alert_active[component]:
                self._alert_streak[component] = (
                    self._alert_streak[component] + 1
                    if incorrect_probability[component] >= self.config.alert_threshold
                    else 0
                )
                if self._alert_streak[component] >= self.config.confirmation_steps:
                    self._alert_active[component] = True
                    self._recovery_streak[component] = 0
                    events.append(
                        self._event(
                            component,
                            "alert",
                            previous_state[component],
                            ComponentState.INSTALLED_INCORRECT,
                            incorrect_probability[component],
                            contribution[component],
                        )
                    )
            else:
                self._recovery_streak[component] = (
                    self._recovery_streak[component] + 1
                    if recovered_probability[component]
                    >= self.config.recovery_threshold
                    else 0
                )
                if (
                    self._recovery_streak[component]
                    >= self.config.recovery_confirmation_steps
                ):
                    self._alert_active[component] = False
                    self._alert_streak[component] = 0
                    events.append(
                        self._event(
                            component,
                            "recovery",
                            ComponentState.INSTALLED_INCORRECT,
                            current_state[component],
                            recovered_probability[component],
                            contribution[component],
                        )
                    )

        result = {
            "posterior": posterior.copy(),
            "state": current_state.copy(),
            "alert_active": self._alert_active.copy(),
            "fused_fault_probability": fused_fault.copy(),
            "residual_contribution": contribution.copy(),
            "events": tuple(events),
        }
        self._step += 1
        return result

    def _fuse_residuals(
        self, values: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(
            values, self.config.epsilon, 1.0 - self.config.epsilon
        )
        logits = np.log(clipped) - np.log1p(-clipped)
        active_weights = masks * self._weights
        weight_sum = active_weights.sum(axis=-1)
        fused_logit = np.divide(
            (logits * active_weights).sum(axis=-1),
            weight_sum,
            out=np.zeros(self.num_components, dtype=np.float64),
            where=weight_sum > 0,
        )
        probability = 1.0 / (1.0 + np.exp(-fused_logit))
        magnitude = np.maximum(logits, 0.0) * active_weights
        contribution = np.divide(
            magnitude,
            magnitude.sum(axis=-1, keepdims=True),
            out=np.zeros_like(magnitude),
            where=magnitude.sum(axis=-1, keepdims=True) > 0,
        )
        return probability, contribution

    def _event(
        self,
        component: int,
        kind: str,
        previous_state: int | ComponentState,
        state: int | ComponentState,
        confidence: float,
        contribution: np.ndarray,
    ) -> BeliefEvent:
        labels = ("procedure", "execution", "effect")
        attribution = (
            labels[int(contribution.argmax())] if contribution.max() > 0 else "unknown"
        )
        residuals = {
            label: (
                float(self._last_residuals[component, index])
                if self._last_masks[component, index]
                else None
            )
            for index, label in enumerate(labels)
        }
        return BeliefEvent(
            step=self._step,
            component=component,
            kind=kind,
            previous_state=ComponentState(int(previous_state)).name.lower(),
            state=ComponentState(int(state)).name.lower(),
            confidence=float(confidence),
            attribution=attribution,
            residuals=residuals,
        )


def canonicalize_stategraph_emissions(stategraph_probabilities: np.ndarray) -> np.ndarray:
    """Map historical ``incorrect, pending, correct`` probabilities to canonical order."""

    probabilities = canonical_state_emissions(stategraph_probabilities)
    return probabilities[..., [1, 2, 0]]


def canonical_state_emissions(emissions: np.ndarray) -> np.ndarray:
    values = np.asarray(emissions, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[-1] != 3:
        raise ValueError(
            "State emissions must have shape [component, 3] or [time, component, 3]."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("State emissions must be finite and non-negative.")
    totals = values.sum(axis=-1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Every state emission row must contain positive mass.")
    return values / totals


def _as_time_component(
    values: np.ndarray, num_components: int | None, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if num_components is None:
            raise ValueError(
                f"num_components is required when {name} is a one-dimensional timeline."
            )
        array = np.repeat(array[:, None], num_components, axis=1)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time] or [time, component].")
    if num_components is not None and array.shape[1] != num_components:
        raise ValueError(
            f"{name} has {array.shape[1]} components; expected {num_components}."
        )
    return array


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if ((finite < 0) | (finite > 1)).any():
        raise ValueError(f"{name} residuals must lie in [0, 1].")
    return np.where(np.isfinite(values), values, 0.5)


def _mask_or_finite(
    mask: np.ndarray | None, original: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    if mask is None:
        return np.isfinite(original)
    resolved = np.asarray(mask, dtype=np.bool_)
    if resolved.ndim == 1:
        resolved = np.repeat(resolved[:, None], shape[1], axis=1)
    if resolved.shape != shape:
        raise ValueError(f"Residual mask has shape {resolved.shape}; expected {shape}.")
    return resolved & np.isfinite(original)


def _resolved_mask(mask: np.ndarray | None, values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) if mask is None else mask
