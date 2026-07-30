"""Explicit latency budgeting for near-online STEP recognition."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LookaheadBudget:
    """Quantized lookahead split between the neural head and decoder."""

    requested_seconds: float
    seconds_per_step: float
    total_steps: int
    neural_steps: int
    decoder_steps: int

    @property
    def effective_seconds(self) -> float:
        value = self.total_steps * self.seconds_per_step
        # Report an exact configured multiple as the request rather than a
        # one-ULP floating-point overshoot.
        if value > self.requested_seconds and math.isclose(
            value, self.requested_seconds, rel_tol=1e-12, abs_tol=1e-15
        ):
            return self.requested_seconds
        return value

    @property
    def decoder_seconds(self) -> float:
        return self.decoder_steps * self.seconds_per_step


def quantize_lookahead(
    requested_seconds: float,
    seconds_per_step: float,
    *,
    neural_steps: int = 0,
) -> LookaheadBudget:
    """Floor a latency budget to cached rows without exceeding the request.

    A small relative tolerance recovers exact mathematical multiples that land
    just below an integer in binary floating point. It is far smaller than a
    cache row, so non-multiples are still floored.
    """

    if not math.isfinite(requested_seconds) or requested_seconds < 0:
        raise ValueError("requested_seconds must be finite and non-negative")
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be finite and positive")
    if isinstance(neural_steps, bool) or not isinstance(neural_steps, int):
        raise ValueError("neural_steps must be a non-negative integer")
    if neural_steps < 0:
        raise ValueError("neural_steps must be a non-negative integer")

    ratio = requested_seconds / seconds_per_step
    tolerance = 1e-12 * max(1.0, abs(ratio))
    total_steps = max(0, int(math.floor(ratio + tolerance)))
    if neural_steps > total_steps:
        raise ValueError(
            "neural right context exceeds the quantized lookahead budget: "
            f"{neural_steps} > {total_steps}"
        )
    return LookaheadBudget(
        requested_seconds=float(requested_seconds),
        seconds_per_step=float(seconds_per_step),
        total_steps=total_steps,
        neural_steps=neural_steps,
        decoder_steps=total_steps - neural_steps,
    )
