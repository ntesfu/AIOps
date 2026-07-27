"""Viterbi MAP decode over step posteriors with a legal-transition prior.

Post-hoc smoothing/decode over per-frame STEP posteriors. It buys two things cheaply
(no retraining):

1. **Temporal stickiness** — a self-transition bias collapses the flicker that hurts
   Edit / segmental-F1 (the reference's largest post-hoc lift).
2. **Legality** — a ``forbidden`` transition mask (True = illegal) encodes the task
   graph; illegal step→step jumps get ``-inf`` prior. The mask is shared with Track B's
   graph auditor and can be filled from the procedure order or a real task graph.

Numpy only, so it runs and is tested locally without the GPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_NEG_INF = -1e30  # a finite stand-in for -inf so all-forbidden rows don't produce NaNs


def build_transition_matrix(
    num_steps: int,
    *,
    self_bias: float = 2.0,
    forbidden: Optional[np.ndarray] = None,
    forward_only: bool = False,
    background_step: int = 0,
) -> np.ndarray:
    """Return an ``(S, S)`` log-transition prior ``logT[i, j]`` (from step i to j).

    - ``self_bias`` adds log-weight to the diagonal (temporal stickiness).
    - ``forward_only`` forbids moving to a *lower-numbered* real step (monotone
      procedure); transitions to/from ``background_step`` stay free.
    - ``forbidden`` (bool ``(S, S)``) marks illegal transitions as ``-inf``.
    Rows are log-softmax normalized so the matrix is a proper conditional prior.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    log_t = np.zeros((num_steps, num_steps), dtype=np.float64)
    np.fill_diagonal(log_t, self_bias)

    if forward_only:
        for i in range(num_steps):
            for j in range(num_steps):
                if j == background_step or i == background_step or i == j:
                    continue
                if j < i:
                    log_t[i, j] = _NEG_INF

    if forbidden is not None:
        forbidden = np.asarray(forbidden, dtype=bool)
        if forbidden.shape != (num_steps, num_steps):
            raise ValueError("forbidden mask must be (num_steps, num_steps)")
        log_t = np.where(forbidden, _NEG_INF, log_t)

    # Row-wise log-softmax -> proper conditional prior; guard fully-masked rows.
    row_max = log_t.max(axis=1, keepdims=True)
    stabilized = log_t - row_max
    row_logsumexp = np.log(np.exp(stabilized).sum(axis=1, keepdims=True)) + row_max
    return log_t - row_logsumexp


def viterbi_decode(
    log_emissions: np.ndarray, log_transition: np.ndarray
) -> np.ndarray:
    """MAP step path. ``log_emissions`` is ``(T, S)`` log posteriors; returns ``(T,)``.

    Pass log posteriors (e.g. ``np.log(probs + eps)``). Argmax of ``log_emissions``
    alone recovers the greedy path; the transition prior regularizes it.
    """
    log_emissions = np.asarray(log_emissions, dtype=np.float64)
    if log_emissions.ndim != 2:
        raise ValueError("log_emissions must be (T, S)")
    num_frames, num_steps = log_emissions.shape
    if num_frames == 0:
        return np.empty(0, dtype=np.int64)
    if log_transition.shape != (num_steps, num_steps):
        raise ValueError("log_transition must be (S, S) matching log_emissions")

    dp = np.full((num_frames, num_steps), _NEG_INF, dtype=np.float64)
    back = np.zeros((num_frames, num_steps), dtype=np.int64)
    dp[0] = log_emissions[0]
    for t in range(1, num_frames):
        # scores[i, j] = dp[t-1, i] + logT[i, j]
        scores = dp[t - 1][:, None] + log_transition
        back[t] = np.argmax(scores, axis=0)
        dp[t] = log_emissions[t] + scores[back[t], np.arange(num_steps)]

    path = np.zeros(num_frames, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(num_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
