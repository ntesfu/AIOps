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

from typing import Optional, Sequence

import numpy as np

_NEG_INF = -1e30  # a finite stand-in for -inf so all-forbidden rows don't produce NaNs


def build_transition_matrix(
    num_steps: int,
    *,
    self_bias: float = 2.0,
    forbidden: Optional[np.ndarray] = None,
    transition_penalty: Optional[np.ndarray] = None,
    forward_only: bool = False,
    background_step: int = 0,
) -> np.ndarray:
    """Return an ``(S, S)`` log-transition prior ``logT[i, j]`` (from step i to j).

    - ``self_bias`` adds log-weight to the diagonal (temporal stickiness).
    - ``forward_only`` forbids moving to a *lower-numbered* real step (monotone
      procedure); transitions to/from ``background_step`` stay free.
    - ``transition_penalty`` adds a finite train-derived log score to each edge.
    - ``forbidden`` (bool ``(S, S)``) marks illegal transitions as ``-inf``.
    Rows are log-softmax normalized so the matrix is a proper conditional prior.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    log_t = np.zeros((num_steps, num_steps), dtype=np.float64)
    np.fill_diagonal(log_t, self_bias)

    if transition_penalty is not None:
        transition_penalty = np.asarray(transition_penalty, dtype=np.float64)
        if transition_penalty.shape != (num_steps, num_steps):
            raise ValueError("transition_penalty must be (num_steps, num_steps)")
        if not np.isfinite(transition_penalty).all():
            raise ValueError("transition_penalty must contain only finite values")
        log_t += transition_penalty

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


def build_duration_logpmf(
    num_steps: int,
    max_length: int,
    *,
    mean_length: float,
    sigma: float = 0.7,
    weight: float = 1.0,
    per_step_mean: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return an ``(S, max_length + 1)`` per-step segment-length log-prior table.

    A discretized **log-normal** length model centred on ``mean_length`` (or a
    ``per_step_mean`` vector). This is the piece the plain-Viterbi stickiness bias
    lacks: instead of a per-frame self-transition reward, it prices a *whole
    segment's* length, so implausibly short segments (the fragmentation that tanks
    Edit / F1@50) are penalized directly while the true step duration is favoured.

    - ``sigma`` is the log-length spread (larger = flatter prior).
    - Each row is normalized to ``logsumexp == 0`` then scaled by ``weight``, so
      ``weight`` is an interpretable prior strength (``1`` = proper log-pmf,
      ``>1`` sharpens the duration preference, ``0`` disables it).
    - Column 0 (length 0) is ``-inf`` (a segment has at least one frame).
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    lengths = np.arange(max_length + 1, dtype=np.float64)
    ln_len = np.log(np.clip(lengths, 1.0, None))
    table = np.empty((num_steps, max_length + 1), dtype=np.float64)
    sigma = max(float(sigma), 1e-3)
    for s in range(num_steps):
        mu = float(per_step_mean[s]) if per_step_mean is not None else float(mean_length)
        mu = max(mu, 1.0)
        logpdf = -((ln_len - np.log(mu)) ** 2) / (2.0 * sigma * sigma) - ln_len
        logpdf[0] = _NEG_INF
        row_max = logpdf[1:].max()
        row_lse = np.log(np.exp(logpdf[1:] - row_max).sum()) + row_max
        logpdf[1:] = weight * (logpdf[1:] - row_lse)
        table[s] = logpdf
    return table


def _semi_markov_forward(
    log_emissions: np.ndarray,
    log_transition: np.ndarray,
    duration_logpmf: np.ndarray,
    *,
    min_segment: int,
    max_segment: int,
    boundary_logprob: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass of a segmental (semi-Markov) MAP decode.

    ``dp[e, s]`` = best score of a segmentation of frames ``[0, e)`` whose final
    segment carries label ``s`` and ends exactly at ``e``. A segment ``[start, e)``
    scores ``sum(log_emissions[start:e, s]) + duration_logpmf[s, e-start]`` plus the
    inter-segment transition ``log_transition[prev, s]`` (self-transitions are
    forbidden so segmentation is canonical). Returns ``dp`` and the two
    backpointers ``bp_start`` / ``bp_prev`` for reconstruction.
    """
    num_frames, num_steps = log_emissions.shape
    cum = np.zeros((num_frames + 1, num_steps), dtype=np.float64)
    cum[1:] = np.cumsum(log_emissions, axis=0)
    log_t = np.array(log_transition, dtype=np.float64, copy=True)
    np.fill_diagonal(log_t, _NEG_INF)  # canonical: no segment->same-label segment
    dur_width = duration_logpmf.shape[1] - 1

    dp = np.full((num_frames + 1, num_steps), _NEG_INF, dtype=np.float64)
    bp_start = np.full((num_frames + 1, num_steps), -1, dtype=np.int64)
    bp_prev = np.full((num_frames + 1, num_steps), -1, dtype=np.int64)
    # trans_in_by_start[start, s] = max_prev dp[start, prev] + logT[prev, s]
    trans_in = np.full((num_frames + 1, num_steps), _NEG_INF, dtype=np.float64)
    trans_choice = np.full((num_frames + 1, num_steps), -1, dtype=np.int64)
    trans_in[0] = 0.0  # empty prefix: first segment pays no transition

    for e in range(1, num_frames + 1):
        lo = max(0, e - max_segment)
        hi = e - min_segment  # inclusive largest start
        for start in range(lo, hi + 1):
            seg_len = e - start
            seg_emiss = cum[e] - cum[start]
            dur = duration_logpmf[:, min(seg_len, dur_width)]
            cand = seg_emiss + dur + trans_in[start]
            if start > 0 and boundary_logprob is not None:
                cand = cand + boundary_logprob[start]
            better = cand > dp[e]
            if better.any():
                dp[e][better] = cand[better]
                bp_start[e][better] = start
                bp_prev[e][better] = trans_choice[start][better]
        # dp[e] is now final; publish its transition-out scores for later ends.
        scores = dp[e][:, None] + log_t  # (prev, cur)
        trans_choice[e] = np.argmax(scores, axis=0)
        trans_in[e] = scores[trans_choice[e], np.arange(num_steps)]
    return dp, bp_start, bp_prev


def _semi_markov_labels_from(
    dp: np.ndarray,
    bp_start: np.ndarray,
    bp_prev: np.ndarray,
    end: int,
) -> list[tuple[int, int, int]]:
    """Backtrace the MAP segmentation of ``[0, end)`` as ``(start, end, label)``."""
    segments: list[tuple[int, int, int]] = []
    e = end
    s = int(np.argmax(dp[e]))
    while e > 0:
        start = int(bp_start[e, s])
        prev = int(bp_prev[e, s])
        if start < 0:  # unreachable guard
            break
        segments.append((start, e, s))
        e, s = start, prev
        if s < 0:
            break
    segments.reverse()
    return segments


def semi_markov_decode(
    log_emissions: np.ndarray,
    log_transition: np.ndarray,
    duration_logpmf: np.ndarray,
    *,
    min_segment: int = 1,
    max_segment: Optional[int] = None,
    boundary_logprob: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Offline segmental (semi-Markov) MAP step path; returns ``(T,)``.

    Unlike :func:`viterbi_decode` (frame-Markov + a self-transition bias), this
    scores whole segments against a per-step **duration** prior, so segment
    boundaries fall where the length model and emissions agree rather than wherever
    the per-frame stickiness happens to break. That is the boundary-quality lever
    for Edit / F1@50 without copying a learned refinement decoder.
    """
    log_emissions = np.asarray(log_emissions, dtype=np.float64)
    if log_emissions.ndim != 2:
        raise ValueError("log_emissions must be (T, S)")
    num_frames, num_steps = log_emissions.shape
    if num_frames == 0:
        return np.empty(0, dtype=np.int64)
    if log_transition.shape != (num_steps, num_steps):
        raise ValueError("log_transition must be (S, S) matching log_emissions")
    if duration_logpmf.shape[0] != num_steps:
        raise ValueError("duration_logpmf must have num_steps rows")
    min_segment = max(1, int(min_segment))
    max_segment = num_frames if max_segment is None else min(int(max_segment), num_frames)
    max_segment = max(max_segment, min_segment)
    dp, bp_start, bp_prev = _semi_markov_forward(
        log_emissions, log_transition, duration_logpmf,
        min_segment=min_segment, max_segment=max_segment,
        boundary_logprob=boundary_logprob,
    )
    path = np.zeros(num_frames, dtype=np.int64)
    for start, end, label in _semi_markov_labels_from(dp, bp_start, bp_prev, num_frames):
        path[start:end] = label
    return path


def semi_markov_decode_fixed_lag(
    log_emissions: np.ndarray,
    log_transition: np.ndarray,
    duration_logpmf: np.ndarray,
    lag: int,
    *,
    min_segment: int = 1,
    max_segment: Optional[int] = None,
    boundary_logprob: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Near-online segmental decode: frame ``f`` is committed from the MAP
    segmentation of ``[0, min(f + lag + 1, T))`` — a bounded ``lag``-frame trailing
    lookahead, mirroring :func:`viterbi_decode_fixed_lag`. ``lag=0`` is strict
    causal; ``lag >= T`` equals the offline :func:`semi_markov_decode`.
    """
    log_emissions = np.asarray(log_emissions, dtype=np.float64)
    if log_emissions.ndim != 2:
        raise ValueError("log_emissions must be (T, S)")
    if lag < 0:
        raise ValueError("lag must be >= 0")
    num_frames, num_steps = log_emissions.shape
    if num_frames == 0:
        return np.empty(0, dtype=np.int64)
    if log_transition.shape != (num_steps, num_steps):
        raise ValueError("log_transition must be (S, S) matching log_emissions")
    min_segment = max(1, int(min_segment))
    max_segment = num_frames if max_segment is None else min(int(max_segment), num_frames)
    max_segment = max(max_segment, min_segment)
    dp, bp_start, bp_prev = _semi_markov_forward(
        log_emissions, log_transition, duration_logpmf,
        min_segment=min_segment, max_segment=max_segment,
        boundary_logprob=boundary_logprob,
    )
    out = np.zeros(num_frames, dtype=np.int64)
    for f in range(num_frames):
        end = min(f + 1 + lag, num_frames)
        e = end
        s = int(np.argmax(dp[e]))
        committed = False
        while e > 0:
            start = int(bp_start[e, s])
            prev = int(bp_prev[e, s])
            if start < 0:
                break
            if start <= f < e:
                out[f] = s
                committed = True
                break
            e, s = start, prev
            if s < 0:
                break
        if not committed:
            out[f] = int(np.argmax(log_emissions[f]))
    return out


def viterbi_decode_fixed_lag(
    log_emissions: np.ndarray, log_transition: np.ndarray, lag: int
) -> np.ndarray:
    """Near-online (fixed-lag) MAP decode: frame ``f``'s label is committed using
    observations only up to ``f + lag`` (a bounded trailing lookahead), matching the
    plan's ~1-2 s near-online regime.

    The forward Viterbi DP is causal (``dp[t]`` depends only on frames ``0..t``), so
    reading back from the best state at ``min(f+lag, T-1)`` uses exactly a ``lag``-frame
    lookahead. ``lag=0`` is fully online (no lookahead); ``lag >= T-1`` equals the
    offline :func:`viterbi_decode`.
    """
    log_emissions = np.asarray(log_emissions, dtype=np.float64)
    if log_emissions.ndim != 2:
        raise ValueError("log_emissions must be (T, S)")
    if lag < 0:
        raise ValueError("lag must be >= 0")
    num_frames, num_steps = log_emissions.shape
    if num_frames == 0:
        return np.empty(0, dtype=np.int64)
    if log_transition.shape != (num_steps, num_steps):
        raise ValueError("log_transition must be (S, S) matching log_emissions")

    dp = np.full((num_frames, num_steps), _NEG_INF, dtype=np.float64)
    back = np.zeros((num_frames, num_steps), dtype=np.int64)
    dp[0] = log_emissions[0]
    for t in range(1, num_frames):
        scores = dp[t - 1][:, None] + log_transition
        back[t] = np.argmax(scores, axis=0)
        dp[t] = log_emissions[t] + scores[back[t], np.arange(num_steps)]

    out = np.zeros(num_frames, dtype=np.int64)
    for f in range(num_frames):
        end = min(f + lag, num_frames - 1)
        state = int(np.argmax(dp[end]))
        for t in range(end, f, -1):  # backtrace from end down to f
            state = int(back[t, state])
        out[f] = state
    return out
