"""Trigger stage — decides which detection events are worth a VLM call.

Event-triggering keeps the adjudicator cheap (it runs only on flagged events).
Swap the policy freely; all triggers take candidates and return a subset.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from aiops.adjudication.types import Candidate


@runtime_checkable
class Trigger(Protocol):
    def select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        ...


class AllEventsTrigger:
    """Adjudicate everything (useful for offline evaluation)."""

    def select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        return list(candidates)


class ScoreThresholdTrigger:
    """Adjudicate events whose detector score clears a threshold."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        return [c for c in candidates if c.detector_score >= self.threshold]


class TopKTrigger:
    """Adjudicate the K highest-scoring events (bounds VLM cost per window)."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        return sorted(candidates, key=lambda c: c.detector_score, reverse=True)[: self.k]
