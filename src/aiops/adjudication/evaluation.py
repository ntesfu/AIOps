"""Attribution metrics — score adjudications against ground-truth error labels.

CaptainCook4D provides a per-error ``tag`` (family) and free-text ``description``,
so attribution is directly evaluable: family accuracy (canonicalized), a lightweight
description token-F1, and coverage (fraction of events that produced a valid
attribution). Swap in a stronger text metric or human eval later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from aiops.adjudication.schema import canonical_family
from aiops.adjudication.types import AdjudicationResult, normalize_family

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def description_token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(g)
    return 2 * prec * rec / (prec + rec)


@dataclass
class AttributionScores:
    n: int
    coverage: float
    family_accuracy: float
    description_f1: float
    mean_confidence: float


def evaluate_attributions(
    results: Sequence[AdjudicationResult],
    ground_truth: Mapping[str, dict],
) -> AttributionScores:
    """``ground_truth[event_id] = {"family": str, "description": str}``."""
    n = len(results)
    ok = [r for r in results if r.status == "ok" and r.attribution is not None]
    coverage = len(ok) / n if n else 0.0
    fam_hits, f1_sum, conf_sum, scored = 0, 0.0, 0.0, 0
    for r in ok:
        gt = ground_truth.get(r.event_id)
        conf_sum += r.attribution.confidence
        if not gt:
            continue
        scored += 1
        if canonical_family(r.attribution) == normalize_family(gt.get("family", "")):
            fam_hits += 1
        f1_sum += description_token_f1(r.attribution.description, gt.get("description", ""))
    return AttributionScores(
        n=n,
        coverage=round(coverage, 4),
        family_accuracy=round(fam_hits / scored, 4) if scored else 0.0,
        description_f1=round(f1_sum / scored, 4) if scored else 0.0,
        mean_confidence=round(conf_sum / len(ok), 4) if ok else 0.0,
    )
