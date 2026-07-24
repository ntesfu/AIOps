"""CaptainCook4D adapter for the adjudicator (dataset glue, kept out of core).

Turns CaptainCook4D step segments into ``Candidate``s, resolves before/contact/
effect evidence frames from the GoPro video (decord), supplies procedure context
from the recipe steps, and loads ground-truth (family + free-text description) for
attribution evaluation. The core adjudication stages never import this — it plugs
in via ``FrameProvider`` / ``ProcedureContext`` / ``Candidate``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from aiops.adjudication.types import Candidate, FrameRef
from aiops.data.captaincook4d import CaptainCook4DSegment, video_path

# (role, fraction-through-segment) — a small before/contact/effect story.
DEFAULT_ROLES: tuple[tuple[str, float], ...] = (
    ("before", 0.12),
    ("contact", 0.5),
    ("effect", 0.9),
)


def build_candidates(
    segments: Iterable[CaptainCook4DSegment],
    *,
    errors_only: bool = True,
    score: float = 1.0,
    leak_gt_category: bool = False,
) -> list[Candidate]:
    """Candidates from segments. ``errors_only`` keeps the flagged (error) steps;
    ``score`` is a placeholder detector score.

    ``detector_category`` is left ``None`` by default so the VLM attributes from
    the frames, not a prior — leaking the ground-truth family would invalidate the
    attribution eval. Set ``leak_gt_category=True`` only for an oracle-prior ablation.
    """
    cands: list[Candidate] = []
    for s in segments:
        if errors_only and s.label == 0:
            continue
        cands.append(Candidate(
            event_id=f"{s.recording_id}#{s.step_index}",
            recording_id=s.recording_id,
            step_id=s.step_id,
            step_description=s.description,
            detector_score=float(score),
            detector_category=(s.error_tags[0] if (leak_gt_category and s.error_tags) else None),
            metadata={"start_time": s.start_time, "end_time": s.end_time,
                      "step_index": s.step_index},
        ))
    return cands


def ground_truth(annotations_root: str | Path) -> dict[str, dict]:
    """``{event_id: {"family", "description"}}`` from error_annotations.json.

    event_id is ``{recording_id}#{position}``, matching ``build_candidates`` (the
    two annotation files align positionally)."""
    path = Path(annotations_root) / "annotation_json" / "error_annotations.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    gt: dict[str, dict] = {}
    for row in rows:
        rid = row["recording_id"]
        for i, step in enumerate(row.get("step_annotations", [])):
            errs = step.get("errors") or []
            if not errs:
                continue
            gt[f"{rid}#{i}"] = {"family": errs[0].get("tag", ""),
                                "description": errs[0].get("description", "")}
    return gt


class CaptainCook4DProcedureContext:
    """Expected action + neighbouring steps from the recipe."""

    def __init__(self, segments: Sequence[CaptainCook4DSegment]) -> None:
        self.by_rec: dict[str, dict[int, str]] = {}
        for s in segments:
            self.by_rec.setdefault(s.recording_id, {})[s.step_index] = s.description

    def expected_action(self, candidate: Candidate) -> Optional[str]:
        return candidate.step_description or None

    def surrounding_steps(self, candidate: Candidate) -> list[str]:
        idx = candidate.metadata.get("step_index")
        steps = self.by_rec.get(candidate.recording_id, {})
        out = []
        if isinstance(idx, int):
            for j in (idx - 1, idx + 1):
                if j in steps:
                    out.append(f"step {j}: {steps[j]}")
        return out


class CaptainCook4DFrameProvider:
    """Decode before/contact/effect frames for a candidate and cache them as JPEGs;
    returns ``FrameRef``s pointing at the files (a VLM backend loads them)."""

    def __init__(self, data_root: str | Path, cache_dir: str | Path,
                 roles: Sequence[tuple[str, float]] = DEFAULT_ROLES,
                 max_width: int = 640) -> None:
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.roles = tuple(roles)
        self.max_width = max_width
        self._readers: dict[str, object] = {}

    def _reader(self, recording_id: str):
        from decord import VideoReader
        if recording_id not in self._readers:
            self._readers[recording_id] = VideoReader(
                str(video_path(self.data_root, recording_id)))
        return self._readers[recording_id]

    def __call__(self, candidate: Candidate) -> list[FrameRef]:
        from PIL import Image

        start = float(candidate.metadata.get("start_time", 0.0))
        end = float(candidate.metadata.get("end_time", start))
        reader = self._reader(candidate.recording_id)
        n = len(reader)
        fps = float(reader.get_avg_fps()) or 30.0
        stem = candidate.event_id.replace("#", "_").replace("/", "_")
        refs: list[FrameRef] = []
        for role, frac in self.roles:
            t = start + frac * max(0.0, end - start)
            f = int(min(max(0, round(t * fps)), n - 1))
            pil = Image.fromarray(reader[f].asnumpy())  # RGB
            if pil.width > self.max_width:
                h = round(pil.height * self.max_width / pil.width)
                pil = pil.resize((self.max_width, h))
            out = self.cache_dir / f"{stem}_{role}.jpg"
            pil.save(out, quality=90)
            refs.append(FrameRef(str(out), role, timestamp=t))
        return refs
