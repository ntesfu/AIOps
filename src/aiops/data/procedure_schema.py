from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CACHE_SCHEMA_VERSION = 2
EVENT_OUTCOME_NAMES = ("correct", "incorrect", "remove")


@dataclass(frozen=True)
class ProcedureSchema:
    """Dataset adapter contract for procedure and completion supervision.

    The temporal model only depends on the canonical names and tensor shapes in
    this schema. Dataset-specific raw IDs are resolved while building a cache.
    """

    name: str
    dataset: str
    completion_components: tuple[str, ...]
    state_components: tuple[str, ...]
    raw_completion_map: Mapping[str, str]
    description_aliases: Mapping[str, str]
    event_outcomes: tuple[str, ...] = EVENT_OUTCOME_NAMES
    schema_version: int = CACHE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcedureSchema":
        schema = cls(
            name=str(payload["name"]),
            dataset=str(payload.get("dataset", payload["name"])),
            completion_components=tuple(str(value) for value in payload["completion_components"]),
            state_components=tuple(str(value) for value in payload.get("state_components", ())),
            raw_completion_map={str(key): str(value) for key, value in payload["raw_completion_map"].items()},
            description_aliases={
                _normalize_text(str(key)): str(value)
                for key, value in payload.get("description_aliases", {}).items()
            },
            event_outcomes=tuple(str(value) for value in payload.get("event_outcomes", EVENT_OUTCOME_NAMES)),
            schema_version=int(payload.get("schema_version", CACHE_SCHEMA_VERSION)),
        )
        schema.validate()
        return schema

    @classmethod
    def load(cls, path: str | Path) -> "ProcedureSchema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Procedure schema version {self.schema_version} is unsupported; "
                f"expected {CACHE_SCHEMA_VERSION}."
            )
        if not self.completion_components:
            raise ValueError("completion_components must not be empty")
        if len(set(self.completion_components)) != len(self.completion_components):
            raise ValueError("completion_components contains duplicate names")
        if tuple(self.event_outcomes) != EVENT_OUTCOME_NAMES:
            raise ValueError(f"event_outcomes must be {EVENT_OUTCOME_NAMES}")
        known = set(self.completion_components)
        unknown = (set(self.raw_completion_map.values()) | set(self.description_aliases.values())) - known
        if unknown:
            raise ValueError(f"Completion mappings reference unknown components: {sorted(unknown)}")

    def resolve_component(self, raw_id: str, description: str) -> int:
        component = self.raw_completion_map.get(str(raw_id).strip())
        normalized = _normalize_completion_description(description)
        if component is None:
            component = self.description_aliases.get(normalized)
        if component is None:
            # Exact canonical-name fallback makes small adapters usable without
            # enumerating every textual spelling in their source annotations.
            canonical = {_normalize_text(name): name for name in self.completion_components}
            component = canonical.get(normalized)
        if component is None:
            raise KeyError(
                f"Raw completion {raw_id!r} ({description!r}) is not mapped by schema {self.name!r}."
            )
        return self.completion_components.index(component)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "procedure_schema": self.name,
            "dataset": self.dataset,
            "completion_components": list(self.completion_components),
            "state_components": list(self.state_components),
            "event_outcomes": list(self.event_outcomes),
        }


def _normalize_completion_description(description: str) -> str:
    text = _normalize_text(description)
    prefixes = (
        "incorrectly ",
        "incorrect ",
        "correctly ",
        "correct ",
        "wrong ",
        "remove ",
        "removed ",
        "install ",
        "installed ",
        "complete ",
        "completed ",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()
