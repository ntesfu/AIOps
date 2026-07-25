"""Event-triggered VLM mistake adjudicator — a pluggable pipeline.

Public API. Import stages and swap them freely::

    from aiops.adjudication import build_default_pipeline, MockVLMBackend, Candidate

    pipe = build_default_pipeline(backend=MockVLMBackend())
    results = pipe.adjudicate(candidates)

Each stage (trigger, evidence builder, prompt builder, backend, parser, fuser) is a
Protocol with a default implementation; replace any one via ``build_default_pipeline``
kwargs or by constructing ``AdjudicationPipeline`` directly.
"""

from aiops.adjudication.backends import (
    ClaudeVLMBackend,
    MockVLMBackend,
    QwenVLBackend,
    VLMBackend,
)
from aiops.adjudication.evaluation import AttributionScores, evaluate_attributions
from aiops.adjudication.evidence import (
    DefaultEvidenceBuilder,
    EvidenceBuilder,
    NullProcedureContext,
    ProcedureContext,
)
from aiops.adjudication.fusion import EnrichmentFuser, Fuser
from aiops.adjudication.parsing import JsonResponseParser, ResponseParser
from aiops.adjudication.pipeline import AdjudicationPipeline, build_default_pipeline
from aiops.adjudication.prompts import (
    DEFAULT_FEWSHOT,
    FAMILY_DEFINITIONS,
    ChainOfThoughtPromptBuilder,
    FewShotExemplar,
    PromptBuilder,
    ZeroShotPromptBuilder,
)
from aiops.adjudication.schema import (
    ATTRIBUTION_SCHEMA,
    coerce_attribution,
    validate_attribution,
)
from aiops.adjudication.triggers import (
    AllEventsTrigger,
    ScoreThresholdTrigger,
    TopKTrigger,
    Trigger,
)
from aiops.adjudication.types import (
    MISTAKE_FAMILIES,
    AdjudicationResult,
    Alert,
    Candidate,
    EnrichedAlert,
    EvidencePacket,
    FrameRef,
    MistakeAttribution,
    Prompt,
    normalize_family,
)

__all__ = [
    "AdjudicationPipeline", "build_default_pipeline",
    "VLMBackend", "MockVLMBackend", "ClaudeVLMBackend", "QwenVLBackend",
    "Trigger", "AllEventsTrigger", "ScoreThresholdTrigger", "TopKTrigger",
    "EvidenceBuilder", "DefaultEvidenceBuilder", "ProcedureContext", "NullProcedureContext",
    "PromptBuilder", "ZeroShotPromptBuilder", "ChainOfThoughtPromptBuilder",
    "FewShotExemplar", "FAMILY_DEFINITIONS", "DEFAULT_FEWSHOT",
    "ResponseParser", "JsonResponseParser",
    "Fuser", "EnrichmentFuser",
    "evaluate_attributions", "AttributionScores",
    "ATTRIBUTION_SCHEMA", "validate_attribution", "coerce_attribution",
    "Candidate", "EvidencePacket", "Prompt", "FrameRef", "MistakeAttribution",
    "AdjudicationResult", "Alert", "EnrichedAlert",
    "MISTAKE_FAMILIES", "normalize_family",
]
