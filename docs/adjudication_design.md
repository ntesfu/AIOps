# VLM mistake adjudicator — pluggable design

**What it is.** Phase 4 of the error-first plan: an *event-triggered* module that,
when a detection fires, attributes **what the mistake is** (family, description,
expected-vs-observed, evidence, confidence) — the enrichment a scalar detector
cannot give. It **never raises or suppresses** the primary alert; it only adds
information. Lives in [src/aiops/adjudication/](../src/aiops/adjudication/).

## Pipeline — six swappable stages

```
Candidate ─▶ Trigger ─▶ EvidenceBuilder ─▶ PromptBuilder ─▶ VLMBackend ─▶ ResponseParser ─▶ (Fuser)
             which          what the           model-           raw text      validated        attach to
             events         VLM sees           agnostic prompt                 MistakeAttribution  the Alert
```

Every stage is a `typing.Protocol` with a default implementation; replace any one
via `build_default_pipeline(...)` kwargs or by constructing `AdjudicationPipeline`
directly. The pipeline depends only on the protocols, never on a concrete model or
dataset — so parts disconnect and iterate independently.

| Stage | Protocol | Defaults |
|---|---|---|
| Trigger | `Trigger` | `AllEventsTrigger`, `ScoreThresholdTrigger`, `TopKTrigger` |
| Evidence | `EvidenceBuilder` (+ `FrameProvider`, `ProcedureContext`) | `DefaultEvidenceBuilder` |
| Prompt | `PromptBuilder` | `ZeroShotPromptBuilder`, `ChainOfThoughtPromptBuilder` |
| Backend | `VLMBackend` | `MockVLMBackend`, `ClaudeVLMBackend`, `QwenVLBackend` |
| Parser | `ResponseParser` | `JsonResponseParser` (fence/prose-tolerant, coercing) |
| Fusion | `Fuser` | `EnrichmentFuser` (never suppresses; flags disagreement) |

## Design invariants

- **Non-blocking.** Fusion only attaches; the alert's fire/no-fire is never touched.
  A VLM "no error" verdict is recorded as `disagreement`, not a suppression.
- **Graceful degradation.** A malformed model response is coerced with safe
  defaults (or `schema_invalid`); a backend exception is isolated per event
  (`backend_error`) so one bad event never sinks a batch.
- **Offline-first.** `MockVLMBackend` runs the whole pipeline with no API/GPU, so
  triggers, evidence, prompts, parsing, fusion, and metrics are all testable
  ([tests/test_adjudication.py](../tests/test_adjudication.py), 12 tests) and
  demoable ([scripts/demo_adjudication.py](../scripts/demo_adjudication.py)).

## The structured output (`ATTRIBUTION_SCHEMA`)

`has_error, mistake_family, mistake_subtype, description, expected, observed,
violated_role, evidence, confidence, rationale`. Families default to the
CaptainCook4D taxonomy but stay open-string so the same schema serves assembly.

## Evaluation

`evaluate_attributions(results, ground_truth)` → coverage, **family accuracy**
(canonicalized), **description token-F1**, mean confidence. CaptainCook4D ships a
per-error `tag` + free-text `description`, so attribution is directly measurable;
its zero-shot split tests open-set families.

## Iteration roadmap (fill stages in this order)

1. **Real backend.** `ClaudeVLMBackend` structure is complete — wire frame→base64
   resolution; or `QwenVLBackend` locally on the box. (Backend swap only.)
2. **CaptainCook4D evidence builder.** A `FrameProvider` that decodes before/
   contact/effect frames around a flagged segment (reuse decord), and a
   `ProcedureContext` from the recipe step list. (Evidence swap only.)
3. **Candidate source.** Adapter from the detector/tracker (or, for offline eval,
   from CaptainCook4D error segments) → `Candidate`s. (Glue, outside core.)
4. **Metric hardening.** Add a stronger description metric (embedding similarity /
   light human eval) and a zero-shot-family report.
5. **Prompt ablations.** Zero-shot vs CoT vs in-context exemplars — measured by
   family accuracy + description-F1 on the same events.

Because each is a single-stage swap, we can land and measure them one at a time.
