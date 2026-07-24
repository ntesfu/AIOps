from aiops.adjudication import (
    AdjudicationResult,
    Alert,
    AllEventsTrigger,
    Candidate,
    ChainOfThoughtPromptBuilder,
    EnrichmentFuser,
    FrameRef,
    JsonResponseParser,
    MockVLMBackend,
    ScoreThresholdTrigger,
    TopKTrigger,
    build_default_pipeline,
    coerce_attribution,
    evaluate_attributions,
    validate_attribution,
)
from aiops.adjudication.evidence import DefaultEvidenceBuilder
from aiops.adjudication.types import EvidencePacket


def _cands():
    return [
        Candidate("e1", "1_7", 4, "Microwave the ramekin for 30 seconds", 0.91, "Timing Error"),
        Candidate("e2", "1_7", 1, "Pour 1 egg into the cup", 0.42, "Measurement Error"),
        Candidate("e3", "2_1", 3, "Add 1 tsp salt", 0.77, "Measurement Error"),
    ]


def test_end_to_end_mock_pipeline():
    pipe = build_default_pipeline(backend=MockVLMBackend(), trigger=AllEventsTrigger())
    results = pipe.adjudicate(_cands())
    assert len(results) == 3
    assert all(r.status == "ok" and r.attribution is not None for r in results)
    r1 = next(r for r in results if r.event_id == "e1")
    assert r1.attribution.mistake_family == "Timing Error"
    assert r1.backend == "mock" and r1.latency_ms is not None


def test_trigger_variants():
    cands = _cands()
    assert len(AllEventsTrigger().select(cands)) == 3
    assert {c.event_id for c in ScoreThresholdTrigger(0.5).select(cands)} == {"e1", "e3"}
    top = TopKTrigger(1).select(cands)
    assert len(top) == 1 and top[0].event_id == "e1"


def test_schema_validation():
    ok, errs = validate_attribution(
        {"has_error": True, "mistake_family": "Order Error", "description": "x", "confidence": 0.8})
    assert ok and not errs
    bad, errs2 = validate_attribution({"mistake_family": "x"})  # missing required
    assert not bad and any("has_error" in e for e in errs2)
    bad2, errs3 = validate_attribution(
        {"has_error": True, "mistake_family": "x", "description": "y", "confidence": 5})
    assert not bad2 and any("confidence" in e for e in errs3)


def test_parser_handles_fences_prose_and_coercion():
    pkt = EvidencePacket(candidate=_cands()[0])
    parser = JsonResponseParser()
    raw = 'Sure!\n```json\n{"has_error": true, "mistake_family": "Timing Error", ' \
          '"description": "microwaved too long", "confidence": 1.5}\n```\nDone.'
    res = parser.parse(raw, pkt)
    assert res.status == "ok"
    assert res.attribution.confidence == 1.0  # clamped
    assert res.attribution.mistake_family == "Timing Error"


def test_parser_rejects_non_json():
    pkt = EvidencePacket(candidate=_cands()[0])
    res = JsonResponseParser().parse("no json here", pkt)
    assert res.status == "schema_invalid" and res.attribution is None


def test_parser_strict_mode_rejects_bad_schema():
    pkt = EvidencePacket(candidate=_cands()[0])
    res = JsonResponseParser(strict=True).parse('{"mistake_family": "x"}', pkt)
    assert res.status == "schema_invalid"


def test_coerce_defaults():
    attr = coerce_attribution({"mistake_family": "Weird", "description": "d"})
    assert attr.has_error is True and attr.confidence == 0.0 and attr.evidence == ""


def test_fusion_never_suppresses_alert():
    alert = Alert("e1", "1_7", 4, 0.91, "Timing Error")
    # Even a VLM 'no error' verdict must not drop the alert.
    res = AdjudicationResult(
        event_id="e1", status="ok",
        attribution=coerce_attribution(
            {"has_error": False, "mistake_family": "Other", "description": "looks fine",
             "confidence": 0.9}))
    enriched = EnrichmentFuser().fuse(alert, res)
    assert enriched.alert is alert  # alert unchanged / still present
    assert enriched.adjudicated and enriched.disagreement is True


def test_fusion_missing_or_failed_result():
    alert = Alert("e9", "1_7", 4, 0.6)
    assert EnrichmentFuser().fuse(alert, None).adjudicated is False
    err = AdjudicationResult(event_id="e9", status="backend_error")
    assert EnrichmentFuser().fuse(alert, err).adjudicated is False


def test_backend_error_isolated_per_event():
    class Boom:
        name = "boom"

        def generate(self, prompt):
            raise RuntimeError("model down")

    pipe = build_default_pipeline(backend=Boom(), trigger=AllEventsTrigger())
    results = pipe.adjudicate(_cands())
    assert len(results) == 3
    assert all(r.status == "backend_error" and "model down" in r.error for r in results)


def test_evaluation_family_and_description():
    pipe = build_default_pipeline(backend=MockVLMBackend(), trigger=AllEventsTrigger())
    results = pipe.adjudicate(_cands())
    gt = {
        "e1": {"family": "Timing Error", "description": "microwaved 60s instead of 30s"},
        "e2": {"family": "Measurement Error", "description": "poured two eggs"},
        "e3": {"family": "Order Error", "description": "salt added out of order"},
    }
    scores = evaluate_attributions(results, gt)
    assert scores.n == 3 and scores.coverage == 1.0
    # mock echoes detector_category: e1,e2 families match; e3 detector said Measurement != gt Order
    assert 0.0 <= scores.family_accuracy <= 1.0
    assert scores.family_accuracy > 0.5


def test_pluggability_swap_prompt_and_evidence():
    frames = [FrameRef("f0", "before"), FrameRef("f1", "effect")]
    ev = DefaultEvidenceBuilder(frame_provider=lambda c: frames)
    pipe = build_default_pipeline(
        backend=MockVLMBackend(), trigger=AllEventsTrigger(),
        evidence_builder=ev, prompt_builder=ChainOfThoughtPromptBuilder())
    results = pipe.adjudicate(_cands()[:1])
    assert results[0].status == "ok"
    # evidence frames flowed through to the mock's 'evidence' field
    assert "before" in results[0].attribution.evidence
