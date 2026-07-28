"""Track A step-recognition groundwork — taxonomy, aggregation, metrics, Viterbi.

All local/no-GPU: derives the step taxonomy from the real IndustReal schema and
exercises the fine->step aggregation baseline, step metrics, and Viterbi decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiops.recognition import (
    StepTaxonomy,
    aggregate_and_score,
    build_transition_matrix,
    densify_completion_to_steps,
    frame_accuracy,
    map_via_lut,
    marginalize_to_steps,
    step_level_report,
    step_lut_from_component_indices,
    step_scores,
    viterbi_decode,
    viterbi_decode_fixed_lag,
)
from aiops.evaluation.temporal_metrics import edit_score

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "procedure_schemas"
    / "industreal_v1.json"
)


@pytest.fixture(scope="module")
def taxonomy() -> StepTaxonomy:
    return StepTaxonomy.from_schema_path(str(SCHEMA_PATH))


def test_taxonomy_shape(taxonomy: StepTaxonomy):
    # 10 completion components + background = 11 steps; none + 3 outcomes = 4 types.
    assert taxonomy.num_steps == 11
    assert taxonomy.num_types == 4
    assert taxonomy.step_names[0] == "background"
    assert taxonomy.type_names == ("none", "correct", "incorrect", "remove")


def test_step_and_type_mapping(taxonomy: StepTaxonomy):
    # Base state raw ids (0..2) are background/none.
    for raw in (0, 1, 2):
        assert taxonomy.step_of(raw) == 0
        assert taxonomy.type_of(raw) == 0
    # First component (front_chassis) owns raw 3,4,5 -> step 1, types correct/incorrect/remove.
    assert taxonomy.step_of(3) == 1 and taxonomy.type_of(3) == 1
    assert taxonomy.step_of(4) == 1 and taxonomy.type_of(4) == 2
    assert taxonomy.step_of(5) == 1 and taxonomy.type_of(5) == 3
    # Last component (rear_wheel_assembly) owns raw 30,31,32 -> step 10.
    assert taxonomy.step_of(30) == 10 and taxonomy.step_of(32) == 10


def test_step_equals_raw_floordiv_three(taxonomy: StepTaxonomy):
    # IndustReal's contiguous-triple layout makes step == raw // 3 and type == raw % 3;
    # we derive from the schema, so this cross-checks the derivation.
    for raw in range(3, 33):
        assert taxonomy.step_of(raw) == raw // 3
        assert taxonomy.type_of(raw) == raw % 3 + 1


def test_steps_from_raw_preserves_ignore(taxonomy: StepTaxonomy):
    raw = [0, 3, 3, 6, -100, 6, 0]
    steps = taxonomy.steps_from_raw(raw, ignore_index=-100)
    assert steps.tolist() == [0, 1, 1, 2, -100, 2, 0]


def test_frame_accuracy_ignores_index():
    pred = [1, 2, 3, 4]
    tgt = [1, 2, 9, -100]  # frame 2 wrong, frame 3 ignored
    assert frame_accuracy(pred, tgt, ignore_index=-100) == pytest.approx(200.0 / 3)


def test_step_scores_perfect():
    seq = [0, 0, 1, 1, 2, 2, 0]
    scores = step_scores(seq, seq)
    assert scores["frame_acc"] == 100.0
    assert scores["edit"] == 100.0
    assert scores["f1@50"] == 100.0


def test_aggregate_and_score_via_raw_lut(taxonomy: StepTaxonomy):
    lut = taxonomy.raw_to_step_lut()  # raw-id -> step
    fine_gt = [0, 3, 3, 6, 6, 9]      # steps 0,1,1,2,2,3
    fine_pred = [0, 3, 4, 6, 6, 12]   # 4->step1 (right step), 12->step4 (wrong)
    scores = aggregate_and_score(fine_pred, fine_gt, lut)
    # predicted steps [0,1,1,2,2,4] vs [0,1,1,2,2,3] -> 5/6 frames right
    assert scores["frame_acc"] == pytest.approx(500.0 / 6)


def test_step_lut_from_component_indices():
    # Model config path: action_event_component_indices[fine] = component or -1.
    # fine actions: 0->bg(-1), 1->comp0, 2->comp0, 3->comp1, 4->bg(-1)
    comp_idx = [-1, 0, 0, 1, -1]
    lut = step_lut_from_component_indices(comp_idx)
    assert lut.tolist() == [0, 1, 1, 2, 0]
    # aggregate a fine timeline through it
    fine_gt = [0, 1, 1, 3, 3]     # steps 0,1,1,2,2
    fine_pred = [0, 1, 2, 3, 4]   # 2->comp0 (step1, right), 4->bg (step0, wrong)
    scores = aggregate_and_score(fine_pred, fine_gt, lut)
    assert scores["frame_acc"] == pytest.approx(400.0 / 5)


def test_transition_matrix_is_normalized_and_masks():
    log_t = build_transition_matrix(4, self_bias=2.0)
    # each row is a proper log-distribution (sums to 1 in prob space)
    row_sums = np.exp(log_t).sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)
    # diagonal (stay) is the most likely transition given the self bias
    for i in range(4):
        assert np.argmax(log_t[i]) == i
    # forbidden transitions are effectively -inf
    forbidden = np.zeros((4, 4), dtype=bool)
    forbidden[1, 3] = True
    masked = build_transition_matrix(4, self_bias=1.0, forbidden=forbidden)
    assert masked[1, 3] < -50


def test_transition_matrix_applies_finite_soft_penalty():
    penalties = np.zeros((3, 3), dtype=np.float64)
    penalties[1, 2] = -4.0
    log_t = build_transition_matrix(
        3, self_bias=0.0, transition_penalty=penalties
    )
    assert np.isfinite(log_t[1, 2])
    assert log_t[1, 2] < log_t[1, 0] - 3.9
    assert np.exp(log_t).sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_viterbi_recovers_argmax_without_prior():
    # Zero transition prior (self_bias 0, uniform) -> Viterbi == per-frame argmax.
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=12)
    log_e = np.log(probs + 1e-8)
    flat = build_transition_matrix(5, self_bias=0.0)
    path = viterbi_decode(log_e, flat)
    assert path.tolist() == np.argmax(log_e, axis=1).tolist()


def test_viterbi_smooths_fragmented_sequence():
    # True step sequence is two clean blocks; emissions have a 1-frame spike of noise.
    T, S = 20, 3
    true = np.array([1] * 10 + [2] * 10)
    probs = np.full((T, S), 0.05)
    probs[np.arange(T), true] = 0.9
    probs[5] = [0.05, 0.1, 0.85]  # a spurious spike toward step 2 inside block 1
    probs /= probs.sum(axis=1, keepdims=True)
    log_e = np.log(probs)

    greedy = np.argmax(log_e, axis=1)
    sticky = build_transition_matrix(S, self_bias=3.0)
    smoothed = viterbi_decode(log_e, sticky)

    # Viterbi removes the spurious flip -> better (higher) Edit score, fewer segments.
    assert edit_score(smoothed, true) > edit_score(greedy, true)
    assert smoothed[5] == 1  # the spike is corrected


def test_viterbi_respects_forbidden_mask():
    T, S = 6, 3
    probs = np.full((T, S), 0.1)
    probs[:3, 1] = 0.8  # first half wants step 1
    probs[3:, 2] = 0.8  # second half wants step 2
    probs /= probs.sum(axis=1, keepdims=True)
    log_e = np.log(probs)
    forbidden = np.zeros((S, S), dtype=bool)
    forbidden[1, 2] = True  # 1 -> 2 illegal; must route via 0
    log_t = build_transition_matrix(S, self_bias=1.0, forbidden=forbidden)
    path = viterbi_decode(log_e, log_t)
    # No direct 1->2 transition anywhere in the decoded path.
    assert not any(a == 1 and b == 2 for a, b in zip(path[:-1], path[1:]))


def test_viterbi_empty():
    log_t = build_transition_matrix(3)
    assert viterbi_decode(np.empty((0, 3)), log_t).shape == (0,)


def test_fixed_lag_matches_offline_at_large_lag():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=15)
    log_e = np.log(probs + 1e-8)
    log_t = build_transition_matrix(4, self_bias=2.5)
    offline = viterbi_decode(log_e, log_t)
    # lag >= T-1 -> full offline decode
    assert viterbi_decode_fixed_lag(log_e, log_t, lag=100).tolist() == offline.tolist()


def test_fixed_lag_zero_is_causal_forward():
    # lag=0 commits each frame from the forward DP with no lookahead; still a valid
    # path and generally not worse than raw argmax on a sticky sequence.
    T, S = 16, 3
    true = np.array([1] * 8 + [2] * 8)
    probs = np.full((T, S), 0.05)
    probs[np.arange(T), true] = 0.9
    probs[7] = [0.05, 0.1, 0.85]  # spurious spike
    probs /= probs.sum(axis=1, keepdims=True)
    log_e = np.log(probs)
    log_t = build_transition_matrix(S, self_bias=3.0)
    online0 = viterbi_decode_fixed_lag(log_e, log_t, lag=0)
    assert online0.shape == (T,)
    # a few frames of lookahead should be at least as good as no lookahead (Edit)
    online3 = viterbi_decode_fixed_lag(log_e, log_t, lag=3)
    assert edit_score(online3, true) >= edit_score(online0, true)


def test_step_level_report_online_present_with_lag():
    lut = list(range(3))
    target = [1] * 8 + [2] * 8
    pred = [1] * 4 + [2] + [1] * 3 + [2] * 8
    report = step_level_report(pred, target, lut, num_steps=3, self_bias=3.0, lag=2)
    assert "online" in report and "viterbi" in report and "agg" in report
    # offline viterbi is an upper bound on the fixed-lag online decode's Edit here
    assert report["viterbi"]["edit"] >= report["online"]["edit"] - 1e-9


def test_step_level_report_causal_is_explicit_secondary_mode():
    lut = list(range(3))
    target = [1] * 8 + [2] * 8
    pred = [1] * 4 + [2] + [1] * 3 + [2] * 8
    report = step_level_report(
        pred,
        target,
        lut,
        num_steps=3,
        self_bias=3.0,
        lag=3,
        include_causal=True,
    )
    assert set(report) == {"agg", "viterbi", "causal", "online"}
    assert report["causal"]["frame_acc"] >= 0.0


def test_densify_completion_run_up():
    # 3 components, completions at frames 2 (c0), 5 (c1), 8 (c2), T=11.
    T = 11
    completion = np.zeros((T, 3))
    completion[2, 0] = 1
    completion[5, 1] = 1
    completion[8, 2] = 1
    steps = densify_completion_to_steps(completion)
    # run-up to c0 (frames 0..2) -> step 1; (3..5) -> step 2; (6..8) -> step 3;
    # after last completion (9,10) -> background 0.
    assert steps.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0]


def test_densify_completion_no_events_all_background():
    steps = densify_completion_to_steps(np.zeros((5, 4)))
    assert steps.tolist() == [0, 0, 0, 0, 0]


def test_densify_completion_ties_take_highest():
    completion = np.zeros((3, 3))
    completion[1, 0] = 1
    completion[1, 2] = 1  # two completions same frame -> take highest-indexed (c2)
    steps = densify_completion_to_steps(completion)
    assert steps.tolist() == [3, 3, 0]


def test_map_via_lut_treats_negatives_as_ignore():
    lut = [0, 1, 2, 3]
    # -1 (invalid/padding) and the explicit ignore both become ignore_index
    out = map_via_lut([0, 1, -1, 2, -100], lut, ignore_index=-100)
    assert out.tolist() == [0, 1, -100, 2, -100]


def test_marginalize_to_steps_sums_fine_mass():
    # 4 fine classes -> steps via lut; step posteriors sum constituent fine mass.
    lut = [0, 1, 1, 2]  # fine 1 and 2 both map to step 1
    fine = np.array([[0.1, 0.3, 0.4, 0.2], [0.7, 0.1, 0.1, 0.1]])
    step_post = marginalize_to_steps(fine, lut, num_steps=3)
    assert step_post.shape == (2, 3)
    assert step_post[0].tolist() == pytest.approx([0.1, 0.7, 0.2])
    assert step_post[1].tolist() == pytest.approx([0.7, 0.2, 0.1])


def test_step_level_report_agg_and_viterbi_lift():
    # Fine timeline with a 1-frame spurious flip; step lut is identity-ish.
    lut = list(range(3))  # fine==step, 3 classes
    fine_target = [1] * 8 + [2] * 8
    fine_pred = [1] * 4 + [2] + [1] * 3 + [2] * 8  # one spurious 2 at index 4
    report = step_level_report(fine_pred, fine_target, lut, num_steps=3, self_bias=3.0)
    assert set(report) == {"agg", "viterbi"}
    # agg reproduces the raw argmax step score; viterbi smooths the flip -> >= edit
    assert report["viterbi"]["edit"] >= report["agg"]["edit"]
    assert report["agg"]["frame_acc"] == pytest.approx(100.0 * 15 / 16)


def test_step_level_report_with_posteriors():
    lut = [0, 1, 1, 2]  # 4 fine -> 3 steps
    T = 10
    fine_target = [1] * 5 + [3] * 5           # steps: 1,1,1,1,1,2,2,2,2,2
    # confident-ish fine posteriors matching the target, one noisy frame
    post = np.full((T, 4), 0.02)
    for t in range(5):
        post[t, 1] = 0.94
    for t in range(5, 10):
        post[t, 3] = 0.94
    post[2] = [0.02, 0.02, 0.02, 0.94]  # noisy frame -> fine 3 (step 2) mid-block-1
    post /= post.sum(axis=1, keepdims=True)
    fine_pred = np.argmax(post, axis=1).tolist()
    report = step_level_report(
        fine_pred, fine_target, lut, num_steps=3, fine_posteriors=post, self_bias=3.0
    )
    # soft-posterior Viterbi corrects the single noisy frame -> perfect-ish edit
    assert report["viterbi"]["edit"] >= report["agg"]["edit"]
