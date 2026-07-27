"""Step-aware checkpoint selection (Track A).

The legacy selection score was built from fine metrics + a false-alert penalty and
selected a near-untrained early epoch on the Assembly101 SSv2 run (fine frame-acc
~1.6%). When step-head metrics are present, selection must be step-dominant.
"""

from __future__ import annotations

from aiops.training.train_stategraph_psr import _validation_selection_score as score


def _base() -> dict:
    return {
        "f1@50": 0.0, "edit": 0.0, "frame_accuracy": 0.0, "state_macro_f1": 0.0,
        "state_incorrect_f1": 0.0, "incorrect_event_f1": 0.0,
        "normality_incorrect_average_precision": 0.0,
        "incorrect_false_alerts_per_minute": 0.0,
    }


def test_step_dominant_selection_prefers_trained_epoch():
    # Near-untrained epoch 1: majority-blob step acc but poor step segmentation +
    # high state_macro + zero false alerts (which spuriously won under the old score).
    ep1 = {**_base(), "frame_accuracy": 1.61, "f1@50": 1.61, "edit": 4.2,
           "state_macro_f1": 42.82, "normality_incorrect_average_precision": 30.0,
           "step_frame_accuracy": 44.91, "step_online_edit": 29.30, "step_online_f1@50": 7.66}
    ep30 = {**_base(), "frame_accuracy": 17.54, "f1@50": 4.93, "edit": 7.25,
            "state_macro_f1": 42.73, "normality_incorrect_average_precision": 25.0,
            "incorrect_false_alerts_per_minute": 6.0,
            "step_frame_accuracy": 53.29, "step_online_edit": 29.99, "step_online_f1@50": 19.58}
    assert score(ep30) > score(ep1)


def test_falls_back_to_fine_metrics_without_step_head():
    # No step_* keys -> original fine-metric formula still computes a finite score.
    m = {**_base(), "f1@50": 18.0, "edit": 34.0, "frame_accuracy": 33.0,
         "state_macro_f1": 40.0, "normality_incorrect_average_precision": 10.0,
         "incorrect_false_alerts_per_minute": 1.0}
    s = score(m)
    assert isinstance(s, float) and s > 0
