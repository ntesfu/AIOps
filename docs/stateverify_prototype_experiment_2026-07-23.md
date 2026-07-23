# StateVerify normal-prototype experiment — 2026-07-23

## Motivation

The supervised StateVerify observer learned correct completion and removal but
never detected an incorrect effect on validation. There are only 14 exact
incorrect completion events in training and 5 in validation. This experiment
therefore learns the execution representation from abundant correct component
states and reserves incorrect labels for threshold calibration.

## Method

The retained balanced observer supplies causal per-component embeddings. For
each component and action step, the evaluator fits a spherical multi-prototype
normal model using correct training frames only. Sparse step groups fall back
to a component-level bank.

Anomaly score is minimum cosine distance to a normal prototype, normalized by
the training median and 90th-percentile spread for that group. Five prototypes
are initialized deterministically with farthest-point selection and refined by
spherical k-means. Threshold calibration uses the 14 training errors with a
budget of 0.1 false candidate alerts per minute.

## Validation result

The validation split contains 109 correct and 5 incorrect annotated completion
candidates over 63.3 minutes.

| Method | Incorrect AP | Precision | Recall | F1 | Candidate false alerts/min |
|---|---:|---:|---:|---:|---:|
| Supervised incorrect-effect probability | 5.21 | 0 | 0 | 0 | — |
| One normal prototype | 5.21 | 0 | 0 | 0 | 0.111 |
| Three deterministic prototypes | 7.32 | — | — | — | — |
| **Five deterministic prototypes** | **43.40** | **20.0** | **40.0** | **26.67** | **0.126** |

The five-mode residual detects 2 of 5 validation mistakes with 8 false
candidate alerts. Mean normalized anomaly is 1.37 for incorrect candidates
versus 0.54 for correct candidates. Ranking AP improves by 38.19 absolute
points over the supervised error head.

## Interpretation and limitations

This is the first candidate in the current experiment series with non-zero
held-out incorrect-effect recall. It supports the proposed execution-residual
layer and confirms that modeling several valid execution modes is materially
better than one normal centroid.

It is not yet an online alert result. Scoring is evaluated at annotated
completion candidates, so the reported false-alert rate assumes a completion
gate. The next integration must obtain those candidates from the causal effect
head, persist the fitted prototype bank, and pass residuals through the typed
belief tracker. Only that integrated evaluation can report true live-feed
false alerts/minute and detection delay.

The five validation errors also make confidence intervals very wide. The
result should be repeated across held-out operator folds after online
integration.
