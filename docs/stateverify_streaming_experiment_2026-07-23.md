# StateVerify streaming experiment — 2026-07-23

## Why this experiment was necessary

The normal-prototype experiment showed 43.40% average precision and 40% recall
on incorrect completion rows. That result was useful, but it was not an online
result: it scored only annotated completion rows and selected the prototype bank
with the ground-truth step. A live assembly alert system must instead decide
when an event has happened, use only past and current evidence, and count false
alerts over the complete video.

This experiment closes that gap. It replays all 52 IndustReal recordings,
fits prototypes and thresholds on the training subjects only, and evaluates
unchanged settings on complete validation timelines.

## Implemented online path

1. The gaze-free StateVerify observer produces causal persistent-state,
   typed-effect, and component-feature timelines.
2. The completion candidate is
   `P(complete_correct) + P(complete_incorrect)`.
3. Each component feature is compared with a bank fitted only on correct
   training states.
4. Procedure, execution, and effect residuals enter the factorized typed
   belief tracker.
5. Incorrect-state transitions are event-gated: noisy state logits cannot
   start an incorrect installation unless the causal effect head proposes a
   completion. The tracker still owns persistence, confirmation, recovery,
   and attribution.
6. Configuration is selected on training recordings under a false-alert
   budget and then frozen for validation.

The online component-only path never consumes the ground-truth action label.
The predicted-step ablation uses a causal action model and a zero-filled legacy
gaze slot; no gaze measurement is reintroduced.

## Results

| Experiment | Train recall | Train precision | Train false alerts/min | Validation recall | Validation false alerts/min |
|---|---:|---:|---:|---:|---:|
| Ungated component-only tracker | 71.43% | 10.00% | 0.684 | 0/5 | 0.173 |
| Ungated predicted-step routing | 78.57% | 8.80% | 0.866 | 0/5 | 0.394 |
| Event-gated, 0.1/min budget | 64.29% | 40.91% | **0.099** | 0/5 | **0.000** |
| Event-gated, 0.25/min budget | 85.71% | 36.36% | 0.159 | 0/5 | 0.000 |
| Event-gated, 0.5/min budget | 92.86% | 19.40% | 0.410 | 0/5 | 0.047 |

Event gating is a real architectural improvement: it turns an infeasible train
operating point into one that meets 0.1 false alerts/min. It does not, however,
solve held-out error recognition.

The prior action router achieved only 22.91% validation frame accuracy when
applied to the gaze-free StateVerify cache. Hard predicted-step routing
therefore increased false alerts and did not recover a validation error.

## Exact held-out failure

All five raw incorrect annotations pass the completion gate:

- completion probability range: 0.579–0.715;
- candidate recall at thresholds 0.1, 0.2, 0.35, and 0.5: 100%.

The failure is downstream:

- component-only anomaly scores range from -0.443 to 0.316, versus the
  train-selected threshold of 1.5;
- incorrect-effect probabilities range from 0.0005 to 0.0068;
- incorrect-state probabilities are 0.0013–0.1761.

Thus the current representation sees the erroneous executions as normal. More
threshold search or a larger belief tracker cannot repair this.

Two simultaneous typed annotations also project to the same persistent state
component and timestamp. Streaming state-transition metrics now deduplicate
that projected target because a single component tracker cannot emit two
distinct state transitions at the same instant. The original typed annotations
remain available for the later attribution head. This also motivates replacing
the inherited many-to-one event/state mapping with explicit per-assembly-part
predicates.

## Architecture decision

The next candidate is a procedural multi-task observer, not a larger visual
backbone:

- add a dense causal step head to the StateVerify temporal representation;
- turn its step distribution into a learned procedural context vector;
- condition component state/effect features on that vector;
- train step, state, effect, and transition-consistency losses jointly;
- route the normal prototype bank with the observer's predicted step;
- retain component fallback for uncertain or unseen steps.

Dense step labels provide far more supervision than the 14 raw incorrect
training annotations. The hypothesis is that procedural context will separate
“visually plausible in another step” from “correct here,” while the shared
features improve the fragile action router.

The first implementation uses a small linear step head and a learned expected
step embedding. It adds little memory and keeps all frozen video encoders and
the causal component blocks unchanged, so it remains suitable for one 24 GB
GPU.

## Acceptance gates for the next training run

The new candidate advances only if it satisfies all of the following:

1. validation step frame accuracy exceeds the current 22.91% router;
2. validation incorrect-event anomaly separation improves over the
   component-only score range;
3. at least one unique held-out incorrect state transition is detected using a
   train-selected operating point;
4. false alerts remain at or below 0.25/min for the development gate;
5. the final claim continues to use full causal timelines, never
   ground-truth-step routing.

## Reproduction artifacts

- Evaluator: `scripts/evaluate_stateverify_streaming.py`
- Prototype utilities:
  `src/aiops/evaluation/stateverify_prototypes.py`
- Online evidence and matching:
  `src/aiops/evaluation/stateverify_streaming.py`
- Event-gated result on Ubuntu:
  `runs/industreal_stateverify_streaming_curve_s7/result.json`
- Signal diagnostics:
  `runs/industreal_stateverify_streaming_diagnostics_s7/result.json`
- Predicted-step result:
  `runs/industreal_stateverify_streaming_predstep_s7/result.json`

The result directories live on
`/media/lm-ciss/LM_4TB/aiops/AIOps-stategraph-industreal`.
