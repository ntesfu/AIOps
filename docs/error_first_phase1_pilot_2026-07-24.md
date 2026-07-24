# Error-first Phase 1 pilot — 2026-07-24

First controlled test of the counterfactual error engine + expected-effect
predictor on the IndustReal StateVerify observer. Paired seed-7 run, identical
architecture and settings, on the remote RTX 4090 box.

## Setup

- Cache: `data/processed/industreal_stateverify_v2` (36 train / 16 val, ROI
  causal-verified, motion_aux dim 3095).
- Command per arm (20 epochs, batch 4, seq 192/128, hidden 256, 6 blocks, seed 7):
  `python -m aiops.training.train_stateverify_observer ...`
- **OFF (baseline):** `--no-counterfactuals --expected-effect-weight 0`.
- **ON (treatment):** `--counterfactual-rate 0.5 --expected-effect-weight 0.5`.
- Counterfactual donors are **train-only**; the validation dataset is never
  wrapped, so every validation number below is on real held-out data.

## Key data finding (shaped the engine)

On this cache the `active_object` detection is present only ~18% of frames and
~0.3% around completion events (the part is occluded by the hands exactly when it
completes), while `interaction_context` (the crop spanning hands + part) is
present ~100%. The wrong-part evidence swap therefore keys on
`interaction_context` (10/10 components get donors, 235 total) instead of
`active_object` (only 3 components, 4 donors). This occlusion-at-completion is
itself a plausible root cause of the project-long fault-detection failure.

## Result (validation, real held-out)

| Metric | OFF | ON |
|---|---:|---:|
| best epoch (trainer selection score) | 12 | 8 |
| state accuracy | 0.780 | 0.734 |
| `installed_incorrect` state recall @ best | 0.017 | 0.094 |
| peak `installed_incorrect` state recall | 0.035 | **0.350** |
| `complete_incorrect` effect recall @ best | 0.000 | **0.667** |
| peak `complete_incorrect` effect recall | 0.000 | **0.667** |
| `complete_correct` effect recall @ best | 0.831 | 0.580 |

The `complete_incorrect` head moved from a hard zero (every prior generation) to
0.667 recall, and `installed_incorrect` state recall improved ~10× at peak. This
is the first movement on held-out fault recognition in the project.

## Caveats — not yet a pass

- **Recall only.** The observer eval reports per-class recall, not precision. The
  prior failure mode was degenerate high recall from overprediction, so this is a
  necessary but insufficient signal.
- **Small support.** Validation has 12 `complete_incorrect` frames and 711
  `installed_incorrect` frames; effect numbers are high-variance.
- **Correct-class cost.** State accuracy and `complete_correct` recall dropped;
  the trade-off must be checked against the operational gate.
- The **definitive Phase-3 gate** is the streaming evaluation
  (`scripts/evaluate_stateverify_streaming.py`): precision and false-alerts/min at
  a train-selected operating point, plus unique held-out incorrect transitions.
  This pilot does not run it yet, and does not yet route the learned
  expected-effect residual into the tracker.

## Next

1. Streaming eval of the ON checkpoint vs OFF: precision, false-alerts/min, unique
   held-out incorrect transitions detected.
2. Route the learned expected-effect residual (`expected_effect_residual`) into
   the streaming/tracker effect channel, replacing the onset proxy.
3. Repeat seeds 17/29 before any promotion; official test stays sealed.

Runs: `runs/cf_pilot_s7_off`, `runs/cf_pilot_s7_on` on the remote workstation.

## Streaming gate result (2026-07-24)

Ran `scripts/evaluate_stateverify_streaming.py` (component-only, causal, prototypes
+ thresholds calibrated on train, evaluated on val) on both checkpoints.

**Per-event signal on the identical held-out incorrect events** — the decisive
evidence:

| val event | comp | OFF incorrect-effect prob | ON incorrect-effect prob |
|---|---:|---:|---:|
| 05_assy_2_2 | 7 | 0.001 | **0.419** |
| 20_assy_3_6 | 9 | 0.026 | **0.464** |
| 24_assy_2_4 | 7 | 0.006 | **0.183** |
| 26_assy_1_5 | 10 | 0.003 | **0.445** |

Mean incorrect-effect prob at true val events: **0.008 → 0.386** (~50×); mean
incorrect-*state* prob **0.031 → 0.267** (~8.5×). This overturns the project's
standing conclusion that "the representation sees erroneous executions as normal."
**Upstream representation is fixed.**

**But streaming recall is still 0** at every calibrated operating point (and
lowering the alert-threshold grid to 0.12–0.35 only raised false alerts to
1.83/min, still 0 recall). Diagnosis of why:

1. The strong new signal is in the **effect** head (~0.42 at events); the tracker
   alerts on the **persistent-state** posterior, whose emission is the weaker
   ~0.3 and never builds past threshold under the not_completed/correct prior.
2. **Train→val generalization gap:** train incorrect-effect prob mean 0.81 vs val
   0.19–0.46, so a train-calibrated threshold overshoots the val events.
3. Only **4 unique** val incorrect events — too few to calibrate reliably.

**The bottleneck moved from representation to the tracker/calibration layer** —
a much more tractable problem the project has never reached before.

## Direct effect-onset detector (bypasses the tracker)

To separate "is the signal usable" from tracker-fusion choices, a direct detector
alerts when `P(complete_incorrect)` crosses tau (first crossing per component,
refractory 32 frames), calibrated on train, evaluated on val (4 events, 63 min):

| tau | ON recall | ON FA/min | OFF recall | OFF FA/min |
|---:|---:|---:|---:|---:|
| 0.50 | **0.75 (3/4)** | 10.2 | 0.00 | 0.49 |
| 0.60 | 0.00 | 2.4 | 0.00 | 0.05 |
| 0.70 | 0.00 | 0.13 | 0.00 | 0.00 |

- **ON detects 3/4 held-out events; OFF detects 0/4 at any threshold** (OFF never
  fires on a true event even at 24 FA/min). Categorical representation win.
- **Precision is now the bottleneck:** 75% recall costs ~10 FA/min; the model
  over-predicts `complete_incorrect` on many non-event frames.
- Train-calibrated tau → val recall 0, because train events sit at ~0.81 while val
  events sit at 0.18–0.46 (the generalization-magnitude gap).

**Summary: the representation fix is real and categorical; the remaining problem
is precision (background false onsets) + train→val calibration, not sensitivity.**

## Precision phase result (2026-07-24)

Added a same-component margin-ranking loss (incorrect > correct completions,
`StateEffectLossConfig.rank_weight`, active because counterfactuals supply the
incorrect positives) and retrained ON+rank (seed 7, 20 ep). Also built a gated
direct detector (completion gate + expected-effect residual gate + temporal
confirmation).

| | ON (pilot) | ON+rank |
|---|---:|---:|
| observer `complete_incorrect` recall @ best | 0.667 | 0.333 |
| observer `complete_correct` recall @ best | 0.58 | **0.70** |
| direct detector: FA/min for first val detection | ~10 | **~1.4** |
| val-oracle recall @ ≤2 FA/min | 0.0 | **0.25** |
| val-oracle recall @ ≤0.25 FA/min (operational) | 0.0 | 0.0 |

- **Ranking loss helped precision**: first detection at ~1.4 FA/min vs ~10, and
  correct-completion recall 0.58→0.70. Real, modest.
- **Completion gate, expected-effect residual gate, and temporal confirmation
  (K=1..8) did NOT help** reach the operational gate. The residual is high at both
  real errors and false onsets, so it does not discriminate; FPs are neither
  transient nor low-completion.
- **Operational gate still not cleared** (0.25 FA/min → 0 recall). Root cause is
  now firmly the **train→val magnitude gap** (train errors ~0.81 vs val 0.18–0.46)
  and **real-error scarcity** (14 train / 4 val).
- **Caveat: 4 val events is statistically meaningless for a gate.** The
  operator-disjoint grouped-fold screen (19 pooled events,
  `scripts/prepare_industreal_grouped_evaluation.py`) is the defensible next
  measurement before any promotion claim.

## Operator-disjoint 19-event screen (2026-07-24) — the defensible verdict

Trained ON+rank across 5 operator-disjoint folds (`experiments/grpsv_folds_s7`,
`scripts/prepare_industreal_grouped_evaluation.py`), pooled all 19 held-out
incorrect events (195 min held-out). Direct `P(complete_incorrect)` onset detector.

| operating point | pooled recall | FA/min |
|---|---:|---:|
| train-calibrated @ ≤0.25 FA/min | **0/19 (0.0)** | 0.0 |
| train-calibrated @ ≤0.5 FA/min | **0/19 (0.0)** | 0.0 |
| train-calibrated @ ≤1.0 FA/min | 1/19 (0.05) | 0.53 |
| global-tau oracle, confirm=3, tau=0.6 | 3/19 (0.16) | 0.93 |
| global-tau oracle, confirm=3, tau=0.3 | 9/19 (0.47) | 22.8 |

**Verdict: the counterfactual + ranking approach does NOT clear an operational
fault-detection gate.** At ≤0.5 FA/min, train-calibrated pooled recall is **0/19**;
even the oracle threshold reaches only ~16% recall at ~0.9 FA/min. The 4-event
official-val optimism did not survive the 19-event screen.

**Root cause is now precision, and it is severe.** Precision is ~0.1–1.6%
everywhere; at tau=0.2 the detector fires on ~7,600 background frames (nearly
every available slot). The effect head **massively over-predicts
`complete_incorrect` on background**. Prime suspect: **counterfactual rate 0.5 is
too aggressive** — it inflates the incorrect-class prior so the model predicts
"incorrect" far too readily. The representation genuinely improved at true events
(0.008→0.386, 50×), but it is not frame-level separable from background.

**Honest bottom line:** counterfactuals + ranking are a real representation
improvement but not, at these settings, an operational fault detector. The next
work is precision-first, not another sensitivity lever.

### Low-rate test refutes the rate hypothesis

Re-ran the full 5-fold 19-event screen at counterfactual rate **0.15** (vs 0.5).
It cut false-positives (5,332 vs 7,654 at tau=0.2) but cut true-positives
proportionally (oracle recall 3/19 max, and **0/19 at ≤0.5 FA/min** train-cal).
Separability did **not** improve. **The precision problem is not a rate or
calibration artifact — the frame-level fault signal is fundamentally not
separable from background on held-out operators**, at any counterfactual rate
tested. The obstacle is structural: 19 dev events, and the wrong-part evidence
channel (`active_object` present <18%, so we swap `interaction_context`, a noisy
proxy) cannot cleanly localize "which part / how it is wrong."

## Concrete next steps (precision-first)

1. **Lower counterfactual rate** (e.g. 0.1–0.2) + class-balanced/asymmetric
   negative supervision to stop background over-prediction; re-run the 19-event
   screen (the only defensible measurement).
2. Stronger `no_change` suppression and temporal-context features so background
   frames stop crossing threshold.
3. Per-component / adaptive calibration to survive the train→val magnitude gap.
4. Ultimately **more real error recordings** — 19 dev events is a hard ceiling.

## Superseded earlier next-step notes

1. **Grouped operator-disjoint eval (19 events)** with the ON+rank config — the
   only statistically defensible screen; do this before further tuning.
2. **Close train→val gap:** more/diverse counterfactuals (higher rate, hand ROIs,
   orientation perturbation), stronger regularization to stop memorizing the 14
   real train errors, and calibration that doesn't assume train-magnitude signal.
3. **More real errors** — the fundamental limiter; Hand Atlas labeler + new
   HoloLens captures.
4. Then seeds 17/29; official test stays sealed.

## Old next steps (superseded)

1. Drive the tracker's incorrect-state transition directly from the strong
   `complete_incorrect` effect evidence (test the existing
   `promote_execution_to_state_emission` / `effect_positive_threshold` knobs, now
   that the effect signal is real — expose them as CLI args first).
2. Route the **learned expected-effect residual** into the tracker effect channel
   (replacing the onset proxy) — implemented but not yet used online.
3. Close the train→val gap: more diverse counterfactuals + regularization.
4. Repeat seeds 17/29; the official test stays sealed.
