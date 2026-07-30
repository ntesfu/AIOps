# Error-first execution-error detection — consolidated findings

**Scope.** This document consolidates the error-first line of work across the two
datasets it has touched — **IndustReal** (egocentric HoloLens assembly) and
**CaptainCook4D** (egocentric cooking) — into one verdict on the architecture and
a concrete path forward. It is the executive companion to the blow-by-blow lab
notebook ([error_first_research_journal.md](error_first_research_journal.md)) and
the visual scorecard (architecture diagram + all metrics).

---

## 1. Objective

A full build/assembly egocentric monitor that decides, from a live stream, whether
the **correct step** was taken, with the **correct tools/parts**, done the
**correct way** — and if not, **what the mistake was** (and eventually how to
recover). Error detection is the gating milestone; recovery builds on top.

## 2. The architecture (as it stands)

A three-part decomposition, all built and sound:

- **Action / Step Expert** (StateGraph-PSR v5 XL, 87.5M) — causal timeline over
  frozen VideoMAEv2-giant + Swin3D-S motion + ConvNeXt appearance, with a learned
  modality gate; emits the **procedure residual**.
- **Component Evidence Observer** — per-component state and typed-effect heads over
  4 ROI tokens; emits **execution** and **effect residuals**. Now trained with the
  error-first additions: a **counterfactual error engine**, an **expected-effect
  predictor** (learn normal, flag deviation), and a **same-component margin-ranking
  loss**.
- **Typed Component Belief Tracker** — a factorized causal Bayesian filter fusing
  the three residuals into persistent states and hysteretic alerts. Sound and
  reusable as-is.

Planned but not built: an **event-triggered VLM adjudicator** (names the mistake)
and a **recovery planner**. Both are gated behind working detection.

## 3. IndustReal — the blocker, exhaustively characterised

**Recognition works; the error alarm does not generalise.** The action/step expert
is healthy (frame acc ~33–35, edit ~34, F1@10/25/50 ≈ 34.9 / 29.3 / 17.9, state
acc ~88–90 on the official 16-rec val). But held-out execution-error detection
fails: `installed_incorrect` state F1 ≈ 2.5%, `complete_incorrect` effect F1 = 0%.

**The error-first additions produced the project's first real movement.**
Counterfactuals + expected-effect lifted the per-event incorrect-effect probability
**0.008 → 0.386 (~50×)** and the direct detector from **0/4 → 3/4** held-out events
(official split); the ranking loss cut first-detection false alarms ~10 → ~1.4/min.

**But it does not clear an operational gate.** On the defensible **19-event
operator-disjoint screen**, detection is **0/19 at ≤0.5 FA/min** (oracle only ~9/19
at ~23 FA/min). Two structural causes, both confirmed by experiment:

1. **Scarcity** — ~19 real error events. A 4-recording overfit gate reaches 66.7
   Incorrect F1, so the branch *can* fit faults; the official-split zero is a
   generalisation/data-support gap, not a broken model.
2. **A genuinely weak 2-D fault signal.** Operator-disjoint separability
   (correct-vs-incorrect completions) across every available source:

   | Source | AUC | | Source | AUC |
   |---|--:|---|---|--:|
   | Swin3D-S motion | **0.669** | | Grounding DINO (parts) | 0.64 |
   | ConvNeXt / ROI | 0.33–0.50 | | SAM 3 concept-seg | 0.28 |
   | VideoMAEv2-giant | 0.43–0.52 | | InternVideo2-B | **0.500** |
   | all fusions | ≤ 0.65 | | | |

   Every 2-D source is weak; the best (motion) is far from operational.

## 4. CaptainCook4D — architecture validation

To separate *scarcity* from *method*, we validated on an error-dense dataset:
**5,413 step segments, 1,683 real errors (31%), 8 categories, 8 persons** — ~88×
IndustReal's 19. Held-out AUC (the meaningful metric at this prevalence):

| Backbone | Detector | person | recordings |
|---|---|--:|--:|
| VideoMAEv2 (motion) | supervised | 0.493 | 0.534 |
| VideoMAEv2 (motion) | expected-normal (ours) | 0.561 | 0.577 |
| **InternVideo2-B (semantic)** | supervised (MLP) | 0.595 | **0.635** |
| InternVideo2-B (semantic) | expected-normal (ours) | 0.596 | 0.574 |

Three findings:

1. **Feature family is decisive.** A semantic backbone lifts the supervised
   classifier **+0.08–0.12 AUC** over motion, which is at chance for cooking errors.
2. **Our expected-normal detector is backbone-robust and wins in the weak-feature
   regime** — 0.56–0.60 regardless of backbone, and it *beats* the supervised head
   with weak features (0.561 vs 0.493). That is exactly the IndustReal regime.
3. **Data-limited, not method-limited — with a caveat.** Abundant errors reach the
   achievable band (~0.635, ≈ the paper's Omnivore baseline), but that band is
   itself modest: procedural error detection is **intrinsically hard** even at
   scale. A V2 attention head did not beat mean-pool, so the ceiling is
   feature-bound, not head-bound.

## 5. The cross-dataset synthesis (the key insight)

Pointing the CaptainCook4D-winning backbone at IndustReal is at **chance (0.500)**,
and fusing it with motion *hurts* (0.592 < 0.647). So the two datasets fail/succeed
for **orthogonal reasons**:

| | error *type* | data | outcome |
|---|---|---|---|
| **CaptainCook4D** | semantic (ingredient/quantity/temp) | abundant (1,683) | learnable, ~0.635 AUC |
| **IndustReal** | spatial/geometric (pin/orientation/connection) | scarce (19) | blocked, ~chance 2-D |

The decisive lesson: **a fault you cannot see in 2-D appearance cannot be recovered
by a better 2-D encoder, however strong.** IndustReal faults are geometric
properties of the placed part; the only levers that can move them are **3-D /
orientation / pose evidence** and **more real errors**. The "try better features"
hypothesis is now closed — tested across frozen motion, appearance, ROI,
open-vocab detection (Grounding DINO), concept segmentation (SAM 3), and a SOTA
semantic video model (InternVideo2).

## 6. Verdict & what is validated

- **The StateVerify decomposition and belief tracker are sound.** Not the blocker.
- **The error-first additions (counterfactuals, expected-effect, ranking) are a
  genuine representation improvement** — categorical on IndustReal (0 → real
  signal) and, on CaptainCook4D, the expected-normal detector is robust and
  *superior* to a discriminative head exactly when features/data are weak.
- **Operational IndustReal fault detection is not achievable on the data available
  here** — spatial faults + 19 errors, definitively not a backbone problem.
- **The method works where the error is visible and data exists** (CaptainCook4D
  ~0.635, balanced per-category recall 0.76–0.81), validating the direction.

## 7. Where experimentation goes next

Detection is characterised; the informative next steps move up the stack or change
the evidence:

- **Name the mistake (VLM adjudicator).** The next un-built component in the plan
  and the natural continuation: event-triggered structured attribution of *which*
  error, evaluated against CaptainCook4D's ground-truth error descriptions, where
  detection already works. Directly serves the "what was the mistake" objective.
- **Assembly-domain scale-up (Assembly101).** Closer to the target domain than
  cooking, with more errors than IndustReal — tests the architecture on assembly
  faults with real data support.
- **3-D / orientation evidence for IndustReal** (SAM 3D or pose) — the only lever
  for the spatial-fault regime; access-gated and orientation-subset only.
- **More real IndustReal errors** — the fundamental fix for the scarcity axis.

**Recommended:** build the VLM adjudicator on CaptainCook4D (Phase 4) — it advances
the end goal, is measurable, and runs where detection is real.

---

*Sources: [error_first_research_journal.md](error_first_research_journal.md)
(Campaigns 1–8), [error_first_phase1_pilot_2026-07-24.md](error_first_phase1_pilot_2026-07-24.md),
[stategraph_psr_v5_dualmotion_report.md](stategraph_psr_v5_dualmotion_report.md).
Code committed at 5c5d66d; CaptainCook4D features and results on the remote 4 TB drive.*
