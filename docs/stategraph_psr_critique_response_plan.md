# StateGraph-PSR critique response and next research plan

## Executive decision

The critique is directionally correct: further head-capacity sweeps should stop, exact mistake
localization remains the gating failure, and the next investment should be state-centric evidence
and mistake-bearing data. It is partly out of date, however. The current repository already has
event-centered crops, frozen-action event adaptation, procedural event context, operational
checkpoint gates, and a dual-expert composition path. Those changes produced non-zero validation
mistake F1 in repeated Assembly101 ablations, although the official held-out test remains at zero.

The next architecture should therefore be an **Action Expert + Component Evidence Expert + Typed
Belief Tracker**, not a larger shared temporal head and not another unqualified any-mistake
factorization experiment.

The audit also found a result-interpretation issue not called out strongly enough in the critique:
the IndustReal v5 VideoMAEv2 path and native Swin fallback use clips centered on the prediction
timestamp, so they include future frames. The temporal head is causal, but that full feature
pipeline is not. The reported IndustReal v5 segmentation result must be described as
offline/bounded-lookahead until trailing-window features are rebuilt and evaluated.

## Assessment of the critique

| Critique | Assessment against current code/evidence | Decision |
|---|---|---|
| Capacity has not solved faults | Correct. Larger heads improve dense tasks but exact held-out mistake detection remains unsolved. | Stop XL/XXL sweeps until evidence quality clears a fault gate. |
| StateGraph is only a soft prior | Correct. `_apply_graph_filter` is a bounded first-order action prior; persistent component state is emitted by a classifier, not maintained by a typed state machine. | Add a separate causal component-belief tracker at inference. Keep the soft action prior as a regularizer. |
| One trunk serves incompatible tasks | Historically correct, now partly addressed. Event temporal blocks existed already; staged freezing and dual-expert routing now decouple optimization and inference representations. | Treat the dual-expert route as the new baseline. Do not return to joint-only training. |
| Normality path is mostly a ranker | Correct on held-out results. AP exceeds prevalence, while exact event F1 remains zero. | Use ranking as evidence, not as an alert by itself. Calibrate alerts from belief transitions. |
| Timing resampling can smear onsets | Correct, and the audit found a more serious issue: official Assembly101 clips are past-only/end-anchored, while the IndustReal native fallback currently builds center-anchored clips containing future frames. Legacy precomputed features can also be index-resampled without timestamp provenance. | Make anchor/timestamp provenance mandatory, rebuild IndustReal trailing clips, and forbid interpolation in the strict online benchmark. |
| Appearance is too global | Correct. A current-frame global ConvNeXt vector is weak evidence for a wrong pin, tool, orientation, or connection. | Add hand/object/tool ROI evidence before increasing temporal capacity. |
| Ten-term loss obscures attribution | Correct for joint training. The staged event expert reduces gradient conflict, but it still computes a broad objective and past ablations altered multiple coupled mechanisms. | Define minimal phase-specific objectives and add one term at a time. |

Two proposed remedies need qualification:

1. The repository already tested a shared any-mistake timing head plus conditional component
   localization. It collapsed without event crops and remained worse balanced with them, including
   large detection delay. Do not repeat that exact factorization. Keep component-conditioned event
   evidence and revisit hierarchy only after ROI evidence exists.
2. A full joint HMM over all components is computationally and statistically inappropriate:
   `4^N` states explode, while fault supervision is sparse. Use factorized per-component beliefs
   coupled by sparse procedure constraints and a small beam/particle set.

## Target architecture: StateGraph-PSR Evidence v2

```text
causal video + hands/pose + optional depth
      |
      +--> frozen motion encoder --------------------------+
      |                                                    |
      +--> global appearance encoder --> Action Expert ----+--> action timeline
      |                                                    |    next-step distribution
      +--> hand/object/tool ROIs --> Component Evidence ----+--> per-component emissions
                                                           |    install/remove/repair
procedure schema + action expert expectation --------------+    correct/incorrect/unknown
                                                                |
                                                                v
                                                Typed Component Belief Tracker
                                                not_completed/correct/incorrect
                                                (+ optional latent in_progress)
                                                                |
                                                                +--> immediate alert
                                                                +--> mistake type
                                                                +--> recovery state/action
```

The neural model estimates emissions. The belief tracker owns persistence, legal state changes,
alert hysteresis, and recovery transitions. This separates visual uncertainty from procedural
logic and makes every alert auditable.

## Work plan

### Phase 0 — Freeze the evidence baseline

**Goal:** establish one immutable comparison point.

- Compose and validate the current dual-expert model using the existing action and event experts.
- Freeze action/event thresholds on validation and evaluate the held-out test once.
- Record checkpoint hashes, cache identity, label taxonomy, git commit, seeds, and the full alert
  metrics.
- Treat the current 34.9M dual expert as the capacity ceiling for the next phases.

**Gate:** a reproducible baseline report exists; no test-driven threshold changes.

### Phase 1 — Timing and label integrity

**Goal:** ensure the event benchmark measures the model rather than cache alignment error.

- Store source frame indices, source timestamps, clip anchor (`end` or `center`), feature emission
  timestamp, and annotation timestamp for every cache row.
- Add an audit that fails strict-event evaluation if features were interpolated or if maximum
  timestamp error exceeds half an update stride.
- Use causal nearest-past joins for visual features; never use a future-centered feature for an
  online alert.
- Replace the IndustReal native `_clip_indices(center, ...)` extraction path with trailing clips
  ending at the prediction timestamp. Until rebuilt, report those centered caches as offline-only.
- Rebuild the VideoMAEv2/SSv2 cache with trailing 16-frame clips and compare centered versus
  trailing features to quantify the accuracy cost of removing lookahead.
- Report event metrics at exact, +/-1 s, +/-2 s, and +/-4 s, plus signed detection delay.

**Gate:** 100% of evaluated rows have traceable timestamps; synthetic onset tests recover the
expected row exactly; no source frame index exceeds the prediction frame in causal mode.

### Phase 2 — Component-centric evidence cache

**Goal:** make wrong-part/tool/connection evidence visible to the model.

- Build hand-centered and active-object ROI crops at the native current frame.
- Add compact ROI embeddings for left hand, right hand, held object/tool, target component, and
  connection region. Missing detections get explicit masks, not zero-as-valid features.
- Use AssemblyHands/Assembly101 hand supervision for pretraining and IndustReal hand/pose/CAD data
  for target adaptation.
- Preserve the global streams for action context; route ROI evidence primarily to the event expert.
- Cache detector version, crop coordinates, confidence, and feature hashes.

**Gate:** same-component correct versus incorrect installs show improved validation separability
over the global-only baseline across at least three seeds.

### Phase 3 — Event learning as component-conditioned bags

**Goal:** learn rare faults without pretending every frame is an independent labeled example.

- Construct positive bags around each incorrect/remove event.
- Construct matched hard-negative bags from correct installs of the same component, the same
  verb with a different object/tool, and near-miss procedure positions.
- Train component-conditioned contrastive/ranking evidence plus completion/outcome/state losses.
- Keep a sparse onset/localization objective inside each bag, but use bag-level supervision when
  the exact annotation is uncertain.
- Start with the existing component event head. Reconsider an any-mistake hierarchy only after ROI
  features improve timing AP; previous factorized runs are the negative control.

**Minimal event objective:**

```text
L_event = L_state + L_completion + L_outcome + L_bag_rank + L_localize
```

Normality becomes one emission among several, not a direct alert score.

**Gate:** validation mistake AP exceeds prevalence and exact event recall is positive for all three
seeds, with <=2 false alerts/minute and acceptable median delay.

### Phase 4 — Typed component belief tracker

**Goal:** turn the soft state predictions into persistent, legal, explainable state estimates.

- Begin with the three states directly supervised by IndustReal:
  `not_completed`, `installed_correct`, and `installed_incorrect`.
- Treat `in_progress` as an optional latent state inferred from AR evidence. Do not report it as
  supervised ground truth unless new annotations are collected. `removed` is an operator outcome
  that returns a component to `not_completed`, rather than an unsupported persistent label.
- Define typed operators: `start_install`, `complete_correct`, `complete_incorrect`, `remove`,
  `inspect`, and `repair`.
- Convert neural state/event/action outputs into calibrated emission likelihoods.
- Run factorized Bayesian/semi-Markov filtering per component, coupled by procedure prerequisites.
- Maintain a small beam of plausible global procedure states rather than enumerating the full
  Cartesian product.
- Emit a fault only when posterior mass transitions into `installed_incorrect` and persists for a
  configurable confirmation interval. Emit a recovery when `repair` or a correct reinstall moves
  belief back to `installed_correct`.

**Gate:** compared with thresholded onset logits, the tracker reduces false alerts and improves
event F1 or delay without using test calibration.

### Phase 5 — Loss attribution and controlled ablation

**Goal:** know which parts of the system earn their complexity.

- Action phase baseline: action + refinement; test boundary, smoothing, next-action, and soft graph
  one at a time.
- Event phase baseline: the five-term minimal event objective above; test normality, procedure
  context, disagreement, and consistency one at a time.
- Use identical caches, splits, seeds, selection rules, and training budgets.
- Keep an addition only if its predeclared target metric improves across seeds without violating
  false-alert or latency gates.

### Phase 6 — Target-domain transfer

**Goal:** use Assembly101 as representation pretraining and IndustReal as the PSR/fault target.

- Pretrain action/hand-object/anticipation representations on Assembly101.
- Compare frozen transfer, parameter-efficient temporal adaptation, and full event-head fine-tune
  on IndustReal.
- Do not merge raw taxonomies; transfer via verb/object/component representations.
- Use participant-disjoint and leave-error-type-out evaluation on IndustReal.
- Until more target faults exist, prioritize PR-AUC, error recall, false alerts/hour, and delay over
  a single unstable F1 number.

### Phase 7 — Baselines and deployment gate

- Run ASFormer and DiffAct on identical caches after the evidence cache is fixed.
- Add constrained Viterbi/beam decoding for the action timeline independently of the component
  belief tracker.
- Measure decode + feature + neural + confirmation delay end to end on live video.
- Do not begin recovery recommendation learning until the tracker can reliably identify the failed
  component and resulting state.

**Deployment research gate:** positive held-out incorrect-event recall across multiple seeds,
false-alert rate within the operational limit, calibrated uncertainty, and a documented recovery
transition for every supported fault class.

## Immediate next experiments

1. Finish and report the current dual-expert composition; this closes the existing line of work.
2. Implement the strict timestamp audit and rerun event evaluation without interpolation.
3. Build one ROI pilot for wheel/pin or another component with both correct and incorrect examples.
4. Run global-only versus ROI-plus-global matched-bag training on the unchanged event expert.
5. Prototype the belief tracker offline from saved emissions before integrating it into live inference.

## Evidence sources

- Current repository reports: `docs/assembly101_stategraph_80epoch_report.md`,
  `docs/assembly101_ego30_ablation_wave10.md`, and
  `docs/assembly101_ego30_ablation_waves8_9.md`.
- EgoPER uses holistic and active-object relational features with contrastive step prototypes for
  procedural error detection: https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf
- PREGO separates online action recognition from symbolic procedure reasoning for open-set mistake
  detection: https://openaccess.thecvf.com/content/CVPR2024/papers/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.pdf
- AssemblyHands reports that improved 3D hand pose improves action recognition:
  https://openaccess.thecvf.com/content/CVPR2023/papers/Ohkawa_AssemblyHands_Towards_Egocentric_Activity_Understanding_via_3D_Hand_Pose_Estimation_CVPR_2023_paper.pdf
- Differentiable Task Graph Learning reports learned task-graph gains for online procedural mistake
  detection: https://openreview.net/forum?id=2HvgvB4aWq
