# Error-first fault detection — research journal

**Purpose.** A living lab-notebook for the execution-error-detection line of work.
Unlike the results report ([error_first_phase1_pilot_2026-07-24.md](error_first_phase1_pilot_2026-07-24.md)),
this file records the *thought process*: what question each experiment asked, why
we chose that approach, what broke, how we fixed it, what we concluded, and what
we did next. It is written to be the raw material for final documentation.

**How to read it.** Chronological "campaigns," each with **Question → Approach →
Issues & resolutions → Result → Decision**. Two appendices at the end: a
quick-reference of engineering gotchas, and the reusable methodology. Update this
file whenever an experiment finishes.

---

## 0. Project context & objective

Goal: a full build/assembly egocentric model that decides whether the *correct
step* was taken, with the *correct tools/parts*, done the *correct way*, and if
not, *what the mistake was* (and eventually how to recover). Dataset: IndustReal
(HoloLens egocentric assembly, Procedure Step Recognition *with execution
errors*). Architecture at the start of this line of work: **StateVerify** — a
causal action/step expert + a per-component state/effect observer + a factorized
causal Bayesian belief tracker.

The standing blocker inherited from every prior generation: **held-out execution
errors are not detected** (`installed_incorrect` F1 ≈ 2.5%, `complete_incorrect`
F1 = 0%). The user's steer for this line: **crack error detection first**, using
**counterfactual error generation** and an **event-triggered VLM adjudicator** as
first-class components.

## 0b. Working environment & conventions

- **Compute/data live on a remote Ubuntu box** (RTX 4090, 24 GB) reachable at
  `aiops@lmciss-aiops.tail2d0daa.ts.net:2222` (Tailscale). Repo on the 4 TB drive:
  `/media/lm-ciss/LM_4TB/aiops/AIOps-stategraph-industreal`. Conda env:
  `/home/aiops/miniconda3/envs/psr_env`. The local Windows checkout has no
  torch/GPU/data — it is for editing + orchestration only.
- **Git workflow:** the remote box is the canonical committer (that's where the
  work runs); local pulls from it. Commits are authored there and local
  fast-forwards via an SSH git remote. Nothing is pushed to GitHub without an
  explicit ask.
- **Measurement philosophy adopted early:** the official validation split has only
  **4 unique incorrect events** — statistically meaningless for a gate. We switched
  to an **operator-disjoint 5-fold screen pooling all 19 development errors** as
  the defensible measurement, and to a cheap **separability probe** (operator-
  disjoint leave-one-operator-out AUC of a feature at completion events) to test
  hypotheses before committing to full training runs.

---

## Campaign 1 — Representation fix: counterfactuals + expected-effect

**Question.** Can we make the observer's `complete_incorrect` signal actually
fire on held-out errors, given only ~14–19 real error events in training?

**Approach / reasoning.** Two coupled ideas:
- *Expected-effect predictor* — learn the *normal* effect (trained on abundant
  correct completions) and flag deviation; residual = divergence(observed,
  expected). Learning "normal" sidesteps the rare-error scarcity.
- *Counterfactual error engine* — manufacture the missing supervision by turning
  correct installs into synthetic wrong-part incorrect installs (label flip +
  persistent-state relabel + a wrong-part evidence swap), train-only so the
  held-out measurement stays honest.

**Issues & resolutions.**
- *The wrong-part swap keyed on the wrong ROI.* We first swapped the
  `active_object` ROI crop. Diagnosing donor coverage showed only **3/10**
  components had donors. Root cause: `active_object` detection is present only
  ~18% of frames and **≈0.3% around completions** (the part is occluded by the
  hands exactly when it completes), while `interaction_context` (hands+part crop)
  is present ~100%. **Resolution:** re-key the swap onto `interaction_context`
  → donor coverage 3→**10/10** components (235 donors). This occlusion-at-
  completion is itself a plausible root cause of the project-long failure.
- *Trainer wiring.* Counterfactual donors must be collected from *training*
  records only and the validation dataset never wrapped; enforced with a
  train-only leakage guard.

**Result.** On real held-out validation, per-event `P(complete_incorrect)` at the
true errors moved **0.008 → 0.386 (~50×)**; direct-detector recall 0/4 → 3/4
(official split). **First movement in the project's history on this signal.**
Baseline detects zero at any threshold.

**Decision.** The representation genuinely improved — but recall alone is not a
pass; precision/false-alarms are unmeasured. Proceed to a precision phase and a
defensible screen.

---

## Campaign 2 — Precision phase (ranking loss, gates, confirmation)

**Question.** The improved head over-fires; can we get precision at an operational
false-alarm rate (≤0.25/min)?

**Approach.** (a) A same-component margin-ranking loss (incorrect > correct
completions), made active by the counterfactual positives. (b) A gated detector:
completion gate + the learned expected-effect residual as a gate + temporal
confirmation (require the signal sustained K frames).

**Issues & resolutions.**
- *Predicted-negative reasoned in advance.* We noted the expected-effect residual
  is high at *both* real errors and false onsets (both diverge from "normal"), so
  it should not discriminate — and it didn't. Recording the prediction first kept
  us honest when the result confirmed it.

**Result.** Ranking loss helped **modestly** (first detection ~10 → ~1.4 FA/min;
correct-completion recall 0.58 → 0.70). Gates + confirmation gave **no** gain.
Operational gate still not cleared.

**Decision.** Stop adding sensitivity levers; get a statistically defensible
measurement before drawing conclusions.

---

## Campaign 3 — The defensible 19-event operator-disjoint screen

**Question.** Do the gains survive on a screen that isn't 4 noisy events?

**Approach.** Reused the existing grouped-fold generator
(`scripts/prepare_industreal_grouped_evaluation.py`) to make 5 operator-disjoint
folds; trained ON+rank per fold; pooled all 19 held-out errors; direct
`P(complete_incorrect)` onset detector.

**Result.** **0/19** at ≤0.5 FA/min (train-calibrated); oracle threshold only
~9/19 at ~23 FA/min. The 4-event optimism did **not** survive. Root cause looked
like precision: ~7,600 background fires at low thresholds.

**Issues & resolutions.**
- *Rate hypothesis, tested and refuted.* Prime suspect: counterfactual rate 0.5
  inflates the incorrect prior. Retraining at rate 0.15 cut false-positives *and*
  true-positives proportionally — separability unchanged, still 0/19 at ≤0.5
  FA/min. So it is **not** a rate/calibration artifact.

**Decision.** The problem is deeper than tuning. Two directions the user picked:
episode-level reformulation (#3) and better spatial evidence (#2).

---

## Campaign 4 — Episode-level reformulation (#3) & the real root cause

**Question.** If we only alert at *detected completions* (not every frame), does
the false-positive flood collapse?

**Approach.** Detect completion episodes (peaks of `P(cc)+P(ci)`), classify each
by the **conditional** incorrect prob `P(ci)/(P(cc)+P(ci))` — robust to a
globally inflated `P(ci)`.

**Result & the key diagnosis.** Episode-level did **not** help — and the
diagnostic explained why: the effect head predicts a completion (`cc+ci ≥ 0.5`)
at **49% of all frames**. Completions are not localized. **Root cause:** the
effect loss subsamples `no_change` negatives (`effect_no_change_ratio=4`), so the
head trains on a ~20%-completion distribution and massively over-predicts
completions at inference (reality ≈ 1%).

**Issues & resolutions.**
- *Over-correction.* Retraining with `effect_no_change_ratio=40` fixed the
  over-prediction (49% → 5.5%) **but collapsed the signal at true events too**:
  real completions fell to `comp ≈ 0.46`, *below* the top 5.5% of background
  frames. **Real completions do not stand out at any calibration.** So the
  bottleneck is evidence quality, not detection logic — which motivated #2.

---

## Campaign 5 — Evidence sweep (#2): is the fault signal even in the features?

**Question.** Before building anything, is *any* available feature able to
separate correct from incorrect completions?

**Approach.** The `cache_probe`: at every completion event, pull each already-
cached feature (motion, appearance, the 4 ROI embeddings) and measure operator-
disjoint LOO-AUC (correct vs incorrect). No training, no video decode.

**Result (decisive).** Every 2D source is weak: appearance/ROI 0.33–0.50 (at/below
chance), **only Swin motion 0.669** carries signal. This overturns "we just need
better crops": the frozen appearance is fault-uninformative. We then found the
observer cache uses the *weak* Swin motion (768-d), not the preferred
VideoMAEv2-giant — but probing VideoMAEv2-giant motion gave **0.434** (worse; it's
the UnlabeledHybrid pretrain, not the SSv2 finetune) and dual-motion 0.517. So
even swapping motion backbones doesn't help.

**Issues & resolutions.**
- *Why cache-probe instead of video decode?* An earlier attempt to re-crop from
  video for a fresh probe kept collecting only a handful of events — see the
  decode gotchas in the appendix. Realizing the cache *already stores* the ROI
  embeddings let us answer the question with a fast, decode-free numpy probe.

---

## Campaign 6 — Part/state detectors: Grounding DINO, SAM3, SAM3D

**Question.** Can a trained/foundation part detector supply fault-relevant spatial
evidence the frozen features lack? (User asked to check SAM3 specifically.)

**Reasoning about data.** IndustReal here has **no part-level labels or
CAD/synthetic** (only whole-assembly GT bbox + a state-string class; `error_state`
is defined but never used, and OD frames are on a different sampling grid than the
cache). So we cannot *train* a part detector — the only option is **open-vocabulary**
detection/segmentation prompted by part name.

**Approach.** Separability probe again, but with detector features at completion
events: (a) Grounding DINO (non-gated, available immediately); (b) SAM 3
(`facebook/sam3`, gated — user obtained access).

**Issues & resolutions.**
- *Video decode reliability.* Random `cv2` frame seeking on the HoloLens MP4s is
  unreliable; sequential reads also lost most events. **Resolution:** use
  `decord.VideoReader` random access.
- *OD ↔ cache frame-grid mismatch.* `OD_labels.json` is sampled on a different
  frame grid than the cache prediction frames, so exact-frame lookup returned
  empty. **Resolution:** nearest-OD-frame match within a tolerance.
- *transformers API drift.* Grounding DINO post-process arg was `box_threshold`
  in 4.x but `threshold` in 5.x. SAM3 needed transformers ≥ 4.66 → we upgraded to
  5.14.1 (reversible; our training code doesn't use transformers).
- *Gated weights + HF auth.* SAM3 weights are gated. Chose the secure path: the
  user ran `hf auth login` on the box themselves (token never in our transcript);
  we pass it to jobs via `HF_TOKEN=$(cat ~/.cache/huggingface/token)` while caching
  the 3.45 GB model on the big drive with `HF_HOME=data/model_cache`.

**Result.** Grounding DINO part-presence 0.64, geometry 0.49, fusion-with-motion
0.65 (no gain). **SAM3 concept-segmentation 0.28 (noise); fusion with motion
0.598 — worse than motion alone.** The best open-vocab segmenter carries no fault
signal.

**SAM 3D — blocked.** `facebook/sam-3d-objects` is *separately* gated (403; the
`facebook/sam3` grant doesn't cover it) and is a heavy custom SLAT/Gaussian-splat
pipeline (not transformers). It would address only *orientation-subset* faults and
remains unmeasurable on ~19 events.

---

## Consolidated verdict (so far)

Full evidence sweep (operator-disjoint AUC, correct-vs-incorrect completions):
ConvNeXt/ROI 0.33–0.50, VideoMAEv2 0.43–0.52, **Swin motion 0.65–0.67 (best)**,
Grounding DINO 0.64, **SAM3 0.28**, all fusions ≤ 0.65. **Every 2D evidence source
fails.**

Operational execution-error detection is **not achievable on the IndustReal data
available here** — limited by (a) **~19 real errors** and (b) a **genuinely weak
visual fault signal** no evidence extraction surfaces. This is a real finding: we
conclusively ruled out the "better features/detector" hypothesis. The
counterfactual + expected-effect + ranking work remains a genuine *representation*
improvement, and the belief-tracker decomposition is sound; the objective is
**data-limited, not method-limited**.

## Dataset landscape (next-step research)

The project's own Assembly101 run is the key prior: **140 training errors still
gave 0% held-out exact-error F1** — so the bar is high error *density*, not just a
few more events. Candidates that fit "egocentric + procedural + real errors":
**CaptainCook4D** (cooking; purpose-built, error-dense; best for an
architecture-validation test), **Assembly101** (assembly; best domain fit; already
tooled; must scale well beyond the 98-rec/140-error subset), plus EgoPER,
HoloAssist, EgoOops, and the newer IMPACT (industrial assembly). Recommendation:
validate the architecture on error-dense CaptainCook4D; scale up Assembly101 for
the assembly-domain deliverable; keep IndustReal as the sealed final benchmark.

---

## Campaign 7 — CaptainCook4D: does the architecture work when errors are abundant?

**Question.** The IndustReal verdict was *data-limited, not method-limited* — but
that is a hypothesis, not a proof, until we run the same class of detector on a
dataset where errors are *not* starved. CaptainCook4D (Peddi et al., NeurIPS 2024;
egocentric cooking, purpose-built for procedural errors) is the test: if a
supervised error detector generalizes to held-out participants here, IndustReal's
failure was scarcity (19 errors) and the architecture is vindicated; if it *still*
fails with thousands of real errors, the problem is representation/method.

**The data (established before spending any GPU).**
- 384 recordings, **8 persons, 10 environments, 24 recipes**; 94.5 h.
- **5,413 video-observable step segments, 1,683 with errors (31.1%)**; 2,574 error
  instances across 8 categories (Order 795 · Technique 502 · Preparation 410 ·
  Measurement 331 · Missing-Step 285 · Timing 177 · Temperature 66 · Other 8).
  That is ~**88× IndustReal's 19** — exactly the density we needed.
- Official **person split** (participant-disjoint, the analogue of our operator
  screen): train 917 / val 323 / **test 443 error segments**.
- **Missing-Step** errors are encoded `start=end=-1` (the step was skipped, no
  video) — a step-*absence* signal, not a visual anomaly, so they are excluded
  from the visual detector (287 segments) and flagged separately.

**Reference band (paper, Table 2).** Best vision-only baseline = **Omnivore V2**
(transformer over 1-s sub-segments): step-split **F1 55.4 / AUC 75.7**;
recording-split **F1 56 / AUC 65.3**. So even *with* 2.5K errors the task tops out
near ~55 F1 / ~65–75 AUC — hard but clearly learnable (contrast IndustReal ~0–2.5%
F1). That band is the target.

**Approach / reasoning.**
- *Features.* The paper's pre-extracted features are email-gated, so we extract our
  own — one frozen **VideoMAEv2-giant** (1408-d) clip embedding per short clip,
  K clips spread across each step segment (K scales with duration), stored both
  mean-pooled (V1-style) and per-clip (for a later V2 attention head).
- *Detectors, person-disjoint.* (a) **linear** + (b) **MLP** supervised
  normal-vs-error heads — the direct scarcity test; (c) **normal_proto** — the
  StateVerify "expected-normal" idea: per-step centroid of *correct-only* training
  segments, error score = cosine distance from the expected-normal feature (tests
  learn-normal-flag-deviation when normal data is abundant). Threshold-free AUC/AP
  are primary; F1/precision/recall at a val-tuned threshold; per-category recall.
- New repo modules: [src/aiops/data/captaincook4d.py](../src/aiops/data/captaincook4d.py)
  (+ tests), [scripts/extract_captaincook4d_features.py](../scripts/extract_captaincook4d_features.py),
  [scripts/run_captaincook4d_error_recognition.py](../scripts/run_captaincook4d_error_recognition.py).

**Issues & resolutions.**
- *Features are email-gated* (only 2D video is in the downloader). Resolution:
  extract our own VideoMAEv2 features from the GoPro 360p video (~30 GB total, on
  the 4 TB drive) — keeps the pipeline under our control and consistent with the
  IndustReal line.
- *box.com rejects HTTP HEAD* on shared static links (404) → size estimation
  failed. Resolution: ranged/streaming GET reads `content-length` (avg ~78 MB → ~30
  GB total).
- *`np.savez_compressed` auto-appends `.npz`* to a path, so the `.npz.tmp` staging
  file never existed for the atomic rename. Resolution: write through an open file
  handle.
- *The SAM3 transformers upgrade (5.14.1) broke VideoMAEv2* remote-code loading
  (`'VideoMAEv2' object has no attribute 'all_tied_weights_keys'`). SAM3 work is
  done, so we **reverted to transformers 4.57.6** (training code doesn't use
  transformers; fully reversible).
- *Home disk 99% full* (4.9 GB free). Everything — data, features, repo — lives on
  the 4 TB drive; the model cache was already there.

**Issues & resolutions (during the run).**
- *box.com throttled the download into a crash.* The vendor downloader wraps only
  the initial GET in a retry and sets **no timeout**, so a mid-stream
  `IncompleteRead`/`ChunkedEncodingError` killed the whole job at 165/384. We wrote
  a robust downloader (per-file timeout, whole-file retry, size-verified resume) →
  **384/384**.
- *VideoMAE features were near-chance and we diagnosed why.* Supervised heads on
  VideoMAEv2-giant (UnlabeledHybrid) sat at **AUC ~0.49–0.55** (sklearn LogisticRegression
  confirmed — not a training bug), and max/attention pooling didn't help. Diagnosis:
  the feature is the bottleneck, and it is the *wrong kind* — VideoMAE is
  motion-pretrained, but CaptainCook4D errors are largely **semantic** (wrong
  ingredient/quantity/temperature). So the evolve is a **feature-family** change,
  not a head change.
- *Chose a stronger semantic backbone: InternVideo2-B14* (cached, K710-finetuned,
  8-frame). It needs the official model code, so we reproduced it self-contained
  (RMSNorm blocks + QK-norm + LayerScale + Conv3d patch-embed + attention-pooling
  `clip_projector` + LayerNorm `fc_norm` + K710 head) and **verified**: 0 missing /
  0 unexpected state-dict keys, and a peaked, varied K710 head (max prob 0.12–0.52
  vs uniform 0.0014) proving the forward is correct.
- *`pkill -f <script>.sh` self-matched* the ssh command line (which contained the
  script name) and killed its own shell. Fix: don't `pkill -f` a pattern your own
  command contains.

**Result (full 384 recordings, held-out; AUC/AP are the meaningful metrics — F1 at
the val-tuned threshold collapses toward the predict-all-positive rate at ~31–39%
prevalence).**

*Same-data backbone comparison (logreg / expected-normal AUC, identical 384 recs):*

| Split | backbone | supervised AUC | expected-normal AUC |
|---|---|---|---|
| person | VideoMAEv2-giant | **0.493** (chance) | 0.561 |
| person | InternVideo2-B | **0.610** | 0.596 |
| recordings | VideoMAEv2-giant | 0.534 | 0.577 |
| recordings | InternVideo2-B | 0.608 | 0.574 |

*Full 3-detector screen on InternVideo2 features (val-tuned):* recordings-split
**MLP AUC 0.635 / AP 0.520** (≈ the paper's Omnivore recording-split AUC 0.653);
person-split linear AUC 0.604. Per-category recall (person, ~40% precision) is
**balanced across all error types**: Order 0.75, Timing 0.78, Measurement 0.74,
Temperature 0.75, Technique 0.71, Preparation 0.63 — a genuinely usable multi-class
error signal (contrast IndustReal held-out error F1 ≈ 0–2.5%).

**Three findings.**
1. *Feature family is decisive.* A semantic backbone (InternVideo2) lifts the
   supervised classifier by **+0.08–0.12 AUC** over motion (VideoMAE), which is at
   chance (0.49–0.53). Cooking errors are semantic; the IndustReal-era motion
   features were the wrong tool for this dataset.
2. *The expected-normal detector is backbone-robust and wins in the weak-feature
   regime.* Its AUC (0.56–0.60) barely moves with backbone, and with the **weak**
   VideoMAE features it **beats** the discriminative classifier (0.561 vs 0.493).
   With **strong** features the supervised head catches up/exceeds. This is direct
   support for the StateVerify "learn normal, flag deviation" design *precisely in
   the low-data / weak-feature regime that IndustReal is*.
3. *Abundant errors validate "data-limited, not method-limited" — with a caveat.*
   With 1,683 real error segments + semantic features, supervised error recognition
   reaches the achievable band (AUC ~0.61–0.635). But that band is itself modest
   (paper SOTA ~0.65–0.76): procedural error detection is **intrinsically hard**
   even with thousands of errors. So IndustReal failed from a *combination* —
   scarcity (19 errors) **and** the wrong feature type **and** intrinsic difficulty.

**Decision.** Architecture validated on the two axes that matter: (a) the method
reaches the achievable band once errors are abundant, and (b) our expected-normal
contribution is robust and *superior* exactly in the weak-feature/low-data regime
IndustReal lives in. Remaining levers to push the ceiling: a **V2 attention-pool
head** over the stored per-clip features, the **stage-2 CLIP-aligned** InternVideo2
(more semantic than the K710 stage-1), and a **counterfactual-augmentation** ablation.

**Addendum — V2 attention head (tested, no gain).** Trained a `[CLS]`+transformer
attention-pool head over the per-clip InternVideo2 sequence per segment (the paper's
stronger V2 head; `scripts/run_captaincook4d_v2_head.py`). Result: **it does not beat
V1 mean-pool** — person AUC 0.589 (vs V1 0.604), recordings AUC 0.606 / AP 0.485 (vs
V1 MLP **0.635 / 0.520**). Interpretation: on these per-segment features the
discriminative signal is roughly uniform across clips, so mean-pool already captures
it — **the head is not the bottleneck, the feature is.** The only remaining lever
that can move the ceiling is a more semantic backbone (stage-2 CLIP-aligned
InternVideo2); a bigger head cannot. Best result on the dataset stays **V1 MLP,
recordings-split AUC 0.635 / AP 0.520.**

---

## Campaign 8 — Cross-dataset control: InternVideo2 on IndustReal

**Question.** InternVideo2 lifted CaptainCook4D from chance to 0.635. If we point
the *same* backbone at IndustReal, does the wrong-pin/orientation fault signal
appear — or is IndustReal blocked for a reason a better backbone can't fix?

**Approach.** The same operator-disjoint separability probe as the evidence sweep
(Campaign 5): an 8-frame trailing clip at each of the 243 completion events,
InternVideo2-B feature, leave-one-operator-out AUC (correct vs incorrect), plus
fusion with the cached Swin motion. Directly comparable to the sweep numbers.

**Result (decisive).** InternVideo2-B **AUC = 0.500 — exactly chance.** Swin motion
0.647 (reproduces the sweep's 0.669, so the harness is sound); fusion motion+IV2
**0.592 — *below* motion alone.** The strongest semantic video backbone available
is not merely weak on IndustReal, it is at chance, and fusing it *hurts*.

**Interpretation — the two datasets fail/succeed for orthogonal reasons.**
CaptainCook4D errors are **semantic** (wrong ingredient/quantity/temperature) →
InternVideo2's action-semantic representation captures them. IndustReal errors are
**spatial/geometric** (wrong pin, orientation, connection at the *same* step) → no
2-D semantic feature sees them; the fault is a geometric property of the placed
part, not an appearance/action category. A better 2-D encoder — even a SOTA one
that works on a different error *type* — cannot manufacture a signal that is not in
the 2-D pixels.

**Decision.** Confirms, now with the strongest semantic backbone, that the
IndustReal blocker is **not** a backbone problem. The two real levers stay: (a) 3-D
/ orientation / pose evidence, and (b) more than 19 real errors. Closes the
"try better features" hypothesis for good.

---

## Appendix A — Engineering gotchas & resolutions (quick reference)

| Symptom | Cause | Resolution |
|---|---|---|
| git-bash `ssh` "Permission denied (publickey)" | git-bash's bundled ssh uses a different key/HOME | Use the **Windows** OpenSSH binary `/c/Windows/System32/OpenSSH/ssh.exe` (and `scp.exe`) from git-bash; for git-over-ssh set `GIT_SSH_COMMAND` to it |
| Local commit shows whole-file diffs vs remote | Windows CRLF vs Unix LF on tracked files | Commit on the remote (canonical) with LF; local `git reset --hard ubuntu/<branch>` to adopt it |
| Nested heredoc ate `$VAR` / `$(...)` | unquoted heredoc + ssh double-nesting expands early | Write runner scripts locally, `scp`, then `nohup` them on the box |
| `bash: syntax error near unexpected token '('` | parentheses inside a double-quoted remote `echo` | Avoid `()` in remote echo strings |
| 10-min tool timeout kills long ssh command | training/eval > 10 min in a foreground ssh | `nohup` on the box writing a `.done` marker; poll for it |
| Probe collected only a handful of events | `cv2` random seek / sequential-read `break` on EOF | `decord.VideoReader[idx]` random access |
| OD detections empty at event frames | `OD_labels.json` on a different frame-sampling grid than the cache | nearest-OD-frame match within tolerance |
| Grounding DINO `post_process(... box_threshold=)` TypeError | arg renamed `threshold` in transformers 5.x | use `threshold=` |
| `Sam3Model` ImportError | transformers 4.57 predates SAM3 | upgrade to ≥ 4.66 (we used 5.14.1) |
| `hf auth whoami` "Not logged in" but downloads work | `HF_HOME` override pointed away from the token | keep default HF_HOME for the token, or pass `HF_TOKEN` explicitly |
| Counterfactual donors 3/10 components | keyed on occluded `active_object` ROI | key on always-present `interaction_context` ROI |
| box.com download link HEAD → 404 | box shared-static links reject HEAD | ranged/streaming GET reads `content-length` |
| `np.savez_compressed(path.tmp)` then rename fails (file missing) | savez auto-appends `.npz` unless given a file handle | write through an open `wb` handle |
| VideoMAEv2 load: no attribute `all_tied_weights_keys` | transformers 5.x (installed for SAM3) broke the pinned remote-code model | revert `transformers==4.57.6` for VideoMAEv2 extraction (SAM3 done) |
| CaptainCook4D Missing-Step segment has no video | error encoded `start=end=-1` (step skipped) | exclude from the visual detector; treat as a step-absence signal |

## Appendix B — Reusable methodology

- **Separability probe.** At each completion event, extract feature X and label
  (correct=0 / incorrect=1) with operator id. Leave-one-operator-out: fit
  class-centroid directions on train operators, score held-out by cosine to
  (incorrect − correct) centroids, pool, compute AUC. Fast, decode-free when X is
  cached; the cheapest way to kill or greenlight a hypothesis before training.
- **19-event operator-disjoint screen.** 5 operator-disjoint folds, train per
  fold, pool all held-out errors, report recall at ≤0.25/0.5/1.0 FA/min
  (train-calibrated) and an oracle curve. The only defensible gate; the 4-event
  official val is not.
- **Remote run pattern.** Local edit → `scp` script → `nohup ... > out; echo DONE
  > done` on the box → poll the `.done` marker. Long training as background jobs
  that survive SSH drops.
- **Commit pattern.** Edit locally → `scp` doc + message → `git add`/`commit -F` on
  the remote (canonical) → local `git fetch ubuntu && git reset --hard`.
