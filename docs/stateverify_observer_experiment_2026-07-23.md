# StateVerify observer experiment — 2026-07-23

## Data and contract

All candidates used the same 52-recording IndustReal split and the same causal
feature cache:

- Swin3D global motion plus ConvNeXt global appearance;
- left hand, right hand, active object, and interaction-context ROI features;
- all 104 hand coordinates plus 9 headset-pose values;
- no gaze input;
- 36 training and 16 validation recordings.

The training split contains 76,990 `not_completed`, 95,384
`installed_correct`, and 1,374 `installed_incorrect` state observations. Typed
effects are substantially scarcer: 172,929 `no_change`, 308
`complete_correct`, 19 `complete_incorrect`, and 97 `remove`. Validation has
only 12 `complete_incorrect` observations.

## Results

| Candidate | Main change | Best state incorrect recall | Best effect incorrect recall | Finding |
|---|---|---:|---:|---|
| 1 — dense multiclass | Dense focal loss | 8.9% transient | 0% | Collapsed to no-change for effects |
| 2 — balanced multiclass | One rare window per batch; hard no-change mining | 5.5% | 0% | Correct/remove effects became learnable; incorrect remained unseen |
| 3 — factorized correctness | Completion/type × correctness heads | 5.5% | 0% | Correctness overfit training and did not generalize |

Candidate 2 is retained as the implementation baseline. At its best early
epoch, correct-completion recall was about 82.6%, remove recall about 10.9%,
and state incorrect recall 5.5%. It did not satisfy the mistake gate because
incorrect-effect recall remained 0/12.

## Interpretation

The experiment rules out simple optimization collapse as the only problem:
hard-negative mining and guaranteed rare batches made the effect head learn
meaningful correct and removal transitions. It also rules out output
factorization as a sufficient fix: the conditional correctness head memorized
the 19 training errors while validation correctness degraded.

The limiting factor is the combination of extremely sparse incorrect labels
and operator/recording shift. A larger supervised classifier is unlikely to
solve this reliably.

## Next architecture

The next candidate should add an execution residual trained primarily from
correct examples:

1. Build component- and step-conditioned normal prototypes from correct
   training windows.
2. Compare the current component embedding with several normal prototypes,
   using cosine/Mahalanobis energy rather than a single centroid.
3. Calibrate residual thresholds on validation at a fixed false-alerts-per-
   minute budget.
4. Fuse the residual with the existing procedure and effect residuals through
   the typed belief tracker.
5. Use the 19 incorrect training events for calibration/ranking only, not as
   the sole source of a new error representation.

This is the evidence-supported path toward open-set mistakes such as wrong
tool, wrong part, or wrong connection when exhaustive error labels are
unavailable.
