# Assembly101 ego-30 controlled ablation report

Source commit: `ee98b6c`.
The held-out test split was not evaluated in this screening wave.

## Absolute validation results

| Run | Factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min | VRAM GiB | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | expanded training actors only | 16.395 | 4.818 | 1.547 | 48.784 | 7.321 | 44.290 | 0.562 | 5.263 | 2.857 | 5.263 | 0.019 | 2.680 | 14 |
| E1 | expanded data plus procedural/event-centered training, seed 7 | 13.956 | 4.732 | 0.937 | 45.519 | 8.008 | 35.472 | 0.294 | 6.452 | 5.714 | 6.452 | 0.231 | 2.660 | 10 |
| E2 | expanded procedural candidate, seed 17 | 16.946 | 6.548 | 2.336 | 48.777 | 0.538 | 25.897 | 0.481 | 5.263 | 2.857 | 5.263 | 0.019 | 2.662 | 22 |
| E3 | expanded procedural candidate, seed 27 | 17.776 | 8.772 | 3.407 | 47.747 | 1.007 | 27.899 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 2.660 | 28 |

## Attributed deltas

A positive delta is better except for false alerts/minute, where a negative delta is better.

| Contrast | Isolated factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1−E0 | expanded data plus procedural/event-centered training, seed 7 | -2.438 | -0.085 | -0.610 | -3.265 | +0.688 | -8.818 | -0.268 | +1.188 | +2.857 | +1.188 | +0.213 |

## Data-expansion contribution

The expanded train split increases recordings from 73 to 133, actors from 8 to 16,
and labeled incorrect onsets from 62 to 174. Validation and test actors and
recordings are unchanged. Comparing the same procedural/two-crop candidate and
the same seeds against wave 4 isolates this training-data change.

| Three-seed median | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Recall | False alerts/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original train split (D0–D2) | 15.005 | 7.704 | 2.744 | 45.659 | 4.973 | 24.586 | 0.000 | 0.000 | 0.000 | 0.000 |
| Expanded train split (E1–E3) | 16.946 | 6.548 | 2.336 | 47.747 | 1.007 | 27.899 | 0.294 | 5.263 | 2.857 | 0.019 |
| Expanded − original | +1.941 | -1.156 | -0.407 | +2.087 | -3.966 | +3.312 | +0.294 | +5.263 | +2.857 | +0.019 |

Data expansion is the first intervention to make constrained incorrect-event
detection nonzero across the three-seed median. It also improves median frame
accuracy, state macro-F1, and mistake AP. It does not improve every metric:
median Edit, F1@50, and incorrect-state F1 regress, so additional data alone is
not the final architecture.

## Stability and interpretation

- E1 and E2 select nonzero operational event checkpoints; E3's aggregate
  selection score chooses epoch 28 with zero event recall, although seed 27 did
  produce a nonzero constrained checkpoint at epoch 8. Sparse-event selection
  remains temporally unstable.
- E1 versus E0 changes procedural context and event-centered crops together.
  It improves exact event F1 and recall at seed 7 but reduces action and state
  metrics. This contrast is intentionally treated as confounded, not as proof
  that either factor independently helps.
- Wave 7 completes the 2x2 factorial with procedure-only (G0) and crops-only
  (G1) cells. The final combination decision is deferred until those cells
  finish.
- No held-out test recording was evaluated during this selection wave.
