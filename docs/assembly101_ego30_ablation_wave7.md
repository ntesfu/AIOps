# Assembly101 ego30 ablation wave 7

Wave 7 completes the seed-7 2x2 factorial for procedural event context and two
event-centered crops per training mistake. The four cells share the expanded
133/15 recording train/validation split and do not evaluate the held-out test.

## Selected validation results

The table uses the legacy selector under which these runs were launched, making
the four cells directly comparable.

| Run | Procedure | Crops | Epoch | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Normality AP | Incorrect-event F1 | Recall | F1 at +/-4 s | False alerts/min |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | no | no | 14 | 16.395% | 4.818 | 1.547% | 48.784% | 7.321% | 44.290% | 5.263% | 2.857% | 5.263% | 0.019 |
| G0 | yes | no | 14 | 16.884% | 4.730 | 1.774% | 44.443% | 2.934% | 46.959% | 2.041% | 2.857% | 4.082% | 0.574 |
| G1 | no | yes | 6 | 12.320% | 3.346 | 0.638% | 35.701% | 2.239% | 53.404% | 6.250% | 5.714% | 6.250% | 0.250 |
| E1 | yes | yes | 10 | 13.956% | 4.732 | 0.937% | 45.519% | 8.008% | 35.472% | 6.452% | 5.714% | 6.452% | 0.231 |

## Attributable effects

Effects are percentage-point differences except Edit and false alerts/min.

| Change at fixed counterpart | Frame accuracy | F1@50 | State macro-F1 | Incorrect-state F1 | Normality AP | Incorrect-event F1 | Recall | False alerts/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Add procedure, no crops (G0-E0) | +0.489 | +0.226 | -4.341 | -4.387 | +2.669 | -3.222 | +0.000 | +0.555 |
| Add crops, no procedure (G1-E0) | -4.075 | -0.909 | -13.083 | -5.081 | +9.114 | +0.987 | +2.857 | +0.231 |
| Add procedure with crops (E1-G1) | +1.637 | +0.299 | +9.818 | +5.769 | -17.932 | +0.202 | +0.000 | -0.019 |
| Add crops with procedure (E1-G0) | -2.928 | -0.837 | +1.076 | +5.075 | -11.487 | +4.411 | +2.857 | -0.342 |

The interaction is substantial for state metrics: +14.159 points for state
macro-F1 and +10.156 for incorrect-state F1. It is negative for normality AP
(-20.601). This means the two changes are not independently additive and should
not be described as two universally beneficial features.

## What each change contributed

- Event-centered crops are the only factor that consistently doubles strict
  mistake recall from one to two matched validation events. They improve exact
  event F1 without procedure and improve it even more relative to procedure-only.
  Their cost is poorer action segmentation at the event-oriented checkpoint.
- Procedural context alone gives a small action gain but reduces exact mistake
  F1 and incorrect-state F1. It is not justified as a standalone addition.
- Procedure plus crops recovers much of the state-quality loss and achieves the
  best exact event F1 in this factorial, but it does not recover baseline action
  quality and its normality AP is lower.

The new operational selector further reveals checkpoint conflict. For G1 it
chooses epoch 14 rather than legacy epoch 6: frame accuracy rises from 12.320% to
15.013%, state macro-F1 from 35.701% to 47.414%, exact event F1 becomes 4.651%,
and +/-4 s event F1 reaches 9.302% at 0.065 false alerts/min. This is a better
operating point, though still not a satisfactory detector.

## Decision

Retain event-centered crops as the strongest data-side mistake intervention, but
do not carry procedural context into the next clean structural ablation. Wave 8
tests whether shared any-mistake timing plus conditional component localization
can preserve the crop recall gain without forcing the action and mistake tasks to
select very different epochs. Procedure can be reconsidered only after that head
is shown to work, as a separate factor.
