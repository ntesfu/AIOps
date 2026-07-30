# Assembly101 ego30 ablation waves 8 and 9

Waves 8 and 9 test whether mistake detection should be factorized into a shared
temporal `any mistake` decision and conditional component localization. All
comparisons are validation-only and use the operational checkpoint selector.

## Results

| Run | Factorized | Crops | Any-positive weight | Selected epoch | Frame accuracy | Edit | F1@50 | State macro-F1 | Normality AP | Exact event F1 | Recall | False alerts/min |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | no | no | — | 14 | 16.395% | 4.818 | 1.547% | 48.784% | 44.290% | 5.263% | 2.857% | 0.019 |
| H0 | yes | no | 1 | none | — | — | — | — | — | 0.000% at every epoch | 0.000% | 0.000 |
| G1 | no | yes | — | 14 | 15.013% | 4.961 | 1.581% | 47.414% | 25.873% | 4.651% | 2.857% | 0.065 |
| H1 | yes | yes | 1 | 14 | 14.162% | 5.589 | 1.154% | 40.376% | 35.455% | 2.353% | 5.714% | 1.231 |
| H2 | yes | yes | 50 | 20 | 15.227% | 7.024 | 2.096% | 43.268% | 29.423% | 4.167% | 2.857% | 0.111 |

H0 completed all 32 epochs but deliberately exported no promoted checkpoint:
every calibrated epoch had zero strict mistake recall, so the operational gate
rejected the run instead of treating silence as success.

## Timing versus localization diagnostics

The current decoder was rerun on H1's exported validation checkpoint after the
diagnostics were added. H2 recorded them during training.

| Run | Any-mistake frame AP | Component top-1 at true onset | Exact event F1 | Mean matched delay |
|---|---:|---:|---:|---:|
| H1 | 0.480% | 8.571% (3/35) | 2.326% | 6.56 s |
| H2 | 1.088% | 11.429% (4/35) | 4.167% | 88.13 s |

The positive weight does strengthen the temporal head and modestly improves
component localization, but it does not produce a deployable detector. H2 loses
half the strict recall of H1, and its matched detections occur far too late.
Most importantly, both factorized crop runs remain worse balanced than the
non-factorized staged models in wave 10.

## Attribution and decision

- Factorization without event-centered crops causes complete temporal collapse.
- Crops make factorization trainable, but H1 regresses exact event F1 by 2.298
  points and state macro-F1 by 7.038 points versus crops-only G1.
- Increasing the any-positive weight from 1 to 50 improves H2 over H1 on action
  metrics and exact event F1, but decreases strict recall and creates an 88-second
  matched delay. The new head is learning a weak timing signal, not a usable
  onset detector.

Reject factorization for the final 80-epoch model. Retain its implementation as
an opt-in research path and its separate diagnostics, but use the legacy
component event head with staged crop-centered adaptation, which has already
produced materially higher exact event F1 at much lower false-alert rates.
