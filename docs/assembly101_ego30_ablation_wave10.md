# Assembly101 ego30 ablation wave 10

Wave 10 tests staged training as a remedy for the observed checkpoint conflict:
action segmentation matures late, while useful mistake evidence often appears
early and is then forgotten. Each staged run initializes from a mature no-crop
checkpoint, freezes the action representation, and adapts the event/state branch
for 16 epochs using two mistake-centered crops per training event. Selection is
validation-only and operationally constrained. The held-out test is untouched.

## Selected results

| Run | Seed | Procedure | Training | Epoch | Frame accuracy | Edit | F1@50 | State macro-F1 | Normality AP | Event precision | Exact event F1 | Recall | False alerts/min | Matched delay |
|---|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 7 | yes | joint crops | 10 | 13.956% | 4.732 | 0.937% | 45.519% | 35.472% | 7.407% | 6.452% | 5.714% | 0.231 | 5.16 s |
| I0 | 7 | yes | staged crops | 10 | 16.900% | 4.760 | 1.780% | 43.399% | 42.934% | 13.333% | 8.000% | 5.714% | 0.120 | 4.98 s |
| G1 | 7 | no | joint crops | 14 | 15.013% | 4.961 | 1.581% | 47.414% | 25.873% | 12.500% | 4.651% | 2.857% | 0.065 | 11.73 s |
| I1 | 7 | no | staged crops | 1 | 16.423% | 4.831 | 1.551% | 44.824% | 42.435% | 20.000% | 5.000% | 2.857% | 0.037 | 45.73 s |
| I2 | 17 | no | staged crops | 6 | 18.303% | 7.833 | 2.783% | 44.703% | 31.667% | 8.333% | 6.780% | 5.714% | 0.204 | 0.93 s |

## What staging contributed

At fixed procedure-plus-crop architecture, I0 versus E1 provides the cleanest
comparison. Staging changes no inference parameters; it changes which parameters
are allowed to move during event adaptation.

| I0 - E1 | Delta |
|---|---:|
| Frame accuracy | +2.944 points |
| F1@50 | +0.843 points |
| Normality AP | +7.462 points |
| Event precision | +5.926 points |
| Exact event F1 | +1.548 points |
| False alerts/min | -0.111 |
| Matched delay | -0.178 s |
| State macro-F1 | -2.120 points |
| Incorrect-state F1 | -5.209 points |

Thus staging resolves much of the action/event tradeoff and improves the actual
alert behavior, though it does not improve the dense incorrect-state label.
That remaining state-head weakness is reported rather than hidden.

The seed-17 replication is especially important. Relative to its F0 initializer,
I2 keeps frame accuracy within 0.016 points and F1@50 within 0.099 points while
raising exact event F1 from 1.923% to 6.780%, doubling recall, reducing false
alerts from 0.629 to 0.204 per minute, and retaining sub-second matched delay.
The event improvement is therefore not a seed-7 accident.

## What procedural context contributed under staging

I0 versus I1 holds seed, initialization epoch, frozen-action adaptation, and crop
sampling fixed. Procedure adds 3.000 points exact event F1, doubles recall, and
reduces matched delay by 40.76 seconds, while frame accuracy changes by only
+0.477 points. Precision falls from 20.0% to 13.3%, but false alerts remain low
at 0.120/min. Unlike joint training, procedural context is useful once it cannot
destabilize the action representation.

## Resource result and decision

I0 trains only 8.44M of 34.92M parameters during adaptation and peaks at 1.641
GiB VRAM, far below the 23 GiB constraint. The architecture is unchanged at
inference; freezing affects training only.

Select the staged procedure-plus-crops family for the final model. The final
protocol uses 64 epochs of full procedural-model pretraining selected for action
quality, followed by 16 epochs of frozen-action crop-centered event adaptation
selected with the operational mistake gate. This is one final 34.9M-parameter
model trained for 80 total epochs. Test evaluation occurs only after both stages.
