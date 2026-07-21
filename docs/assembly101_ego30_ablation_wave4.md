# Assembly101 ego-30 controlled ablation report

Source commit: `16a034b`.
The held-out test split was not evaluated in this screening wave.

## Absolute validation results

| Run | Factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min | VRAM GiB | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | procedural two-crop candidate, seed 7 | 15.005 | 8.078 | 2.920 | 45.659 | 3.023 | 24.727 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.586 | 32 |
| D1 | procedural two-crop candidate, seed 17 | 12.903 | 5.292 | 1.204 | 44.682 | 4.980 | 24.586 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 2.586 | 20 |
| D2 | procedural two-crop candidate, seed 27 | 15.486 | 7.704 | 2.744 | 47.141 | 4.973 | 21.798 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.588 | 30 |
| D3 | hard-negative candidate under constrained selection | 14.696 | 6.732 | 2.458 | 45.111 | 2.283 | 28.293 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 2.588 | 26 |

## Attributed deltas

A positive delta is better except for false alerts/minute, where a negative delta is better.

| Contrast | Isolated factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
