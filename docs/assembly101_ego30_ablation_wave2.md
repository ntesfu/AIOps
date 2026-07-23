# Assembly101 ego-30 controlled ablation report

Source commit: `pending`.
The held-out test split was not evaluated in this screening wave.

## Absolute validation results

| Run | Factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min | VRAM GiB | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | procedure context without event-centered crops | 11.107 | 3.681 | 0.667 | 33.685 | 1.890 | 34.000 | 0.008 | 0.089 | 5.714 | 0.178 | 41.338 | 2.547 | 12 |
| B1 | one event-centered crop per mistake | 9.445 | 3.007 | 0.555 | 38.889 | 1.909 | 31.139 | 0.011 | 0.083 | 11.429 | 0.208 | 88.570 | 2.463 | 6 |
| B2 | two-crop procedural candidate, seed 17 | 10.288 | 3.966 | 0.963 | 36.061 | 4.043 | 23.245 | 0.007 | 0.070 | 2.857 | 0.070 | 26.266 | 2.586 | 8 |
| B3 | two-crop procedural candidate, seed 27 | 10.506 | 5.199 | 0.642 | 37.529 | 2.448 | 29.100 | 0.005 | 0.044 | 14.286 | 0.089 | 208.615 | 2.586 | 12 |

## Attributed deltas

A positive delta is better except for false alerts/minute, where a negative delta is better.

| Contrast | Isolated factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1−B0 | one event-centered crop per mistake | -1.661 | -0.674 | -0.112 | +5.204 | +0.019 | -2.861 | +0.003 | -0.006 | +5.714 | +0.030 | +47.232 |
