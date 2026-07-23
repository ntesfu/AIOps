# Assembly101 ego-30 controlled ablation report

Source commit: `pending`.
The held-out test split was not evaluated in this screening wave.

## Absolute validation results

| Run | Factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min | VRAM GiB | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 10 hard negatives per incorrect positive | 11.838 | 4.737 | 0.752 | 41.489 | 4.876 | 34.403 | 0.008 | 0.061 | 14.286 | 0.135 | 150.197 | 2.586 | 12 |
| C1 | 20 hard negatives per incorrect positive | 10.354 | 3.919 | 0.815 | 39.769 | 4.225 | 44.494 | 0.018 | 0.355 | 2.857 | 1.418 | 4.885 | 2.588 | 6 |
| C2 | 50 hard negatives per incorrect positive | 12.813 | 4.727 | 1.121 | 41.304 | 3.421 | 36.493 | 0.008 | 0.059 | 8.571 | 0.177 | 93.695 | 2.586 | 12 |
| C3 | hard-negative objective plus learned evidence fusion | 11.004 | 5.032 | 0.934 | 38.636 | 2.410 | 35.249 | 0.011 | 0.049 | 25.714 | 0.108 | 341.409 | 2.588 | 12 |

## Attributed deltas

A positive delta is better except for false alerts/minute, where a negative delta is better.

| Contrast | Isolated factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3−C1 | hard-negative objective plus learned evidence fusion | +0.650 | +1.113 | +0.119 | -1.133 | -1.816 | -9.245 | -0.007 | -0.306 | +22.857 | -1.310 | +336.524 |
