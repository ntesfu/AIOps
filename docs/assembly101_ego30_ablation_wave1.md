# Assembly101 ego-30 controlled ablation report

Source commit: `d5f865e`.
The held-out test split was not evaluated in this screening wave.

## Absolute validation results

| Run | Factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min | VRAM GiB | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | recording-level corrected evaluation baseline | 8.232 | 3.255 | 0.414 | 37.497 | 1.889 | 29.285 | 0.005 | 0.051 | 11.429 | 0.126 | 145.932 | 2.523 | 8 |
| A1 | event-centered mistake exposure | 7.788 | 3.971 | 0.667 | 37.113 | 1.724 | 39.752 | 0.011 | 0.660 | 2.857 | 1.320 | 2.470 | 2.566 | 6 |
| A2 | causal graph and learned-next-action context | 9.133 | 3.873 | 0.497 | 38.749 | 1.729 | 41.587 | 0.019 | 1.460 | 2.857 | 1.460 | 0.934 | 2.586 | 6 |
| A3 | learned event evidence fusion | 12.406 | 4.986 | 1.014 | 35.715 | 4.498 | 43.129 | 0.007 | 0.047 | 37.143 | 0.115 | 514.477 | 2.564 | 12 |

## Attributed deltas

A positive delta is better except for false alerts/minute, where a negative delta is better.

| Contrast | Isolated factor | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Mistake AP | Incorrect-event PR-AUC | Incorrect-event F1 | Incorrect-event recall | Incorrect-event F1 ±4s | False alerts/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1−A0 | event-centered mistake exposure | -0.444 | +0.717 | +0.253 | -0.383 | -0.165 | +10.467 | +0.006 | +0.609 | -8.571 | +1.194 | -143.462 |
| A2−A1 | causal graph and learned-next-action context | +1.345 | -0.098 | -0.169 | +1.636 | +0.005 | +1.835 | +0.008 | +0.800 | +0.000 | +0.140 | -1.536 |
| A3−A1 | learned event evidence fusion | +4.618 | +1.015 | +0.347 | -1.399 | +2.774 | +3.377 | -0.004 | -0.613 | +34.286 | -1.205 | +512.007 |
