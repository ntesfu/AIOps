# Assembly101 ego30 ablation wave 6

Wave 6 replicates the expanded-data baseline at seeds 17 and 27. Together with
E0 (seed 7), this measures how much of the apparent improvement is robust rather
than a favorable initialization. All runs use the same 133/15 recording
train/validation split, one ego view, the frozen 30 FPS-derived cache, no
event-centered crops, and no procedural event context. The held-out test split
was not evaluated.

## Legacy checkpoint selection

| Run | Seed | Epoch | Frame accuracy | Edit | F1@50 | State macro-F1 | Incorrect-state F1 | Normality AP | Incorrect-event F1 | Event recall | False alerts/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 7 | 14 | 16.395% | 4.818 | 1.547% | 48.784% | 7.321% | 44.290% | 5.263% | 2.857% | 0.019 |
| F0 | 17 | 26 | 18.319% | 7.817 | 2.882% | 45.926% | 0.248% | 31.203% | 1.923% | 2.857% | 0.629 |
| F1 | 27 | 28 | 17.048% | 7.254 | 2.639% | 48.496% | 0.389% | 34.429% | 0.000% | 0.000% | 0.000 |
| Median | — | — | 17.048% | 7.254 | 2.639% | 48.496% | 0.389% | 34.429% | 1.923% | 2.857% | 0.019 |

Action segmentation and overall state classification replicate: all three seeds
land near 16.4–18.3% frame accuracy and 45.9–48.8% state macro-F1. Exact mistake
detection does not. One seed selects a completely silent detector, and the
incorrect-state F1 ranges from 0.248% to 7.321%. This variance is too large to
promote the legacy mistake head as a reasonable final model.

## Effect of operational selection

The new selector rejects silent checkpoints and balances action and mistake
quality harmonically. Retrospectively, it leaves E0 epoch 14 and F0 epoch 26
unchanged. For F1 it selects epoch 10 instead of silent epoch 28:

| F1 checkpoint | Frame accuracy | Edit | F1@50 | Normality AP | Incorrect-event F1 | Recall | False alerts/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy epoch 28 | 17.048% | 7.254 | 2.639% | 34.429% | 0.000% | 0.000% | 0.000 |
| Operational epoch 10 | 12.032% | 3.536 | 0.640% | 45.521% | 3.030% | 2.857% | 0.278 |

This fixes the unsafe selection behavior but exposes the underlying conflict:
for seed 27, useful mistake evidence appears before action segmentation matures
and is later forgotten. Selection alone therefore cannot solve the architecture.
It is an evaluation safeguard, while the factorized head in wave 8 targets the
learning problem directly.

## Resource and provenance audit

- Peak training VRAM was 2.680, 2.680, and 2.682 GiB for E0, F0, and F1,
  respectively, against the enforced 23 GiB ceiling.
- Each run completed all 32 epochs and produced a summary and inference
  checkpoint.
- F1's allocation shell returned nonzero only after the Python process had
  completed and exported its artifacts because the shared launcher was updated
  during the live allocation. Its persisted 32-row history, summary, and
  checkpoint are intact; no metrics were reconstructed from console output.

## Decision

Expanded training data is robustly useful for action and overall state metrics,
but the independent per-component onset formulation is not robust enough for the
final model. Continue with the factorized timing/localization ablation and retain
the operational selector. Do not spend the held-out test split on any wave-6
checkpoint.
