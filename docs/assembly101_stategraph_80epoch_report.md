# Assembly101 StateGraph-PSR: 80-epoch report

## Result

The selected StateGraph-PSR model completed all 80 epochs on an NVIDIA RTX 5000 Ada (32 GB). The validation-selected epoch is 67. On the actor-held-out test split, the model obtains **46.82% frame accuracy**, **27.24 edit**, **13.91 F1@50**, and **45.43% state macro-F1** across 87 action classes and 55 component states.

This is a reasonable research architecture for the next iteration: it generalizes procedural actions and coarse component state substantially beyond trivial 87-way chance performance, fits comfortably on the available GPU, and exports to a 21 MB BF16 checkpoint. It is not yet a satisfactory exact mistake-alert system: test mistake normality AP is 20.19%, exact incorrect-event F1 is 0.37%, and false alerts are 13.12/minute.

## Architecture selection

Two 12-epoch pilots used identical cached ConvNeXt-Tiny appearance and causal feature-difference inputs.

| Candidate | Parameters | Best val frame acc. | Best val edit | Best val F1@50 | Peak val state macro-F1 |
|---|---:|---:|---:|---:|---:|
| Compact temporal baseline | 1.37 M | 41.68% | 10.99 | 7.44 | 40.07% |
| Selected graph + refinement model | 10.79 M | 42.39% | 13.99 | 7.12 | 41.11% |

The selected model uses a 256-dimensional causal temporal trunk with eight dilated blocks, attention every two blocks, eight heads, one four-block action-refinement stage, four event blocks, and a differentiable procedure graph initialized at strength 0.12. It has 10,793,341 trainable parameters. The larger candidate was selected for stronger temporal consistency and state reasoning, and its benefit became clear during longer training.

## Dataset and controls

- 60 recordings and 28,035 sampled frames; 40/10/10 train/validation/test recordings.
- Actor-disjoint splits prevent identity leakage.
- One consistent fixed RGB camera (`C10095`) sampled at 1 fps.
- Official mistake-detection annotations: 511 correct, 142 incorrect, and 75 correction segments.
- Frozen ConvNeXt-Tiny ImageNet-1K embeddings (768 appearance + 768 causal difference dimensions).
- Thresholds calibrated only on validation and frozen before the single test evaluation.
- Dataset mirror and annotation sources are revision-pinned and every downloaded archive is SHA-256 verified.

See [the methodology](assembly101_stategraph_methodology.md) and the committed subset manifest for complete provenance. This fixed-view derivative is not the official egocentric Assembly101 release, so these numbers are not an official Assembly101 benchmark.

## Final metrics

| Metric | Validation (epoch 67) | Test |
|---|---:|---:|
| Frame accuracy | 50.06% | 46.82% |
| Seen-action accuracy | 50.58% | 47.86% |
| Edit | 29.72 | 27.24 |
| F1@10 | 27.32 | 23.88 |
| F1@25 | 23.77 | 19.01 |
| F1@50 | 16.28 | 13.91 |
| State accuracy | 96.37% | 96.73% |
| State macro-F1 | 43.42% | 45.43% |
| Mistake normality AP | 32.81% | 20.19% |
| Exact incorrect-event F1 | 0.00% | 0.37% |
| Incorrect recall within 4 s | 15.63% | 29.17% |
| False alerts/minute | 16.18 | 13.12 |

The accuracy gap from validation to test is modest, supporting useful actor generalization. The high state accuracy is class-imbalanced, so macro-F1 is the more informative state result. Event-level scores expose the primary limitation: reducing 30 fps event annotations to a 1 fps visual mirror makes exact sparse timing difficult, and only 142 mistake segments exist across all splits.

Five test action compositions (105 frames) are absent from training and zero-shot composition accuracy is 0%. These examples were deliberately retained and reported rather than leaking test identities or actions into training.

## Reproduction and artifact

Run `scripts/train_assembly101_80ep.sh` after downloading and caching the pinned subset. It fixes the seed, model, optimizer, sampling, split, and 80-epoch schedule used here.

- Checkpoint: `artifacts/stategraph_assembly101_80ep_bf16.pt`
- Size: 21,751,479 bytes (about 20.7 MiB)
- SHA-256: `d45fa88e777bd32bc03762f262cd06797beac5c5c442ce0724c7108c8b782196`
- Selected checkpoint epoch: 67

The exported checkpoint contains BF16 model weights, the full model configuration, dataset label metadata, transition matrix, validation-frozen event thresholds, and validation/test metrics. It excludes optimizer state and raw data.

## Recommended next work

The architecture is adequate; data resolution is now the bottleneck. The highest-value improvement is access to the full-resolution Assembly101 videos (preferably egocentric views), followed by a temporal video encoder and event-centered sampling. Exact mistake alerts should not be deployed from this checkpoint. The current model is useful for action/state research, reproducible baselines, and testing those improvements.
