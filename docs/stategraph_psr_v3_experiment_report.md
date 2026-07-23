# StateGraph-PSR v3 experiment report

Date: 2026-07-20

## Decision

The implementation is healthy enough for controlled pilots, but the fallback
feature configuration is **not yet approved for the final 80-epoch benchmark**.
The four-recording overfit test proves that the model and losses can learn both
segmentation and incorrect events. On the official recording-level validation
split, however, the rare-fault result remains too fragile and false-alert-heavy.

This distinction matters: extending a weak pilot to 80 epochs does not create
new fault information. The next long run should use the preferred VideoMAEv2
SSv2 cache and should retain early stopping and the multi-seed gate below.

The test split has not been used.

## Data and compute

- Dataset: IndustReal, cache schema v2.
- Split: 36 train recordings, 16 validation recordings, 0 test records in the
  current cache index.
- Sampled rows: 15,801 train and 7,613 validation at 10 FPS with stride 5.
- Completion events, train: 235 correct, 14 incorrect, 66 remove.
- Completion events, validation: 109 correct, 5 incorrect, 34 remove.
- Cached visual inputs: fallback Swin3D-S motion and ConvNeXt-Tiny appearance,
  both represented as precomputed 768-d arrays.
- Hardware: RTX 4090, 24 GB; BF16; batch size 2; accumulation 4.

The 14-to-235 incorrect/correct training ratio is the main statistical risk.
Metrics for five validation faults can move by roughly 20 recall points from a
single event, so no single-seed fault result should be treated as stable.

## Architecture refinements evaluated

The v3 branch adds:

1. a bounded evidence/transition-prior mixture, preventing the procedure graph
   from overwhelming current visual evidence;
2. a causal progress head inspired by progress-aware online segmentation;
3. asymmetric completion loss and outcome-aware batches for sparse events;
4. action- and state-conditioned component event features;
5. per-component normal-execution prototypes and an anomaly loss;
6. an outcome-balanced event-to-state mapping learned from training only;
7. a single-outcome event decoder with temporal suppression;
8. outcome-specific peak proposals so majority-correct evidence cannot hide a
   sharper incorrect peak;
9. causal state-transition evidence using positive probability differences;
10. strict and latency-aware event metrics at +/-1, +/-2, and +/-4 cache steps.

## Results

### Functional overfit gate

Four-recording v3 overfit, 15 epochs:

| Metric | Validation on repeated subset |
|---|---:|
| Frame accuracy | 60.39 |
| Edit | 48.21 |
| F1@50 | 45.67 |
| Completion-event F1 | 35.51 |
| Incorrect-event precision / recall / F1 | 100 / 100 / 100 |
| Remove-event F1 | 66.67 |
| State accuracy | 93.53 |
| Incorrect normality AP | 100 |

This is a wiring test, not a generalization result. It confirms gradients,
labels, rare-window sampling, and the event/state branches can fit fault data.

### Official seed-7 pilot

Twenty epochs on all 36 training recordings; best checkpoint at epoch 15:

| Metric | Result |
|---|---:|
| Frame accuracy | 28.51 |
| Edit | 26.70 |
| F1@10 / F1@25 / F1@50 | 25.84 / 21.81 / 12.80 |
| Graph-filtered vs raw F1@50 | 12.80 vs 12.56 |
| State accuracy | 87.19 |
| Incorrect normality AP | 7.05 |
| Mean anomaly, correct / incorrect | 0.295 / 0.327 |

An intermediate outcome-specific decoder found one incorrect event, but at
below 0.1% precision. A training-derived refractory rule (the minimum observed
same-component gaps were 4.0 s for any event and 14.5 s between installations)
removed that unstable match. Final strict and latency-aware incorrect F1 are
therefore zero. The prototype score weakly ranks faults in this seed, but does
not produce a deployable alert policy.

The procedure graph was helpful but small (+0.24 F1@50), which is the intended
behavior of the bounded filter. Segmentation continued improving late in the
pilot, so action recognition may benefit from more epochs. Fault recognition is
limited by representation and sample support, not merely optimization length.

### Official seed-17 pilot

Twenty-five epochs; best checkpoint at epoch 22:

| Metric | Result |
|---|---:|
| Frame accuracy | 30.77 |
| Edit | 27.53 |
| F1@10 / F1@25 / F1@50 | 29.38 / 24.78 / 15.32 |
| Graph-filtered vs raw F1@50 | 15.32 vs 15.06 |
| State accuracy | 87.46 |
| Remove-event precision / recall / F1 | 17.86 / 9.80 / 12.66 |
| Incorrect-event precision / recall / F1 | 0 / 0 / 0 |
| Incorrect-event PR-AUC | 0 |
| Incorrect normality AP | 5.36 |

This independent seed confirms repeatable action improvement and a small,
non-destructive graph benefit. It also confirms that the current fault branch
does not generalize. The incorrect normality means actually reversed in this
seed (0.252 for correct versus 0.249 for incorrect), so another 55 epochs on
the same frozen features is not an evidence-based remedy.

## Readiness gate

Approve the 80-epoch, early-stopped run only after at least two of three
independent 20--30 epoch seeds satisfy all of the following on validation:

- action Edit and F1@50 improve over the v2 pilot and do not peak immediately;
- incorrect precision and recall are both non-zero;
- incorrect PR-AUC beats the empirical positive-rate baseline;
- a manually reviewed alert table shows plausible component and time matches;
- the fixed-threshold result does not collapse while calibrated F1 improves;
- graph-filtered action metrics are neutral or better than raw metrics.

Even after passing this gate, call the fallback-cache result an ablation. The
preferred benchmark should replace Swin3D-S with cached VideoMAEv2-giant SSv2
features. More clearly annotated incorrect examples, or a procedural-error
pretraining dataset such as EgoPER, are higher-value changes than increasing
the temporal head size.

## Dashboard

Every run writes TensorBoard events, `history.jsonl`, `system.jsonl`, a best
checkpoint, and a JSON summary. The live desktop dashboard is served at:

```text
http://192.168.20.148:6006
```

The desktop system Python had an incompatible OpenSSL package, so TensorBoard
is isolated at `/home/aiops/.venvs/aiops-dashboard`; the training environment
was not changed. Restart it with:

```bash
nohup /home/aiops/.venvs/aiops-dashboard/bin/python -m tensorboard.main \
  --logdir /home/aiops/AIOps/runs --bind_all --port 6006 \
  >/home/aiops/AIOps/runs/tensorboard.log 2>&1 </dev/null &
```

The reusable monitor is `aiops.training.monitoring.TrainingMonitor`. It logs
task losses, validation metrics, learning rate, gradient norm, graph strength,
and NVIDIA utilization, allocated/reserved/peak memory, temperature, and power.

## Evidence base

- Shen et al., *Progress-Aware Online Action Segmentation for Egocentric
  Procedural Task Videos*, CVPR 2024.
- Lee et al., *Error Detection in Egocentric Procedural Task Videos*, CVPR
  2024 (EgoPER).
- Flaborea et al., *PREGO: Online Mistake Detection in Procedural Egocentric
  Videos*, CVPR 2024.
- Ridnik et al., *Asymmetric Loss for Multi-Label Classification*, ICCV 2021.
- Bahrami et al., *How Much Temporal Long-Term Context is Needed for Action
  Segmentation?*, ICCV 2023.
- Xu et al., *Differentiable Task Graph Learning*, 2024.

Full paper links and the design-to-evidence mapping are in
`docs/stategraph_psr_v3_refinement.md`.

## Subsequent large-model result

The higher-capacity v4 candidate is documented in
`docs/stategraph_psr_large_refinement.md`. It increases the causal head from
about 3.3M to 34.1M parameters, adds two deeply supervised action-refinement
stages and a dedicated event/state branch, and uses 384-step context.

On the official validation split it improved frame accuracy/Edit/F1@50 to
33.41/29.92/18.05, versus 30.77/27.53/15.32 for the best lite pilot. Incorrect
F1 and PR-AUC remained zero. The result validates the larger model for Stage-1
action segmentation while strengthening the conclusion that fault recognition
is representation/data limited under the fallback cache.
