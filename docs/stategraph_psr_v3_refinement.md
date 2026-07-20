# StateGraph-PSR v3 refinement and live training dashboard

## Why v2 was not ready for a long run

The cache-v2 seed-7 pilot was useful because it failed early and cheaply. Its
selected checkpoint reached 9.70 F1@50 and 0.00 incorrect-event F1. Inspection
found four correctable causes beyond visual-backbone quality:

1. The original graph filter added a learnable multiple of the log transition
   prior to every action logit. Its initial effective multiplier was about 1.7,
   so an early wrong prediction could suppress the correct visual class by tens
   of logits and propagate the error through the recording.
2. Completion was an extremely sparse multi-label target but used symmetric
   focal BCE. Easy negatives still accumulated most of the optimization signal.
3. Outcome classification saw only isolated positive event rows—14 incorrect
   events in training—and the weighted sampler did not guarantee that an
   optimizer step contained an incorrect completion.
4. Event decoding thresholded completion at 0.5 and then took an outcome argmax.
   This discarded useful joint probability information and provided no
   validation-derived operating point or PR-AUC.

## Evidence used

- [ProTAS (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf)
  shows that causal training, action-progress prediction, and task-graph context
  address online oversegmentation in egocentric procedures. This motivates the
  new progress auxiliary head and retaining a causal graph without allowing it
  to dominate current evidence.
- [Error Detection in Egocentric Procedural Task Videos (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html)
  combines action segmentation with contrastive step prototypes and hand-object
  relational evidence. This motivates conditioning component-event features on
  both the current action belief and persistent assembly-state belief, plus a
  component normal-execution prototype score.
- [PREGO (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.html)
  treats procedural errors as open-set and detects inconsistency between the
  recognized action and expected procedure. StateGraph keeps supervised outcome
  labels but exposes prototype normality and graph inconsistency for later
  open-set alert fusion.
- [Asymmetric Loss for Multi-Label Classification (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)
  separates positive and negative focusing and clips easy negatives under severe
  multi-label imbalance. Completion now uses this formulation.
- [LTContext (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Bahrami_How_Much_Temporal_Long-Term_Context_is_Needed_for_Action_Segmentation_ICCV_2023_paper.html)
  finds full-video context important for segmentation. The compact model retains
  sparse causal attention and must later compare longer windows or persistent
  streaming memory; v3 does not pretend 256 cache steps settle that question.
- [Differentiable Task Graph Learning (2024)](https://arxiv.org/abs/2406.01486)
  reports gains from learned task graphs in online mistake detection. StateGraph
  retains differentiable graph reasoning but changes it from an unbounded
  product-of-experts to a bounded convex evidence/prior mixture.

## Architecture changes

### Bounded graph evidence fusion

For current visual posterior `e_t`, transition prior `g_t`, and learned gate
`alpha` in `[0,1]`, v3 uses:

```text
p_t = (1 - alpha) e_t + alpha g_t
```

The default `alpha` is 0.12. The graph can stabilize an uncertain observation,
but cannot assign arbitrarily large negative evidence to a visually supported
action. Raw and graph-filtered action metrics are both logged.

### Progress-aware causal segmentation

A scalar progress head predicts normalized position within the current action
segment. It uses Smooth-L1 supervision from 0 to 1. This does not use future
frames as model input: the target is available during training, while inference
remains causal. Boundary, progress, action, and next-action heads provide
complementary segmentation signals.

### Action/state-conditioned event branch

The event representation adds three causal signals:

```text
temporal feature
  + expected compositional action prototype
  + projected persistent component-state probabilities
```

Completion and per-component outcome heads operate on this representation. A
normal-execution prototype per component produces a component normality score;
incorrect events receive higher weight in its binary auxiliary loss, and anomaly
evidence contributes to the incorrect-outcome logit. This supplies an open-set
signal without adding a second video encoder.

Each completion component is also mapped, from training annotations only, to
the persistent state column whose correct/incorrect/pending values best agree
with its observed outcomes. Mapping agreement is balanced across outcomes so
the abundant correct events cannot erase the scarce incorrect signal, and it
allows a two-cache-step post-event delay for state annotations to update. At
inference, a fault can be proposed by either the
supervised completion/outcome score or the geometric mean of a rising mapped
incorrect-state probability and prototype anomaly. This prevents the sparse
completion head from vetoing dense, independently learned evidence of a state
failure.

### Rare-event optimization and decoding

- Every training batch contains a configurable number of windows with a real
  incorrect completion event (`--rare-windows-per-batch`, default 1).
- Incorrect-event windows are not randomly jittered away from their event.
- Completion uses asymmetric multi-label loss.
- Decoding proposes one completion peak per component, applies temporal
  suppression, and assigns exactly one outcome using
  `P(completion) * P(outcome | completion)`. It cannot emit correct, incorrect,
  and remove alerts for the same physical event.
- Correct, incorrect, and remove thresholds are calibrated independently on
  validation; fixed-threshold metrics remain visible for auditability.
- Event PR-AUC and selected thresholds are logged. Test thresholds must be
  frozen before a single test evaluation.

Re-evaluate a saved checkpoint after a decoder-only change without retraining:

```bash
python3 -m aiops.evaluation.evaluate_stategraph_checkpoint \
  --checkpoint /path/to/best_checkpoint.pt \
  --cache-index /path/to/cache/index.json \
  --split val --output /path/to/decoder_metrics.json
```

## Reusable dashboard

Training always writes:

- `history.jsonl`: epoch losses and validation metrics;
- `system.jsonl`: timestamped GPU utilization, memory, temperature, and power;
- `tensorboard/`: live scalar events when the monitor extra is installed;
- `best_checkpoint.pt`, `metrics.json`, and `summary.json`.

Install once on any training project environment:

```bash
python -m pip install -e '.[ml,monitor]'
```

Launch one dashboard for every run below a common directory:

```bash
python3 -m tensorboard.main --logdir /home/aiops/AIOps/runs --bind_all --port 6006
```

Open `http://192.168.20.148:6006`. The dashboard automatically groups:

- per-update total and task losses;
- learning rate and gradient norm;
- epoch action, event, state, calibration, and raw-vs-graph metrics;
- learned graph gate;
- GPU utilization, allocated/reserved/peak memory, temperature, and power.

The logger is model-agnostic: import `TrainingMonitor` from
`aiops.training.monitoring`, call `log_scalars(group, values, step)`, and call
`log_system(step, device)`. If TensorBoard is unavailable, JSONL logging and
training continue rather than failing.

Threshold sweeps run at epoch 1 and then every five epochs by default; other
epochs reuse the latest validation thresholds. This preserves live event metrics
without making the calibration grid the dominant training cost. Change the
cadence with `--calibration-interval`.

## Readiness gate for 80 epochs

Authorize the long run only when at least two of three 20–30 epoch seeds meet
all of these validation conditions:

- action F1@50 and Edit improve materially over v2 and do not peak only at the
  beginning of training;
- calibrated incorrect-event precision and recall are both non-zero;
- incorrect-event PR-AUC exceeds the random/always-positive diagnostic baseline;
- fixed-threshold metrics do not collapse while calibrated metrics improve;
- raw and graph-filtered metrics show that the graph helps or is neutral;
- false alerts from the five validation incorrect events are manually reviewed;
- the overfit test still decreases every new loss and detects incorrect events.

The fallback Swin3D-S cache is sufficient to validate architecture and training
logic. A final result claim requires a controlled repeat with the preferred
VideoMAEv2-giant SSv2 features.
