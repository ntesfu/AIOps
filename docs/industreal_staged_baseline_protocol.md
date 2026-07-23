# IndustReal staged dual-expert baseline

## Purpose

This protocol measures the strongest already-implemented StateGraph-PSR design
on IndustReal before adding component ROIs, event-bag learning, or a persistent
belief tracker. That separation is important: the baseline tells us whether a
later gain came from better visual evidence/state reasoning rather than another
training-budget or checkpoint-selection change.

The baseline uses the 34M-class Large expert (384 hidden channels, 12 temporal
blocks), not the 87.5M XL expert. XL improved IndustReal action F1@50 only
slightly and did not improve incorrect-event recall, so repeating that capacity
sweep would conflict with the evidence-first plan. Dual-expert composition runs
two independently selected Large experts at inference.

The machine-readable specification is
[`configs/industreal_staged_baseline.json`](../configs/industreal_staged_baseline.json).
The executable entry point is
[`scripts/train_industreal_staged_baseline.sh`](../scripts/train_industreal_staged_baseline.sh).

## Two baselines, not one mixed result

Run the identical protocol on two independently named caches:

1. `legacy_offline`: the historical center-anchored clip cache. It can reproduce
   the v5 offline/bounded-lookahead result, but it is not a live-feed result.
2. `strict_causal`: a trailing-window cache where every source frame is at or
   before the prediction timestamp. This is the deployment baseline.

Never aggregate these runs. The cache contract is part of every run and artifact
name. Before training, verify the cache-generation manifest rather than trusting
the `CACHE_CONTRACT` string. The launcher enforces this with
`aiops.data.audit_stategraph_cache --require-causal` and refuses a strict-causal
run if any recording lacks source-frame provenance or uses a future frame.

## Why staged dual experts

One optimization trajectory previously traded action segmentation against rare
mistake recall. The staged baseline preserves a selected action representation
and gives the event/state branch a smaller, targeted adaptation:

- **Action stage:** 25-epoch ceiling, action-only validation selection.
- **Event stage:** initialize from that action checkpoint, freeze action modules,
  add two training-only crops around each incorrect event, and adapt for a
  16-epoch ceiling at a lower learning rate.
- **Composition:** route action outputs from the action expert and event/state
  outputs from the event expert. Calibrate event and state thresholds using
  validation only.

The event stage defaults to the legacy balanced validation selector. This is
intentional for a baseline: IndustReal has so few validation faults that a strict
non-zero-recall eligibility rule can produce no checkpoint and therefore hide a
real zero-recall result. Use `EVENT_SELECTION_STRATEGY=operational_harmonic` as
a later deployment-gated comparison.

## Development run

On Linux, from the repository root:

```bash
export CACHE_INDEX=/path/to/strict-trailing-industreal/index.json
export CACHE_CONTRACT=strict_causal
export SEED=7
export PYTHON_BIN=/path/to/python  # optional; defaults to .venv/bin/python
bash scripts/train_industreal_staged_baseline.sh all
```

Repeat with seeds `17` and `29`. TensorBoard logging and GPU telemetry are
enabled by the trainer unless `--disable-dashboard` is added manually.

For the historical cache, use a distinct identity:

```bash
export CACHE_INDEX=/path/to/legacy-centered-industreal/index.json
export CACHE_CONTRACT=legacy_offline
export SEED=7
bash scripts/train_industreal_staged_baseline.sh all
```

Stages may be scheduled separately with `action`, `event`, and `compose`. A run
manifest written by the trainer makes incompatible resume attempts fail rather
than silently mixing configurations.

To continue an interrupted action or event stage from its exact
`last_checkpoint.pt`, repeat the original environment and add `RESUME=1`. The
launcher retains the event stage's initialization path as immutable provenance
while the trainer restores model, optimizer, scheduler, sampler, and RNG state
from the resume checkpoint.

## Test-set seal

Training and composition do not touch test. After comparing all three seeds on
validation, freeze the cache, architecture, seed-selection rule, and thresholds.
Then evaluate the selected composed checkpoint exactly once:

```bash
export CACHE_INDEX=/path/to/strict-trailing-industreal/index.json
export CACHE_CONTRACT=strict_causal
export SEED=17  # replace with the seed selected by the frozen rule
export ALLOW_TEST_EVALUATION=1
bash scripts/train_industreal_staged_baseline.sh test
```

The explicit environment gate prevents an accidental test run during routine
development.

## Decision table

Report mean, standard deviation, and individual seeds. With only a handful of
incorrect incidents, the individual outcomes are more informative than a mean
alone.

| Capability | Required metrics |
|---|---|
| Action timeline | Frame accuracy, Edit, F1@10/25/50 |
| Persistent state proxy | State macro-F1, incorrect-state F1 |
| Exact alert | Incorrect-event precision/recall/F1 at ±1 s |
| Timing tolerance | Incorrect-event F1/recall at ±2 s and ±4 s |
| Ranking | Incorrect-event PR-AUC and normality average precision, with prevalence |
| Operations | False alerts/minute and mean matched-alert delay; add median delay in the subsequent metric refinement |

Frame accuracy cannot be used to select a fault detector. Likewise, a high
normality AP does not establish usable localization if exact event recall is
zero.

## Gate before architectural refinement

The baseline is complete when all three strict-causal validation runs and their
composed reports exist. Proceed to the component-centric evidence work even if
exact-event F1 is zero; a reproducible zero is a useful baseline. Do not respond
to zero recall by enlarging the temporal head. The next controlled change is
hand/tool/part/connection-region evidence, followed by component-matched event
bags and then a persistent component-state filter.
