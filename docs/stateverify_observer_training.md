# StateVerify observer training

This experiment trains the component state/effect observer independently from
the established Stage-1 action model. That separation protects the existing
action-recognition checkpoint while testing whether explicit hand–object
evidence improves mistake attribution.

## Enforced input contract

The trainer refuses incompatible caches. It requires:

- global causal motion and appearance features;
- ROI cache format version 2 in the exact order `left_hand`, `right_hand`,
  `active_object`, `interaction_context`;
- a sensor cache explicitly marked as `hands + headset_pose`;
- `gaze_excluded=true`.

Legacy ROI v1 and legacy sensor caches therefore fail fast instead of silently
feeding gaze values into the model.

## Preparation

The base visual features do not need to be recomputed. The pipeline first
replaces only the sensor tensor using the raw hand and pose CSV files. It then
extracts the four new ROI streams and attaches them to the rebuilt cache:

```bash
PYTHON=/home/aiops/miniconda3/envs/psr_env/bin/python \
  bash scripts/run_stateverify_observer_pipeline.sh all
```

The `all` phase ends with a one-epoch, three-batch smoke run. After that passes:

```bash
PYTHON=/home/aiops/miniconda3/envs/psr_env/bin/python \
  bash scripts/run_stateverify_observer_pipeline.sh train
```

## Supervision and selection

Persistent raw states are remapped from `incorrect / pending / correct` to
`not_completed / installed_correct / installed_incorrect`. Typed effects are
derived from consecutive observed states; exact PSR completion events override
derived labels at their annotated frame.

Class weights are computed from the training split. Incorrect-event windows are
guaranteed in every training batch, and extra crops are centered around rare
incorrect events. The typed-effect loss keeps every change/error target but
only the hardest bounded set of `no_change` frames. This prevents the observed
training distribution (19 incorrect effects versus 172,929 no-change frames)
from collapsing the effect head into a trivial no-change classifier.

The state and effect outputs are factorized for label efficiency. State is
predicted as `installed probability × conditional correctness`; effect is
predicted as `no-change / complete / remove × conditional correctness`.
Incorrect examples therefore reuse the well-supervised installation and
completion representations instead of requiring independent three-way or
four-way class prototypes.

The best checkpoint maximizes:

`state macro recall + effect macro recall + installed-incorrect recall + 2 × incorrect-effect recall`.

This is intentionally more mistake-sensitive than validation loss or overall
accuracy, both of which can favor the dominant normal/no-change classes.

TensorBoard files, GPU telemetry, `last.pt`, `best.pt`, and `result.json` are
written under the run directory.
