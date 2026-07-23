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
oversampled, and extra crops are centered around rare incorrect events. The
best checkpoint maximizes:

`state macro recall + effect macro recall + incorrect-effect recall`.

This is intentionally more mistake-sensitive than validation loss or overall
accuracy, both of which can favor the dominant normal/no-change classes.

TensorBoard files, GPU telemetry, `last.pt`, `best.pt`, and `result.json` are
written under the run directory.
