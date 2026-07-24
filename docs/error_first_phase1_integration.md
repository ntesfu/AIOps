# Error-first Phase 1/2 integration runbook

This runbook wires the three new, locally-tested modules from the approved
error-first architecture plan into the StateVerify training/eval pipeline on the
remote Ubuntu workstation (`192.168.20.148`, RTX 4090 24 GB, conda env
`/home/aiops/miniconda3/envs/psr_env`). The modules are dataset/model building
blocks with CPU unit tests; the training wiring below can only be validated where
PyTorch and the IndustReal cache exist.

## What is already implemented and tested

| Module | Purpose | Local test |
|---|---|---|
| [src/aiops/data/counterfactuals.py](../src/aiops/data/counterfactuals.py) | Manufacture hard-negative incorrect installs from correct ones (label flip + persistent-state relabel + wrong-part ROI swap), with a train-only leakage guard | [tests/test_counterfactuals.py](../tests/test_counterfactuals.py) — 7 pass |
| [src/aiops/models/expected_effect.py](../src/aiops/models/expected_effect.py) | Learn the *normal* effect from correct data; residual = divergence(observed, expected) | [tests/test_expected_effect.py](../tests/test_expected_effect.py) — runs on GPU box |
| [src/aiops/validation/selection_verifier.py](../src/aiops/validation/selection_verifier.py) | Symbolic wrong-part/tool residual from detected active-object category vs. schema | [tests/test_selection_verifier.py](../tests/test_selection_verifier.py) — 7 pass |

Run the tensor tests on the workstation first:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m pytest tests/test_expected_effect.py \
  tests/test_counterfactuals.py tests/test_selection_verifier.py -q
```

## Prerequisite (Phase 0): strict-causal cache

Use a schema-v2, **trailing-clip** cache with ROI `format_version=2` motion_aux
(the ROI path already enforces `--require-causal`). Confirm with the cache audit
before any training:

```bash
PYTHONPATH=src "$PYTHON_BIN" -m aiops.data.audit_stategraph_cache \
  --cache-index "$CACHE_INDEX" --output "$CACHE_DIR/audit.json" --fail-on-warnings
```

## Wiring — counterfactual engine (Phase 1b)

Integration point: [src/aiops/training/train_stateverify_observer.py](../src/aiops/training/train_stateverify_observer.py),
`train()` after `train_dataset` is built (~line 294) and `event_state_indices`
is known (~line 262).

```python
from aiops.data.counterfactuals import (
    CounterfactualConfig, CounterfactualWindowDataset,
    build_wrong_part_donors_from_records,
)

cf_config = CounterfactualConfig(
    roi_count=len(ROI_ORDER), roi_dim=roi_dim,
    event_state_indices=event_state_indices,
)
donors = build_wrong_part_donors_from_records(train_records, cf_config)  # train-only
train_dataset = CounterfactualWindowDataset(
    train_dataset, donors, cf_config, rate=args.counterfactual_rate, seed=args.seed,
)
```

Notes:
- `donors` are collected from **training records only**; `assert_train_only`
  raises otherwise. The validation dataset is a separate instance and is never
  wrapped.
- `CounterfactualWindowDataset` is torch-free and `set_epoch`-aware, so it drops
  into the existing `DataLoader(batch_sampler=...)` unchanged.
- Add `--counterfactual-rate` (default e.g. 0.5) to `build_parser`.
- The matched contrastive/ranking term (plan Phase 1b) pairs each synthesized
  incorrect with its source correct install via `counterfactual_events`
  provenance; extend `build_state_effect_loss` next, keeping the counterfactual
  flag out of any validation path.

## Wiring — expected-effect predictor (Phase 1a)

Replace the proxy effect residual with the learned reference. In `train()`:

```python
from aiops.models.expected_effect import (
    ExpectedEffectPredictorConfig, build_expected_effect_predictor,
    expected_effect_loss, expected_effect_residual,
)

ee_config = ExpectedEffectPredictorConfig(
    hidden_dim=args.hidden_dim, num_components=first.num_components,
    num_steps=len(metadata.get("action_ids", [])), effect_lag=args.effect_lag,
)
expected_model = build_expected_effect_predictor(ee_config).to(device)
# add expected_model.parameters() to the AdamW param groups
```

In the forward/loss step, feed the observer's component features:

```python
observed = model(**_observer_inputs(batch, config))
expected = expected_model(
    observed["component_features"],
    step_probabilities=observed["step_probabilities"],
    valid_mask=batch["valid_mask"],
)
typed = state_effect_targets(batch, event_state_indices)
loss_ee = expected_effect_loss(
    expected["expected_effect_logits"], typed["effect"], typed["effect_mask"],
)  # normal-only: complete_incorrect excluded
# emission-time effect residual for the tracker / streaming eval:
residual = expected_effect_residual(
    observed["effect_probabilities"], expected["expected_effect"],
)  # [B, T, C] in [0, 1]
```

Add `loss_ee` to `losses["total"]` with a small weight (start 0.5). Export
`residual` into the reconstructed validation emissions
(`<run>/stateverify_validation_emissions/`) as the `effect_residual` channel,
replacing the onset proxy.

## Wiring — selection verifier (Phase 2)

1. Author `step_expectations` in
   [configs/procedure_schemas/industreal_v1.json](../configs/procedure_schemas/industreal_v1.json)
   from the real IndustReal OD vocabulary (`OD_labels.json`) and the AR action
   list. Shape per step:

   ```json
   "step_expectations": {
     "<action_id>": {"expected_categories": [<od_category_id>, ...],
                      "components": [<event_component_index>, ...]}
   }
   ```

2. In the streaming evaluator
   [scripts/evaluate_stateverify_streaming.py](../scripts/evaluate_stateverify_streaming.py),
   compute the residual from each recording's `motion_aux` and predicted steps:

   ```python
   from aiops.validation.selection_verifier import (
       SelectionVerifierConfig, load_selection_schema,
       selection_residual_from_motion_aux,
   )
   sv_config = SelectionVerifierConfig(
       roi_count=len(ROI_ORDER), roi_dim=roi_dim,
       num_state_components=num_state_components,
       event_state_indices=event_state_indices,
   )
   schema = load_selection_schema(procedure_schema.get("step_expectations", {}))
   sel_residual, sel_mask = selection_residual_from_motion_aux(
       motion_aux, predicted_steps, schema, sv_config,
   )
   ```

   Feed `sel_residual`/`sel_mask` into the tracker's **procedure** channel of
   `TripleResidualEvidence`. Until the schema is authored it is an empty no-op
   (all masked), so it is safe to land before authoring.

## Gates (unchanged from the plan)

- **Phase 1 counterfactual benchmark:** incorrect-effect AP / correct-vs-incorrect
  separability beat the frozen Phase-0 baseline across seeds 7/17/29; real
  held-out anomaly separation improves over the current −0.44…0.32 range.
- **Phase 2 selection verifier:** near-zero false-positive rate on correct data;
  high recall on counterfactual wrong-part errors.
- **Phase 3 milestone:** ≥1 unique held-out incorrect state transition detected
  at a train-selected operating point, ≤0.25 FA/min dev / ≤2 FA/min operational,
  test split sealed.

## Leakage checklist

- Counterfactual donors and synthesis: training records only (`assert_train_only`).
- The validation `StateGraphCacheDataset` is never wrapped by `CounterfactualWindowDataset`.
- Synthesized samples carry `is_counterfactual=True`; assert this flag is absent
  in every validation/test batch before trusting a held-out number.
