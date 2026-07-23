# StateVerify-PSR Core

## Scope

StateVerify-PSR Core is the first implementable slice of the state-centric
architecture. It does not replace the trained StateGraph action and event
experts. It converts their causal evidence into persistent, typed component
beliefs and auditable alerts.

The implementation deliberately begins after feature extraction:

```text
causal StateGraph emissions
    + procedure residual
    + execution residual
    + effect residual
              |
              v
factorized component Bayesian filter
              |
              v
confirmed alert / recovery transition
```

The filter has three states per component:

1. `not_completed`
2. `installed_correct`
3. `installed_incorrect`

`remove` is an operator that returns a component to `not_completed`; it is not
treated as a fourth persistent state. `in_progress` can be added later as a
latent duration state, but the current IndustReal labels do not supervise it.

## Evidence contract

Every residual is calibrated to `[0, 1]` on a `[time, component]` axis:

- **procedure**: incompatibility between the observed action and the valid
  next-step set;
- **execution**: distance from normal component-conditioned execution
  prototypes;
- **effect**: mismatch between observed and expected component state change.

Missing evidence has an explicit mask and is neutral. It is never represented
as a confident zero. The tracker retains all three values and reports the
dominant positive contribution with each alert.

The neural state head supplies emission probabilities, while the tracker owns:

- persistence;
- legal transition priors;
- confirmation windows;
- alert hysteresis;
- recovery transitions.

Components are filtered independently, keeping inference linear in component
count rather than enumerating a `3^N` global state space.

## Historical StateGraph compatibility

The existing model emits persistent states in this order:

```text
incorrect, pending, correct
```

StateVerify uses:

```text
not_completed, installed_correct, installed_incorrect
```

Use `canonicalize_stategraph_emissions` before tracking historical outputs. The
offline CLI performs this conversion with `--stategraph-order`.

## Offline evaluation

Prepare a compressed NumPy archive:

```python
np.savez_compressed(
    "emissions.npz",
    state_emissions=state_probabilities,       # [T, C, 3]
    procedure_residual=procedure_residual,     # [T] or [T, C]
    execution_residual=execution_residual,     # [T, C]
    effect_residual=effect_residual,           # [T, C]
    component_names=np.asarray(component_names),
)
```

Then run:

```bash
PYTHONPATH=src python scripts/evaluate_stateverify_tracker.py \
  --input emissions.npz \
  --output stateverify_trace.json \
  --stategraph-order
```

The JSON output includes the complete posterior trace, persistent alert state,
fused fault evidence, per-residual contributions, and typed alert/recovery
events.

Thresholds and transition priors must be selected on operator-disjoint
validation data. The test split must not be used for tracker calibration.

## Next integration steps

Selected checkpoints now automatically export reconstructed validation
recordings to:

```text
<run>/stateverify_validation_emissions/
```

Each archive uses the portable state order and contains all three residual
channels. The effect residual is currently a conservative proxy: the positive
onset of the existing incorrect-state probability. It will be replaced by the
dedicated observed-versus-expected state-change estimator after raw spatial
evidence is restored.

Remaining integration steps:

1. Compare direct onset thresholding with the tracker under the same
   false-alerts-per-minute limit.
The gaze-free component observer is implemented in
`src/aiops/models/stateverify_effect.py`. It consumes:

- causal global video features;
- left-hand and right-hand ROI features;
- active-object and interaction-context ROI features;
- hand joints;
- headset pose;
- optional depth.

Component queries attend to the four ROI tokens, then causal component-wise
temporal blocks predict persistent state and four typed effects:
`no_change`, `complete_correct`, `complete_incorrect`, and `remove`. The effect
residual is a bounded mismatch between the observed effect distribution and the
procedure-supplied expected distribution.

2. Restore raw IndustReal RGB, object detections, hand and headset-pose streams, depth, and
   state labels on the working disk.
3. Train the real component state/effect observer.
4. Replace heuristic residual calibration with validation-fitted calibration,
   preserving the explicit residual channels and abstention behavior.
