# Portable procedure schema (cache v2)

## Why the label contract changed

IndustReal's raw PSR completion IDs are not mutually-exclusive action classes. An ID combines a semantic part with an outcome, and several parts can complete at the same timestamp. Converting those rows into a single dense class overwrites co-occurring events and creates classes with zero training frames after temporal sampling.

Cache schema v2 separates three kinds of supervision:

| Target | Meaning | Shape |
|---|---|---|
| `step` | mutually-exclusive action/step timeline from the dataset's action annotations | `T` |
| `completion` | multi-label component event indicator | `T × Cevent` |
| `component_outcome` | outcome at positive component events (`correct`, `incorrect`, `remove`; `-100` elsewhere) | `T × Cevent` |
| `state` / `state_mask` | persistent assembly-state supervision | `T × Cstate` |

The historical tensor name `step` remains in the cache/API for compatibility, but in v2 it always means a true action-segmentation class. It never contains a PSR completion ID.

Unannotated gaps in interval-style action files are assigned to an existing idle/background action when one is present; otherwise the adapter adds `__background__`. This gives live inference explicit negative supervision instead of silently ignoring idle video.

Action descriptions are stored alongside raw IDs. The default head factorizes names into a first-token verb and remaining object phrase, then combines shared verb/object logits with an atomic residual for classes actually seen in training. This supports a dataset split containing a novel verb-object composition while retaining the ordinary closed-set classifier for seen actions.

## Dataset-independent boundary

The model and trainer only consume the tensor contract and dimensions stored in `index.json`. Dataset-specific interpretation happens before the cache is written:

1. An annotation adapter emits a dense mutually-exclusive action timeline.
2. A procedure-schema JSON maps raw completion IDs or descriptions to canonical component names.
3. The cache builder aligns events and sensor/state annotations to feature timestamps.
4. The model receives only canonical arrays and does not contain IndustReal IDs.

IndustReal uses `configs/procedure_schemas/industreal_v1.json`. To add another dataset, provide the same canonical fields (`completion_components`, `state_components`, `event_outcomes`, and `raw_completion_map`) and implement or reuse an annotation reader that produces action segments and completion events. The causal temporal core, graph filter, losses, trainer, and live inference interface remain unchanged.

## IndustReal mapping

The supplied schema collapses the 24 observed raw PSR IDs into ten semantic completion components. Outcome is represented separately, so correct/incorrect/remove variants share a component. Exact same-frame completions become independent positive entries in the `T × 10` completion matrix.

The eleven columns in `PSR_labels_raw.csv` remain a separate `T × 11 × 3` state head. They are deliberately not forced into the ten-event taxonomy because the public raw-state file and completion-event vocabulary serve different annotation roles.

## Migration and validation

Existing v1 caches must be regenerated. Training fails early with an explanatory error when `schema_version != 2`, the label contract is missing, or a recording lacks the new arrays. This prevents an expensive 80-epoch run on invalid targets.

The frozen visual arrays in a v1 cache remain valid. Point both precomputed-feature options at the old cache root; the loader searches split subdirectories and selects the `motion` and `appearance` arrays by name. This rebuilds labels without decoding video or rerunning either encoder:

```bash
python -m aiops.features.industreal_cache \
  --data-root /path/to/industreal \
  --output-dir /path/to/stategraph_v2 \
  --motion-features-dir /path/to/stategraph_v1 \
  --appearance-features-dir /path/to/stategraph_v1
```

Before a full run:

1. Rebuild the cache with the v2 builder.
2. Inspect `index.json`: `schema_version` must be `2`, and `label_contract` must be `action_segmentation_plus_multicomponent_completion`.
3. Overfit two to four recordings as a wiring test.
4. Run a 20–30 epoch pilot and inspect action F1/Edit plus incorrect-event precision, recall, and F1.
5. Only then run with `--epochs 80 --patience 15`; 80 is a maximum controlled by early stopping, not a required stopping point.

The 2026-07-20 seed-7 pilot did **not** pass step 4: its selected checkpoint
reached 9.70 F1@50 but 0.00 incorrect-event F1. Do not launch the 80-epoch
configuration unchanged. See `stategraph_psr_v2_pilot_report.md` for the exact
run, metrics, and the required sampling/calibration experiments.

Final reporting should use at least three seeds and include action F1@10/25/50, Edit, frame accuracy, per-outcome event precision/recall/F1 with the documented ±1 cache-step tolerance, and component-state metrics.

Run the reusable cache gate before any experiment:

```bash
python -m aiops.data.audit_stategraph_cache \
  --cache-index /path/to/stategraph_v2/index.json \
  --output /path/to/stategraph_v2/audit.json \
  --fail-on-warnings
```

The AR/PSR separation follows the dataset authors' published annotation contract: `AR_labels.csv` stores action ID/description/start/end, while PSR files store completion timestamps and persistent component state. See the [official IndustReal repository](https://github.com/TimSchoonbeek/IndustReal).
