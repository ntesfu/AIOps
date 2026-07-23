# Assembly101 ego-30 StateGraph-PSR methodology

## Provenance

The experiment uses the authenticated official Hugging Face dataset
`cvml-nus/assembly101`, pinned to revision
`bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967`. The deterministic manifest is
`data/raw/assembly101/ego30_manifest.json`; it selects 98 `e1` egocentric videos
and records source paths, actors, and split membership. Assembly101 is CC BY-NC
4.0 and this work is for non-commercial research.

## Split and controls

- Train actors: 9021, 9022, 9023, 9025, 9035, 9044, 9045, 9054.
- Validation actors: 9031, 9074.
- Test actors: 9013, 9083.
- Recordings: 73 train, 15 validation, 10 test.
- Only the `e1` egocentric view is used.
- Test is evaluated only after architecture selection and validation threshold
  calibration; test thresholds are never recalibrated.

## Sampling and representation

Official ego videos are 60 fps and annotations use a 30 fps frame clock. The
decoder selects the nearest source frame at each point on that 30 fps clock and
rejects sources below 30 fps. A causal clip contains the current frame and 31
past frames. Features are cached every 8 annotation frames, yielding 3.75 model
observations per second while the video backbone still sees genuine 30 fps
motion.

The motion stream is a frozen Kinetics-400 Swin3D-S embedding. The appearance
stream is a frozen ConvNeXt-Tiny embedding of the current clip frame. Both are
768-dimensional. Resizing is performed once per decoded frame before overlap
reuse. Cache generation is sharded and resumable, and merge validates schema,
shard uniqueness, shapes, and metadata.

## Labels

The model predicts the complete official 203-class coarse action taxonomy,
including background. Official verb and noun columns provide compositional
action factors, including multiword verbs such as `attempt to attach`.

Completion events have three outcomes: correct, incorrect, and remove. There
are 59 aligned completion/state components. Persistent state classes are
incorrect, pending, and correct. Components not relevant to a recording are
masked instead of being falsely labelled pending.

## Training and evaluation

Training uses actor-disjoint validation, deterministic epoch-varying temporal
crops, square-root action balancing, outcome-aware rare-window batches, focal
action/state losses, and asymmetric sparse-event losses. The model has a hard
23 GiB process VRAM guard.

Mistake evidence combines four causal signals: completion/outcome probability,
positive transitions in component state, component-normality evidence, and
component-mapped probability of official `attempt to ...` actions. Event NMS is
per outcome so a failed installation and nearby successful retry cannot erase
one another. Event thresholds and the rare incorrect-state threshold are
calibrated on validation and frozen for test.

The primary point-event metric requires the correct component and a timestamp
within approximately one second. Wider ±2 s and ±4 s diagnostics are also
reported. Mistake-normality average precision measures ranking incorrect versus
correct install events and must be reported separately from exact point-event
F1.

See `docs/assembly101_stategraph_80epoch_report.md` for the final command,
metrics, limitations, checkpoint hash, and VRAM measurement.
