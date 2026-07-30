# Reference implementation review: `ego_psr_repro` and `ego_psr_eval`

This note documents the two local reference folders supplied for review:

- `C:\Users\tenah\Desktop\ego_psr_repro`
- `C:\Users\tenah\Desktop\ego_psr_eval`

The folders are orchestration/evaluation layers around the original PSR research code. They do
not contain a new video backbone or a replacement temporal model. Their value is in the way they
make many PSR variants reproducible and comparable.

## 1. `ego_psr_repro`: dependency-aware reproduction DAG

The entry point is `repro.sh`, which dispatches `orchestrate.py`. The orchestrator defines a
registry of architectures and a directed acyclic graph of stages:

```text
provision -> labels -> frozen feature extraction -> optional CPU fusion -> head training -> evaluation
```

The registry includes Huge+ASFormer, SSv2-giant+ASFormer, giant+InternVideo2 fusion, DiffAct,
causal GRU, TeSTra, and MECCANO variants. Shared stages are declared once and reused. For example,
the SSv2-giant feature extraction is shared by the SSv2 ASFormer head, DiffAct, fusion extraction,
and the MECCANO branch.

Important implementation details:

- Dry-run is the default; `--submit` is required before any SLURM jobs are launched.
- Dependencies are converted into `afterok` SLURM dependencies.
- Existing outputs are detected with paths/globs, making reruns idempotent.
- GPU extraction is separated from CPU fusion and CPU evaluation.
- Provisioning distinguishes auto-downloadable assets from gated or side-loaded weights.
- The pipeline is an adapter over existing project scripts rather than a reimplementation.

This is a strong experiment-management pattern: one expensive frozen feature cache can support
many temporal heads, and the exact command/configuration for each result is explicit.

## 2. `ego_psr_eval`: common evaluation and reporting harness

`run.sh` activates the project environment, optionally starts `gpu_monitor.py`, invokes
`evaluate.py`, renders plots with `plot.py`, and prints a compact summary. `evaluate.py` contains a
registry with a normalized record for each architecture (name, group, kind, config, data path,
model directory, and epoch).

The evaluator dispatches by model kind:

- `step`: invokes the original `eval_step.py`, parses accuracy, Edit, and F1@10/25/50 for raw and
  Viterbi decodes.
- `type`: invokes `eval_type.py` and extracts per-class support, precision, recall, and F1 for
  none/correct/incorrect/remove.
- `diffact`: recomputes metrics from cached predictions on CPU, with a saved-NPY fallback.
- `rt`: invokes online evaluation for multiple latency lags and reports the best F1/latency point.

Every architecture gets a preflight artifact check, a per-architecture log, a status (`ok`,
`missing`, or `error`), elapsed time, and structured JSON output. The plotting layer produces:

- offline segmentation F1@50 comparison;
- all segmentation metrics;
- correctness/fault recall;
- streaming F1 versus latency;
- GPU utilization and memory timelines.

The GPU sampler prefers `nvidia-smi` and falls back to `rocm-smi`, writes a flush-on-each-sample
CSV, and remains safe on CPU-only hosts.

## 3. What is worth borrowing for StateGraph-PSR

### Adopt directly

1. **Architecture registry and normalized result schema.** Add a registry for StateGraph-PSR
   variants (Lite, XL, XXL, motion-only, dual-motion, and future VLM-assisted variants). Store
   config, checkpoint, dataset split, git commit, and metrics in one JSON record.
2. **Shared feature-cache spine.** Keep frozen visual encoders outside the trainable head and make
   cache identity explicit (`backbone`, weights hash, frame stride, crop, feature dimension).
3. **Preflight and resumability.** Before training/evaluation, verify labels, cache manifests,
   class mappings, and checkpoints. Skip complete stages, but fail loudly on stale/incompatible
   feature metadata.
4. **Metric separation.** Continue reporting segmentation (Edit/F1), correctness/error recall,
   and streaming latency as separate metric families. Do not hide rare-error performance inside
   overall frame accuracy.
5. **CPU post-processing.** Keep Viterbi/procedure-graph decoding and metric computation off the
   GPU where possible; this makes repeated comparison cheap.
6. **GPU telemetry format.** Reuse the per-GPU CSV idea in the reusable dashboard, including a
   `source` column for NVIDIA/ROCm and explicit missing values.

### Borrow with changes

- **ASFormer/DiffAct/GRU comparison:** use these as baselines and ablations, but route them through
  the same StateGraph-PSR label contract so comparisons are not confounded by different label
  definitions.
- **Viterbi decoding:** retain it as a constrained decoder, but combine it with the learned
  transition graph and uncertainty thresholding. A hard decoder alone cannot express a novel fault
  or recovery action.
- **Latency sweeps:** evaluate causal windows and look-ahead values, but report alert delay,
  false-alert rate, and time-to-correct in addition to F1.
- **SLURM DAG:** the dependency model is useful on a multi-GPU server; on a single 24 GB GPU,
  expose the same stages through local commands and checkpoint manifests.

## 4. Gaps in the references that our implementation should address

The reference folders do not implement mistake taxonomy, future-state prediction, recovery
recommendation, calibration, or a learned procedure state graph. They also assume the original
research directory layout and hard-coded cluster paths. The orchestration is therefore excellent
for reproducing offline benchmark numbers but is not yet a deployable live assistant.

Our StateGraph-PSR head adds the missing pieces: multimodal feature gates, causal temporal blocks,
explicit positional encoding, action/state/event heads, graph-constrained transitions, uncertainty
signals, and configurable Lite/XL/XXL capacity. The next integration step should be a small
`experiments/registry.json` plus a common evaluator that consumes StateGraph-PSR predictions and
the reference baselines using the same split and label mapping.

The supplied evaluation logs also expose two useful baselines: Viterbi decoding raised SSv2
F1@50 from about 59.4 to 66.5 in the reported penalty sweep, while the fusion type head reached
about 10.1% incorrect-install recall versus 0% for the SSv2-only type head. The causal GRU remained
well below the offline models (roughly 37.4 F1@50 at its best lag), so a longer-memory causal
temporal head is justified for live deployment.

## 5. Recommended comparison experiments

1. Reproduce the reference SSv2-giant+ASFormer and fusion+ASFormer scores from their evaluator.
2. Train StateGraph-PSR XL on the same cached features and split.
3. Add positional encoding, graph loss, event/outcome heads, and modality ablations one at a time.
4. Compare macro-F1 and per-error recall, not just frame accuracy.
5. Run the causal latency sweep and measure alert delay and false-alert rate.
6. Freeze the best checkpoint and test on error types held out during training.

The result should be a fair, auditable comparison: identical data splits and cached features where
possible, explicit model/config metadata, and separate reporting of segmentation, correctness,
anticipation, and recovery metrics.
