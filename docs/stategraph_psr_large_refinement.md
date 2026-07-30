# StateGraph-PSR Large: causal refinement candidate

## Motivation

The v3 pilots used only about 0.8 GB of GPU memory because both visual encoders
were cached and the trainable temporal head had roughly 3.3 million parameters.
Action metrics were still improving at epoch 20--25, while the fault branch
could fit an overfit subset but did not generalize. This supports adding
structured temporal capacity, but not an unconstrained larger classifier.

The large candidate remains causal and keeps dataset-specific interpretation in
the procedure adapter. Old checkpoints remain loadable because every new branch
defaults to zero stages unless enabled explicitly.

## Architecture changes

| Module | Lite v3 | Large candidate |
|---|---:|---:|
| Hidden width | 192 | 384 |
| Shared temporal blocks | 8 | 12 |
| Attention heads | 4 | 8 |
| Action refinement | none | 2 stages x 4 causal blocks |
| Event/state refinement | shared features only | dedicated 4 causal blocks |
| Dropout | 0.20 | 0.15 |
| Trainable parameters | about 3.3M | about 34.1M |

Each action-refinement stage consumes the previous stage's class probability
trajectory, applies a causal multi-scale TCN/attention stack, and predicts a
residual correction. All refinement stages receive focal cross-entropy deep
supervision. This follows the useful principle behind multi-stage temporal
segmentation models while preserving online inference; no stage sees future
frames.

The event/state branch receives its own causal temporal blocks before component
state, completion, outcome, and normality predictions. This prevents the much
denser action objective from using all shared capacity and gives sparse faults a
specialized temporal representation.

The official pilot also increases causal context from 256 to 384 cached steps
(128 to 192 seconds at a 0.5-second cache stride), batch size from 2 to 4, and
guaranteed rare windows from one to two per batch. This uses more GPU while
retaining an effective batch of eight.

## Resource validation

On the RTX 4090, a synthetic BF16 forward/backward pass with batch 2 and length
256 used about 505 MB of PyTorch peak allocation. The real four-recording
training run reported about 1.8 GB total device memory, and the batch-4,
length-384 pilot reported about 2.4 GB. Cached encoders therefore remain the
dominant reason memory use is low; making the temporal head arbitrarily larger
would increase overfitting before exhausting 24 GB.

## Reproducible command

```bash
python3 -m aiops.training.train_stategraph_psr \
  --cache-index /home/aiops/AIOps/data/processed/stategraph_v2_swin_convnext/index.json \
  --output-dir /home/aiops/AIOps/runs/stategraph_psr_large_ctx384_pilot20_seed7 \
  --precision bf16 --batch-size 4 --accumulation-steps 2 \
  --sequence-length 384 --sequence-stride 288 \
  --epochs 20 --patience 10 --learning-rate 0.0002 \
  --hidden-dim 384 --num-temporal-blocks 12 \
  --attention-every 2 --num-heads 8 \
  --num-action-refinement-stages 2 --num-refinement-blocks 4 \
  --num-event-blocks 4 --dropout 0.15 --refinement-weight 0.5 \
  --rare-windows-per-batch 2 --calibration-interval 5
```

Use TensorBoard's **Scalars** view at
`http://127.0.0.1:6006/#scalars`; the newer Time Series frontend in the desktop
TensorBoard 2.14 environment fails to populate runs even though its APIs are
healthy.

## Acceptance criteria

The candidate must first retain the overfit fault gate, then improve official
validation action Edit/F1@50 without making the graph harmful. A larger model is
not considered successful solely because it consumes more memory. Incorrect
event precision/recall and normality AP remain hard gates, and the preferred
VideoMAEv2-SSv2 cache is still the higher-value representation upgrade.

## XXL profile and explicit temporal order

The repository now includes `configs/stategraph_psr_stage1_xxl.json` for systems
with spare memory. It widens the head to 640 channels, 20 shared temporal
blocks, four action-refinement stages, and eight event blocks. The recommended
batch is 2 with four-step gradient accumulation and 512 cached frames; this
preserves the same effective batch while increasing receptive field.

Both large and XXL profiles add a fixed sinusoidal temporal position signal
before the causal stack. Previously, order was represented only indirectly by
causal convolutions; attention could not distinguish identical local patterns
at different offsets. The encoding is non-trainable, supports up to 4096 cached
steps (`--max-sequence-length`), and adds negligible VRAM. Position dropout is
configurable (`--positional-dropout`) and defaults to zero for compatibility.

## Measured results

The four-recording, 15-epoch overfit gate passed:

| Metric | Large overfit |
|---|---:|
| Frame accuracy | 54.53 |
| Edit | 41.71 |
| F1@50 | 38.69 |
| Incorrect precision / recall / F1 | 100 / 100 / 100 |
| Incorrect normality AP | 100 |
| State accuracy | 91.42 |

The official 36-train/16-validation seed-7 pilot selected epoch 16:

| Metric | Lite v3 seed 17 | Large seed 7 |
|---|---:|---:|
| Parameters | 3.3M | 34.1M |
| Frame accuracy | 30.77 | **33.41** |
| Edit | 27.53 | **29.92** |
| F1@10 | 29.38 | **31.88** |
| F1@25 | 24.78 | **27.33** |
| F1@50 | 15.32 | **18.05** |
| Raw F1@50 | 15.06 | 17.86 |
| Incorrect F1 / PR-AUC | 0 / 0 | 0 / 0 |
| Remove F1 | 12.66 | 10.00 |
| Incorrect normality AP | 5.36 | 5.19 |
| State accuracy | 87.46 | **88.66** |

The larger causal model improves F1@50 by 2.73 points (17.8% relative), with a
small positive graph contribution of 0.18 point. This is a successful action
architecture refinement. It is not a successful fault detector: additional
head capacity did not create a separable fault representation. The evidence
therefore supports retaining the large model while upgrading the motion cache
and fault supervision before a final long run.
