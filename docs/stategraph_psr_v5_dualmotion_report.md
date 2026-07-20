# StateGraph-PSR Stage 1 v5: dual-motion XL report

## Decision

Use v5 as the strongest current **action/step recognition** candidate. It beats
the previous 34.1M large baseline on the untouched recording-level validation
split. Do not yet represent it as a validated incorrect-step alarm and do not
spend the final 80-epoch budget on a single seed: incorrect-event F1 is still
zero on the five validation incidents, despite better anomaly ranking.

## Why the architecture changed

The public VideoMAEv2-giant cache and the Swin3D-S cache proved complementary.
Early concatenation improved Edit but blurred which encoder should be trusted.
v5 therefore projects VideoMAEv2 motion, Swin motion, ConvNeXt appearance, and
sensors independently. A learned four-way gate supplies the action stream. The
weighted cross-stream variance is preserved as a disagreement feature for a
separate six-block event/state branch; a wrong tool, part, or connection can be
visible as disagreement even when an averaging gate suppresses it.

The 14 incorrect training events were previously single-row outcome targets.
v5 repeats only the outcome/normality target for two following cache rows. This
is causal and increases rare supervision. Completion remains an exact point
target so the alert is trained at the annotated instant rather than after it.

The task-graph filter was also changed from a Python recurrence to a vectorized
first-order causal prior based on the preceding visual evidence. It never uses a
future row. This removed hundreds of serialized CUDA launches and reduced a
nine-epoch XL overfit interval from minutes to about 40 seconds.

## Architecture and portability

- Frozen VideoMAEv2-giant: 1408-d semantic/motion representation.
- Frozen Swin3D-S: 768-d complementary local-motion representation.
- Frozen ConvNeXt-Tiny: 768-d keyframe appearance.
- Optional 128-d sensor stream.
- Four independent 512-d stems and a learned modality gate.
- Cross-modal weighted variance injected only into the fault/state branch.
- Sixteen causal multi-scale blocks with sparse causal attention.
- Three five-block causal action refinement stages.
- Six causal event/state blocks.
- Factorized verb/object action prototypes and bounded procedure-graph prior.
- 87,519,149 trainable head parameters; frozen video backbones stay outside the
  training graph.

All dimensions are read from cache metadata. `motion_aux` is optional and older
single-motion caches still load, so a new dataset needs only its own annotation
adapter, procedure schema, and aligned feature arrays.

![StateGraph-PSR v5 architecture](assets/stategraph_psr_stage1_architecture.svg)

## Experiments

All comparisons use the official 36-train / 16-validation recording split. The
test split remains untouched.

| Candidate | Params | Frame acc. | Edit | F1@50 | Incorrect AP | Incorrect F1 | State acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Swin large v4 | 34.1M | 33.41 | 29.92 | 18.05 | 5.19 | 0.00 | 88.66 |
| VideoMAEv2 only | 34.3M | 30.06 | 29.18 | 15.05 | 6.72 | 0.00 | 90.31 |
| Early concat VMAE + Swin | 34.6M | 31.41 | 31.26 | 15.66 | 5.81 | 0.00 | 89.83 |
| Gated dual motion v4 | 34.8M | 32.77 | 30.92 | 17.73 | 5.59 | 0.00 | 89.36 |
| **v5 XL + disagreement + outcome horizon** | **87.5M** | **34.99** | **34.26** | **19.12** | **8.82** | **0.00** | **90.37** |

The v5 checkpoint was selected at epoch 24 of a 25-epoch seed-7 pilot. Relative
to v4 Swin, it gains +1.58 frame accuracy, +4.34 Edit, +1.07 F1@50, +3.63
incorrect normality AP, and +1.71 state accuracy.

The corrected four-recording wiring gate reached 34.06 F1@50, 100% incorrect
precision, 50% recall, 66.67 incorrect F1, 100 incorrect AP, and 91.79 state
accuracy after 20 epochs. This proves that labels, loss, decoder, and exact event
timing can fit fault examples. The official-split gap is therefore a
generalization/data-support problem, not a broken branch.

## Compute and dashboard

The official BF16 pilot peaked at 3.17 GB PyTorch allocation and about 4.68 GB
device use on the 24 GB RTX 4090. Sampled utilization averaged 15.3% and peaked
at 37%; the remaining under-utilization is dominated by the very small 58-window
training set and CPU event-metric calibration, not model memory pressure.

TensorBoard reads `/home/aiops/AIOps/runs`. From the original PC use the SSH
tunnel and open `http://127.0.0.1:6006/#scalars`. The run names are:

- `stategraph_psr_xl_outcomehorizon_overfit20_seed7`
- `stategraph_psr_xl_outcomehorizon_pilot25_seed7`

## Reproduce the selected pilot

```bash
PYTHONPATH=src python -m aiops.training.train_stategraph_psr \
  --cache-index /home/aiops/AIOps/data/processed/stategraph_v2_dualmotion_videomaev2_swin/index.json \
  --output-dir /home/aiops/AIOps/runs/stategraph_psr_xl_outcomehorizon_pilot25_seed7 \
  --epochs 25 --patience 10 --batch-size 4 --accumulation-steps 2 \
  --sequence-length 384 --sequence-stride 288 \
  --hidden-dim 512 --num-temporal-blocks 16 --attention-every 2 --num-heads 8 \
  --num-action-refinement-stages 3 --num-refinement-blocks 5 --num-event-blocks 6 \
  --dropout 0.15 --learning-rate 0.0002 --weight-decay 0.002 \
  --rare-windows-per-batch 2 --event-label-horizon 2
```

## Gate for the long run

1. Repeat the 25-epoch pilot with seeds 17 and 29.
2. Require the mean action F1@50 and Edit to exceed v4, not only seed 7.
3. Require non-zero incorrect precision and recall in at least two seeds, and
   report PR-AUC/normality AP because only five validation incidents exist.
4. Manually inspect every incorrect prediction and validation incident.
5. Prefer obtaining the official SSv2-fine-tuned VideoMAEv2 checkpoint or adding
   more sharply annotated incorrect sequences before the final comparison.
6. Only then run `--epochs 80 --patience 15`; 80 is a ceiling governed by early
   stopping, not a result target.

## Primary evidence

- [VideoMAE V2 paper (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html)
- [Official VideoMAEv2 repository](https://github.com/OpenGVLab/VideoMAEv2)
- [Official model zoo and SSv2 checkpoint](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)
- [Official temporal feature extraction](https://github.com/OpenGVLab/VideoMAEv2/blob/master/extract_tad_feature.py)
- [Public UnlabeledHybrid VideoMAEv2-giant](https://huggingface.co/OpenGVLab/VideoMAEv2-giant)
