# VideoMAEv2 motion-cache upgrade

## Decision

The next controlled experiment keeps the validated 34.1M-parameter StateGraph
temporal head and replaces only its frozen Swin3D-S motion input. This isolates
representation quality from head capacity and directly tests the remaining
bottleneck observed in the v3/v4 pilots.

The preferred encoder is the official VideoMAEv2 ViT-giant checkpoint
fine-tuned on Something-Something V2
(`vit_g_hybrid_pt_1200e_ssv2_ft`). SSv2 is a strong domain match because its
labels depend on object manipulation and motion direction rather than scene
identity. The official model uses 16 frames with sampling interval 2 and reports
77.0 top-1 on SSv2. VideoMAEv2's UnlabeledHybrid pretraining also includes SSv2
with sampling interval 2.

The project cannot redistribute that fine-tuned checkpoint: the official model
zoo requires submitting its download-request form. The immediately runnable
fallback is `OpenGVLab/VideoMAEv2-giant` on Hugging Face. It is the public
1.0B-parameter, 1408-dimensional UnlabeledHybrid-pretrained encoder, not the
supervised SSv2 checkpoint. This distinction is stored in the feature manifest
and must remain explicit in experiment reports.

## Why this is preferable to making the head still larger

The large temporal head improved F1@50 from 15.32 to 18.05, but incorrect-event
F1 and PR-AUC remained zero. Its real training allocation was only about 2.4 GB
because visual features were already cached. Adding temporal parameters would
primarily increase fitting capacity over the same non-separable representation.
VideoMAEv2 instead changes the evidence supplied to both action and fault
branches, while frozen sequential extraction keeps the 24 GB training budget
practical.

## Reproducible extraction

Install the isolated feature dependencies:

```bash
python3 -m venv /home/aiops/.venvs/aiops-videomaev2
/home/aiops/.venvs/aiops-videomaev2/bin/pip install -e '/home/aiops/AIOps-stategraph[videomaev2]'
```

Run a one-recording alignment and memory smoke test first:

```bash
/home/aiops/.venvs/aiops-videomaev2/bin/python -m aiops.features.videomaev2_features \
  --data-root /home/aiops/AIOps/data/raw/industreal \
  --output-dir /home/aiops/AIOps/data/processed/videomaev2_giant_motion \
  --max-recordings 1 --precision bf16 --batch-size 1
```

Then rerun the same command without `--max-recordings`. Extraction is atomic
and resumable at recording granularity. It produces one `[time, 1408]` array per
recording and `videomaev2_manifest.json`. The default centers are every five
source frames, exactly matching the current StateGraph cache. Each center uses
16 frames at interval 2; video-boundary indices are clamped.
The public checkpoint and its remote model code are pinned to Hugging Face
revision `d27568eb41ccb2d41bb191fc2e3fe5aad74942d4`; override `--revision` only as
an intentional new experiment.

After obtaining the official SSv2 checkpoint, use a separate output directory:

```bash
/home/aiops/.venvs/aiops-videomaev2/bin/python -m aiops.features.videomaev2_features \
  --data-root /home/aiops/AIOps/data/raw/industreal \
  --output-dir /home/aiops/AIOps/data/processed/videomaev2_giant_ssv2_motion \
  --checkpoint /path/to/vit_g_hybrid_pt_1200e_ssv2_ft.pth \
  --precision bf16 --batch-size 1
```

The loader rejects missing or unexpected non-classifier weights, so a wrong
architecture cannot silently produce a cache.

## Rebuild the portable StateGraph cache

Reuse the existing ConvNeXt appearance features while swapping motion features:

```bash
python3 -m aiops.features.industreal_cache \
  --data-root /home/aiops/AIOps/data/raw/industreal \
  --output-dir /home/aiops/AIOps/data/processed/stategraph_v2_videomaev2_giant \
  --motion-features-dir /home/aiops/AIOps/data/processed/videomaev2_giant_motion \
  --motion-backend-name videomaev2_giant_unlabeledhybrid_16x2 \
  --appearance-features-dir /home/aiops/AIOps/data/processed/stategraph_v2_swin_convnext
```

The model reads dimensions from `index.json`, so no IndustReal-specific shape is
embedded in StateGraph-PSR. Other datasets can generate the same per-recording
arrays and retain the architecture unchanged.

## Experimental gate

1. Verify all 52 arrays, finite values, 1408 columns, and exact cache lengths.
2. Pass the four-recording overfit gate, including non-zero incorrect-event F1.
3. Run the same 20-epoch seed-7 large preset used by the Swin comparison.
4. Continue to three seeds only if action F1@50 or fault PR-AUC improves without
   regression in Edit and state accuracy.
5. Approve an 80-epoch early-stopped run only after the three-seed result.

Do not compare an SSv2-supervised run and the public UnlabeledHybrid run under
the same backend name. Do not use the test split for threshold selection.

## Primary sources

- [VideoMAE V2 paper (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html)
- [Official VideoMAEv2 repository](https://github.com/OpenGVLab/VideoMAEv2)
- [Official model zoo and SSv2 result](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)
- [Official feature-extraction implementation](https://github.com/OpenGVLab/VideoMAEv2/blob/master/extract_tad_feature.py)
- [Public VideoMAEv2-giant feature model](https://huggingface.co/OpenGVLab/VideoMAEv2-giant)
