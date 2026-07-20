# Assembly101 StateGraph-PSR methodology

## Scope and provenance

The complete Assembly101 visual release is hosted in the gated `cvml-nus/assembly101` Hugging Face repository. The training host had no accepted Hugging Face credentials, so this experiment uses a public, derived frame mirror: `kimihiroh/promqa-assembly-frames`, pinned to revision `da4deb832d0e39b4b309eb2260bbb41932d64621`. Labels come only from the official `assembly-101/assembly101-mistake-detection` repository at revision `6f3a953267ffb86cdeabf6751af05a75a011a4f8`.

The committed `data/raw/assembly101/subset_manifest.json` records URLs, expected byte sizes, LFS SHA-256 object IDs, split membership, and source revisions. `scripts/download_assembly101_subset.py` downloads all official mistake CSVs and the selected visual archives, resumes interrupted transfers, and rejects files that fail size or SHA-256 verification.

Assembly101 is CC BY-NC 4.0. This pipeline and its resulting model are for non-commercial research.

## Selection and split

- 60 recordings from the public mirror's 227-recording intersection with the 328 official mistake-annotated sequences.
- One consistent fixed RGB camera, `C10095`, at one frame per second.
- Actor-disjoint split: 40 train recordings from 8 actors, 10 validation recordings from 2 actors, and 10 test recordings from 2 actors.
- 728 annotated action segments: 511 correct, 142 mistakes, and 75 corrections.
- Train/validation/test contain 98/20/24 mistake segments and 56/9/10 correction segments respectively.
- Validation includes four action compositions absent from train; test includes five. This is reported separately rather than silently folded into seen-class accuracy.

The mirror is a fixed-view derivative, not the official egocentric stream. Conclusions therefore apply to procedural state and mistake modeling under this view and should not be presented as an official Assembly101 benchmark result.

## Label mapping

The official part-to-part rows provide 30 fps start/end timestamps, a verb, two working objects, `correct` / `mistake` / `correction`, and an optional mistake remark. Mirror frame numbers are one-frame-per-second indices, so the adapter converts them to the official 30 fps clock before alignment.

- Action class: coarse `verb + subject part` (87 classes including background in this subset).
- State/completion component: subject part (55 components).
- Correct attachment or positioning: `correct` completion and persistent `correct` state.
- Mistaken action: `incorrect` completion and, for attachment/position mistakes, persistent `incorrect` state.
- Corrective detach: `remove` event followed by `not_completed` state.
- Unnecessary detach remains an `incorrect` event, followed by the physically accurate `not_completed` state.

Sub-second corrections occasionally share a one-second mirror frame with the preceding mistake. The adapter preserves their causal order by assigning the later event to the next available row rather than overwriting either label.

## Visual representation and evaluation controls

ConvNeXt-Tiny ImageNet-1K features are extracted once from every available frame. The appearance stream uses the frozen 768-dimensional embedding; the motion stream uses its causal adjacent-frame difference. No visual feature is synthesized from labels. Sensor input is masked because this mirror has no hand, gaze, depth, or pose streams.

Model selection uses validation action segmentation and incorrect-event F1. Event thresholds are calibrated on validation only and then frozen for the actor-held-out test split. The exported checkpoint contains those thresholds. The standalone evaluator refuses test evaluation when a checkpoint lacks validation-calibrated thresholds.

Final metrics and the exact 80-epoch command are recorded in `docs/assembly101_stategraph_80epoch_report.md` after training.
