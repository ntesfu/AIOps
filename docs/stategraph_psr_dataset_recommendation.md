# Dataset Recommendation for StateGraph-PSR

## Executive decision

There is no single public dataset that adequately covers all of StateGraph-PSR: online step segmentation, hand/tool/part relations, explicit assembly state, fine-grained execution mistakes, open-set errors, future prediction, and safe recovery. The best scientifically defensible choice is a **paired benchmark**:

1. **Primary training and architecture-development dataset: Assembly101 + its Mistake Detection annotations.**
2. **Primary industrial transfer and final public test dataset: IndustReal v2 + the 2024 Assembly State Recognition with Error States extension.**

If only one dataset can be used for the first implementation, choose **Assembly101**. It is the only mature public assembly dataset with enough scale to train the temporal and graph modules, official temporal segmentation and anticipation tasks, multi-view egocentric data, 3D hand pose, natural procedural variation, and explicit `correct / mistake / correction` annotations with part-to-part relations.

However, Assembly101 is licensed CC BY-NC 4.0. It is suitable for research and internal model comparison, but the data cannot be used as the training source of a commercial production model without separate permission. IndustReal is Apache-2.0 and much closer to the target problem, but its 84 videos are too small to carry the full architecture alone.

The recommended development sequence is therefore:

`Assembly101 pretraining/development -> IndustReal transfer and public evaluation -> proprietary aerospace fine-tuning and acceptance test`

## Why the recommendation is a pair

StateGraph-PSR has six distinct data requirements.

| Requirement | Assembly101 | IndustReal + error-state extension |
|---|---|---|
| Continuous action/step segmentation | Excellent: >1M fine segments and >100K coarse segments | Good: AR intervals and PSR completion labels |
| Hand/part interaction graph | Excellent hand pose; part-to-part attach/detach labels; multi-view geometry | Good hand tracking, CAD geometries and assembly-state boxes |
| Physical state and state delta | Derivable from attach/detach and part graph, but not densely provided for all frames | Excellent: 11-component state vector with `-1/0/1`, intended state and error categories |
| Natural procedural variation | Excellent: free assembly/disassembly order and unseen toys | Good: 22 correct procedure orders, assembly and maintenance |
| Mistakes and corrections | Excellent: 328 sequences with correct/mistake/correction and causal remarks | Excellent mistake diversity and held-out errors, but little explicit recovery supervision |
| Action anticipation | Official benchmark | Must be derived from the task graph |
| Open-set execution-error test | Limited; mistakes emphasize order/position and unnecessary actions | Excellent: many errors occur only in validation/test |
| Industrial similarity | Toy assembly but mechanically relevant | Industrial-like egocentric assembly/maintenance |
| Data volume | 513 hours / 4,321 view-specific videos | 84 videos / 27 participants |
| License | CC BY-NC 4.0 | Apache-2.0 |

Assembly101 supplies the **scale and recovery trajectories**. IndustReal supplies the **state semantics and unseen-error test**. Combining them mirrors the architecture: learn motion, interaction and procedure dynamics at scale, then learn and evaluate explicit physical correctness in the closest available domain.

## Primary dataset 1: Assembly101

### What is available

Assembly101 contains 362 synchronized assembly/disassembly recordings represented by 4,321 view-specific videos. Each recording has eight fixed cameras and four egocentric cameras. Across all views it contains 513 hours of video, more than one million fine-grained action segments, more than 100,000 coarse segments, and 18 million 3D hand-pose annotations. The official paper reports 1,380 fine-grained action classes formed from 24 verbs and 90 objects.

As of May 2026, the official distribution has moved to a gated Hugging Face repository. Access requires accepting the dataset conditions and sharing contact information. The repository includes raw recordings, annotations, cleaned hand poses, hand-object crops, official TSM features and DINOv2 full-frame/hand-crop features. This makes it possible to validate the pipeline without downloading or re-encoding every raw view first.

The official split is deliberately challenging:

| Split | Hours | Videos | Fine segments | Unseen/shared toys |
|---|---:|---:|---:|---:|
| Train | 287.6 | 2,526 | 566,855 | 40 / 26 |
| Validation | 96.6 | 740 | 186,788 | 16 / 18 |
| Test | 128.8 | 1,055 | 259,880 | 20 / 20 |

The 2023 Mistake Detection extension covers 328 assembly sequences. Each action includes:

- start and end timestamps;
- `attach` or `detach` verb;
- two interacting parts (`this`, `that`);
- `correct`, `mistake`, or `correction` label;
- causal remarks including `wrong order`, `previous one is mistake`, `shouldn't have happened`, and `wrong position`.

This is unusually well aligned with StateGraph-PSR. The two part fields directly define an edge update in the assembly graph, and corrections supply real recovery transitions rather than synthetic undo examples.

### Mapping to StateGraph-PSR

| Dataset field | StateGraph-PSR target |
|---|---|
| Fine action `(verb, object)` | Atomic action and motion head |
| Coarse action segment | Procedure step and temporal segmentation head |
| `attach/detach(this, that)` | Relation-graph edge delta |
| 3D hand pose and hand-object crop | Hand stream, interaction ROI and occlusion features |
| Four ego + eight exo views | View-consistency pretraining; ego-only deployment test |
| Mistake label | Known mistake detector |
| Mistake remark | Hierarchical subtype: order, position, unnecessary action, propagated error |
| Correction label | Recovery edge and time-to-recovery target |
| Next action benchmark | Future-action prediction at 1/3/5-step horizons |
| Toy identity and official unseen-toy split | Object/task generalization test |

### What must be derived

Assembly101 does not provide the complete StateGraph schema directly. Build these derived labels:

1. **Assembly graph state.** Replay every correct `attach/detach(part_a, part_b)` operation to maintain a graph `G_t`.
2. **Expected graph.** Learn or construct the set of valid predecessor graphs for each toy from correct runs.
3. **State delta.** The edge added or removed by the current action.
4. **Procedure belief.** Valid next operations given the current graph and the partial-order constraints learned from normal runs.
5. **Effect residual.** Difference between the predicted edge delta and the annotated edge delta.
6. **Recovery target.** For a mistake, collect the sequence of `correction` actions until the graph re-enters a valid normal state.
7. **PNR proxy.** Use the end of the first incorrect attach/detach action as the initial point-of-no-return target. Later refine using contact/effect annotation on a subset.

### Important limitations

- The objects are snap-together toy parts; there is little tool diversity and no torque, electrical continuity, adhesive curing or tolerance verification.
- Mistake labels are strongest for ordering, position, unnecessary detach and propagated mistakes. They do not adequately cover wrong tool, wrong fastener, under-torque, contamination, damage or subtle incomplete insertion.
- The dataset is long-tailed and contains overlapping fine actions. The model must not treat every fine label as a distinct aerospace step.
- The non-commercial license is a major constraint for the eventual deployed product.

## Primary dataset 2: IndustReal v2 and the error-state extension

### Why it is the final public benchmark

IndustReal is the closest public match to the target use case. It contains 84 HoloLens-2 recordings from 27 participants performing assembly and maintenance on a reproducible construction-toy car. It provides RGB, stereo, depth, ambient light, gaze, hand tracking and head pose, together with CAD geometries for every part.

Its labels cover three complementary tasks:

- action recognition intervals;
- assembly state detection in COCO format;
- procedure-step completion, including incorrectly completed steps.

The raw procedure state is an 11-component vector. Each component is `1` when correctly installed, `0` when not yet completed, and `-1` when incorrectly installed. The paper reports 22 defined correct states and 27 error states; 26.9K frames are annotated for assembly-state detection, including 3,569 error-state frames.

The participant split is 12/5/10 for train/validation/test. The split is by participant rather than by clip, and a substantial subset of error types is present only in validation/test. This makes it valuable for evaluating open-set error detection and domain generalization.

The 2024 extension adds the user-intended state and a fine-grained category for error frames:

1. missing component;
2. incorrect component orientation;
3. incorrect component placement;
4. part-level error inside a subassembly.

### Mapping to StateGraph-PSR

| Dataset field | StateGraph-PSR target |
|---|---|
| AR temporal interval | Action and causal segmentation head |
| PSR completion event | Step completion and completion-delay target |
| 11-value raw state | Sparse state predicates and observed state delta |
| Intended state | Expected state and effect residual |
| Error type extension | Execution/spatial mistake subtype |
| COCO state boxes | State ROI and evidence-localization supervision |
| Hand/gaze/head tracking | Interaction and attention features |
| CAD geometry | Synthetic state/error generation and pose anchors |
| Held-out errors | Open-set AUROC, risk-coverage and abstention test |

### Important limitations

- Eighty-four videos are insufficient for learning a large video representation from scratch.
- Only one product family is represented.
- State labels emphasize stable completed states; manipulation/contact phases require additional derivation or annotation.
- Recovery actions are not richly labeled. The original procedures include assembly and maintenance, but there is no general expert-approved recovery graph.
- Depth is low resolution and 5 FPS; RGB and stereo are 10 FPS.

## Secondary datasets and when to use them

### HoloAssist — best for intervention and explanation

HoloAssist provides approximately 169 hours from 350 instructor-performer pairs doing physical manipulation tasks. It includes RGB, depth, hand pose, gaze, IMU, action correctness, mistake explanations, verbal correction links and intervention-type benchmarks.

Use it for:

- event-triggered VLM adjudication;
- mistake explanations;
- deciding whether and how to intervene;
- hand forecasting;
- learning recovery-language templates.

Do not use it as the main StateGraph training set because object-state and part-connection labels are not uniform across tasks. Its compressed videos are about 145 GB; labels are only 111 MB, so metadata-first experimentation is practical. The dataset is released under CDLAv2.

### MECCANO — best auxiliary source for hands, active tools and future objects

MECCANO contains 20 egocentric assembly recordings of a toy motorbike with RGB, depth and gaze. It includes 8,857 temporal action segments, 64,349 active-object boxes, 89,628 hand boxes, 48,024 next-active-object boxes, 12 verbs, 20 objects and 61 action classes. The object set includes a screwdriver and wrench.

Use it to pretrain or validate:

- hand/tool/part detection;
- egocentric human-object interaction;
- action anticipation;
- next-active-object prediction.

Do not use it as the main dataset: there are only 20 videos and no strong mistake/recovery annotation.

### CaptainCook4D — best auxiliary source for error taxonomy and task graphs

CaptainCook4D contains 384 egocentric recordings totaling 94.5 hours, 5.3K step annotations and 10K fine-grained action annotations. It includes normal and deliberately erroneous executions, task graphs, and errors involving order, preparation/tool choice, measurement/quantity and technique. It is Apache-2.0.

Use it to pretrain:

- hierarchical mistake types;
- procedure-graph alignment;
- zero-shot/open-set error representations;
- quantity and wrong-tool semantic heads.

Its cooking domain is visually far from aerospace assembly, so its main value is semantic and procedural rather than direct visual transfer.

### Ego-Exo4D v2 — high-value but not a first-iteration dependency

Ego-Exo4D v2 contains 1,286 video hours across 5,035 takes, synchronized ego/exo views, hand pose, object masks, keysteps, procedural dependencies, expert commentary and proficiency labels. Bike repair is particularly relevant.

It is valuable for large-scale multi-view and procedural pretraining, but the default download is about 12 TB and the full release is about 38 TB. Access requires a license agreement. Use only a filtered ego-view subset such as bike repair, keystep, relations and task-graph benchmarks after the first implementation is stable.

### IKEA ASM — useful object-state supervision, weak mistake coverage

IKEA ASM provides 371 furniture assemblies with three RGB views, depth, 33 atomic verb-object actions, pose, part masks and part tracking. It is useful for object segmentation and structural-state transfer, but it is exocentric and lacks a mistake/recovery benchmark. It is also CC BY-NC 4.0.

### Ego-Exo Manufacturing (2026) — promising pilot data, not yet a benchmark

The new Apache-2.0 Ego-Exo Manufacturing dataset contains 40 synchronized pairs plus standalone ego/exo sessions, timestamped action descriptions, keysteps, dependencies, proficiency, common failure modes and some mistake labels. It is highly domain-relevant, but currently small, recently released and not supported by a peer-reviewed benchmark. Use it only as an auxiliary transfer sanity check until annotation coverage and evaluation protocols mature.

## Dataset scorecard

Ratings are qualitative for the specific StateGraph-PSR objective.

| Dataset | Assembly domain | Temporal scale | Part/state graph | Mistakes | Recovery/future | Ego deployment | Legal fit | Role |
|---|---|---|---|---|---|---|---|---|
| Assembly101 + mistakes | Excellent | Excellent | Good | Very good | Excellent | Excellent | Research only | Core training |
| IndustReal + extension | Excellent | Limited | Excellent | Excellent | Moderate | Excellent | Excellent | Transfer + final public test |
| HoloAssist | Good | Very good | Moderate | Very good | Very good | Excellent | Good | Intervention/VLM |
| MECCANO | Excellent | Limited | Good | Weak | Good | Excellent | Verify terms | Tool/object branch |
| CaptainCook4D | Low | Good | Moderate | Excellent | Good | Excellent | Excellent | Error semantics |
| Ego-Exo4D | Moderate | Excellent | Excellent | Moderate | Excellent | Excellent | Agreement required | Later pretraining |
| IKEA ASM | Good | Moderate | Excellent | Weak | Weak | Weak | Research only | State/part auxiliary |
| Ego-Exo Manufacturing | Excellent | Limited | Moderate | Limited | Moderate | Excellent | Excellent | Emerging auxiliary |

## Recommended training protocol on one 24 GB GPU

### Stage 0 — metadata and graph audit

- Download annotations before raw video.
- Build a canonical ontology: `verb, part, target, tool, correctness, mistake_type, correction_of`.
- Verify that all 12 Assembly101 views belonging to one physical sequence stay in the same split.
- Preserve the official unseen-toy split and IndustReal participant split.
- Create graph-replay tests that guarantee every annotation produces a valid deterministic state transition.

### Stage 1 — Assembly101 temporal and graph pretraining

- Use one egocentric view per sequence for the first baseline; rotate among the four views across epochs.
- Use synchronized exocentric views only for training-time consistency or teacher supervision.
- Cache VideoMAEv2-B/L or V-JEPA features at 2–4 updates per second.
- Train the causal temporal head, action tokens, relation-graph encoder and next-action head on cached features.
- Train hand/interaction modules using the supplied hand pose and hand-object crops.
- Add the 328-sequence mistake/correction annotations only after the normal graph decoder is stable.

This route avoids retaining gradients through a large video backbone and is comfortable on 24 GB. The first experiment can also use the official TSM or DINOv2 features to validate loaders and labels before expensive extraction.

### Stage 2 — IndustReal state specialization

- Map the 11-component state vector into component predicates.
- Fine-tune the state encoder, effect head and evidence localizer using RGB, hand tracks and state boxes.
- Use CAD renders to generate correct states and controlled missing/orientation/placement negatives.
- Keep the video backbone frozen initially; adapt only the last block or LoRA after the heads converge.
- Train normal prototypes only on correct train states; do not expose the open-set branch to held-out validation/test errors.

### Stage 3 — joint multi-domain training

Alternate batches rather than merging label IDs:

- Assembly101 batches optimize action, segmentation, relation delta, procedure belief, anticipation and recovery.
- IndustReal batches optimize action, completion, component state, effect residual, mistake/open-set and evidence localization.
- Dataset-specific heads map to a shared semantic latent space.
- Use gradient balancing so the much larger Assembly101 dataset does not erase the IndustReal state objective.

### Stage 4 — optional auxiliary curricula

- MECCANO: tool/part detector and next-active-object head.
- CaptainCook4D: hierarchical mistake semantics and task-graph pretraining.
- HoloAssist: intervention type and natural-language correction.
- Ego-Exo4D bike repair: long-horizon procedural pretraining after more storage/compute is available.

## Evaluation design

### Benchmark A — Assembly101 in-domain development

Report separately for seen and unseen toys:

- F1@10/25/50, edit and MoF for segmentation;
- action and part-pair accuracy;
- next-action top-1/top-3 at 1/3/5 actions;
- graph-edge delta F1;
- correct/mistake/correction macro-F1;
- wrong-order and wrong-position recall;
- recovery success and number of actions to re-enter a valid graph;
- causal alert delay and false alerts per hour.

### Benchmark B — IndustReal public transfer test

Preserve the official participant split. Report:

- PSR F1/edit/POS and completion delay;
- 11-component state macro-F1;
- assembly-state retrieval/verification accuracy;
- seen versus unseen error AUROC/AUPR;
- recall for missing, orientation, placement and part-level errors;
- evidence localization IoU/pointing accuracy;
- calibration ECE and risk-coverage under abstention;
- p50/p95 streaming latency and peak VRAM.

### Benchmark C — cross-dataset zero-shot test

Before IndustReal fine-tuning, evaluate the Assembly101-trained model on IndustReal using semantic roles and graph deltas rather than dataset-specific class IDs. This measures whether the architecture learned procedure structure instead of toy identity.

### Benchmark D — proprietary aerospace acceptance set

No public dataset can validate aerospace deployment. The final locked test set must hold out:

- operators;
- work shifts/days;
- camera units and calibration;
- product variants and serials;
- lighting and occlusion conditions;
- exact staged error templates;
- recovery sequences.

At minimum it must include wrong tool, wrong part/fastener, wrong target/connection, wrong orientation, missing/excess quantity, partial insertion, wrong order/omission, damage/contamination where safe to stage, and valid alternative procedure branches.

## Leakage and validity rules

1. Never split synchronized camera views of the same take across train and test.
2. Never split adjacent clips from one continuous run across partitions.
3. Keep operators, product instances and error scripts grouped.
4. Reserve entire mistake subtypes for open-set evaluation.
5. Do not tune thresholds on the final unseen-error test set.
6. Separate deliberately staged mistakes from spontaneous mistakes in reporting.
7. Report performance with and without privileged training modalities; deployment evaluation must be ego RGB plus only the sensors actually available on the line.
8. Treat recovery recommendations as invalid unless every action belongs to an approved recovery graph.

## Legal and operational recommendation

- Use Assembly101 only for research/model selection under CC BY-NC 4.0 unless commercial rights are obtained.
- Use IndustReal and its extension under Apache-2.0 for reproducible public experiments.
- Record dataset version, checksum, license acceptance and source URL in every experiment manifest.
- Do not redistribute gated or non-commercial video inside the repository.
- Before production training, have the organization confirm whether pretrained weights derived from non-commercial data may be used in the intended product. If uncertain, use Assembly101 for architecture selection only, then retrain the selected model on Apache-licensed and proprietary data.

## Concrete first experiment

The first credible StateGraph-PSR experiment should be:

1. Assembly101 official train/validation split, one ego view per take.
2. Cached video features plus 3D hand features.
3. Causal FACT-lite temporal head.
4. Relation graph created from `attach/detach(part_a, part_b)`.
5. Heads for coarse step, graph delta, mistake/correction and next action.
6. Test on official Assembly101 test, separated into seen/unseen toys.
7. Zero-shot semantic transfer to IndustReal.
8. Fine-tune state/effect heads on IndustReal train only.
9. Evaluate the locked IndustReal validation/test errors, including the 2024 four-type extension.

This experiment exercises every essential architectural idea except tool-specific errors and expert-approved recovery language. Those require MECCANO/HoloAssist auxiliary training and project-specific aerospace data.

## Sources

1. Sener et al., [Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities](https://assembly-101.github.io/), CVPR 2022.
2. Assembly101, [official annotations repository](https://github.com/assembly-101/assembly101-annotations).
3. Assembly101, [official Mistake Detection annotations](https://github.com/assembly-101/assembly101-mistake-detection).
4. Ding et al., [Every Mistake Counts in Assembly](https://arxiv.org/abs/2307.16453), 2023.
5. Schoonbeek et al., [IndustReal](https://openaccess.thecvf.com/content/WACV2024/html/Schoonbeek_IndustReal_A_Dataset_for_Procedure_Step_Recognition_Handling_Execution_Errors_WACV_2024_paper.html), WACV 2024.
6. [IndustReal v2 dataset record](https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224), 4TU.ResearchData.
7. Schoonbeek et al., [Supervised Representation Learning Towards Generalizable Assembly State Recognition](https://arxiv.org/abs/2408.11700), IEEE RA-L 2024.
8. [IndustReal Assembly State Recognition with Error States extension](https://doi.org/10.4121/611adbc7-7935-43a6-8c3f-b2260a508e73), 4TU.ResearchData 2024.
9. Wang et al., [HoloAssist](https://holoassist.github.io/), ICCV 2023.
10. Ragusa et al., [MECCANO](https://iplab.dmi.unict.it/legacy/MECCANO/), CVIU 2023.
11. Peddi et al., [CaptainCook4D](https://captaincook4d.github.io/captain-cook/), NeurIPS 2024 Datasets and Benchmarks.
12. Grauman et al., [Ego-Exo4D documentation](https://docs.ego-exo4d-data.org/), 2024–2026 releases.
13. Ben-Shabat et al., [IKEA ASM](https://ikeaasm.github.io/), 2020.
14. Skill AI, [Ego-Exo Manufacturing](https://huggingface.co/datasets/skill-ai/ego-exo-manufacturing), 2026 dataset release.
