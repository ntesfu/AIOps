# IndustReal Procedure Intelligence Architecture

## Executive decision

The next system should not be another clip classifier with a larger frozen backbone. It should be a **state-aware, graph-constrained streaming model** that explicitly represents:

1. what the operator is doing;
2. which parts, tools, hands, and connection points are involved;
3. what physical state changed;
4. which states and steps are valid at this point in the procedure;
5. whether the observation is a known error, an unseen deviation, or merely uncertain;
6. what valid future states and recovery actions remain.

The recommended first implementation is **Candidate A: StateGraph-PSR**. It is designed to train on one 24 GB GPU by caching or freezing the visual backbones and training compact temporal, relation, and task-graph modules. Candidate B adds long-horizon self-supervised procedural learning. Candidate C adds a VLM as an event-triggered adjudicator and explanation layer, not as the continuous safety-critical detector.

## Why the current v2 architecture is insufficient

The current deck uses a frozen VideoMAEv2-giant encoder, clip-level step and outcome heads, Viterbi smoothing, and a final merge/verification stage. This is a useful temporal-segmentation baseline, but four structural limitations prevent it from becoming a reliable assembly assistant:

- **Motion is not state.** Similar hand motion can use the wrong tool, fastener, orientation, or mating point. Correctness often becomes visible only in the resulting object state.
- **A fixed order is not a procedure model.** Viterbi smoothing over a mostly linear order cannot naturally represent optional steps, partial orders, parallel subassemblies, retries, undo operations, or recovery branches.
- **Closed-set outcome labels cannot cover real mistakes.** A small correct/incorrect/removed head does not express error cause and cannot safely detect unseen failures.
- **Clip classification loses causal evidence.** Before state, contact/interaction, point-of-no-return, and after/effect state need to be modeled as linked evidence, not pooled into one clip vector.

Recent procedural-video research supports this change. STORM-PSR combines spatial assembly-state and temporal streams; EgoPED models hand-object relations; PREGO compares online recognition to expected actions; AMNAR models multiple valid next actions; Action Effect Modeling verifies resulting states; generalized task/execution graphs align observations to procedures; and MATT attributes a mistake semantically, temporally, and spatially.

## Target behavior and engineering requirements

### Online outputs

At time `t`, the system emits a structured record:

```json
{
  "current_step": "install_controller_harness",
  "step_probability": 0.93,
  "phase": "contact",
  "completion_probability": 0.61,
  "observed_state_delta": ["connector_A inserted_into socket_C"],
  "expected_state_delta": ["connector_B inserted_into socket_C"],
  "mistake_probability": 0.96,
  "mistake_type": "wrong_connection",
  "point_of_no_return_s": 81.4,
  "evidence": {
    "tool": "none",
    "part": "connector_A",
    "target": "socket_C"
  },
  "next_valid_steps": ["disconnect_connector_A", "insert_connector_B"],
  "recommended_recovery": "Disconnect A, inspect socket C, then insert B.",
  "uncertainty": 0.08
}
```

### Initial deployment targets

- Continuous processing at 5–10 analyzed frames per second.
- Step update latency below 500 ms after each feature window.
- High-confidence alert within 1–2 seconds of visible evidence.
- False alerts below 0.2 per operating hour before pilot deployment.
- Explicit abstention when evidence is occluded or out of distribution.
- Deterministic, auditable recovery planning from approved procedure rules.
- Optional VLM reasoning may add seconds, but cannot block the primary alert path.

These are product targets, not claims from the cited papers; they should be adjusted after measuring the camera, workstation, and operator workflow.

## Data and label model

The current label space should be expanded from a flat step ID plus outcome to four synchronized layers.

### 1. Action layer

Represent each atomic action as semantic roles:

`verb + manipulated_part + target_part + tool + hand + location`

Example: `insert + connector_B + socket_C + none + right_hand + controller_board`.

### 2. Phase and evidence layer

For every action or step, annotate or derive:

- approach/start;
- first contact;
- manipulation;
- completion cue;
- effect frame;
- point-of-no-return (when a deviation becomes irreversible without recovery);
- visible evidence region: hand, part, tool, fastener, target, or assembled interface.

### 3. State layer

Use sparse predicates rather than a single monolithic assembly class:

- `present(part)` / `selected(part)`;
- `aligned(part, target)`;
- `inserted(part, target, depth_bin)`;
- `connected(part, target)`;
- `fastened(fastener, location)`;
- `orientation(part, pose_bin)`;
- `tool_in_hand(tool)` / `tool_applied(tool, target)`;
- `count(fastener_type, location)`;
- `continuity(connection)` or other sensor-confirmed predicates when available.

This factorization is essential: a full assembly state is combinatorial, while most steps change only a few local predicates.

### 4. Mistake layer

Use a hierarchical taxonomy with an open-set escape hatch:

- **Procedure:** omitted, repeated, out of order, premature, wrong step, unauthorized branch.
- **Selection:** wrong part, wrong fastener, wrong tool, wrong material/variant.
- **Spatial:** wrong target, wrong connection, wrong orientation, misalignment, wrong location.
- **Execution/technique:** partial insertion, insufficient fastening, over-tightening, damage, slip, contamination.
- **Quantity/timing:** missing count, excess count, insufficient duration, excessive duration.
- **Outcome/state:** action appeared correct but the resulting state or relation is wrong.
- **Unknown deviation:** statistically and procedurally abnormal but not safely attributable.

The model should output the parent type, subtype, violated semantic role, time/PNR, evidence region, and confidence. This combines the practical taxonomies used by IndustReal, EgoPER, CaptainCook4D, and the MATT formulation.

## Candidate A — StateGraph-PSR (recommended first build)

### A1. Perception encoders

**Global motion stream**

- Input: 16 frames at 224×224 over approximately 2 seconds, sampled at 8 FPS; update every 0.5 seconds.
- First 24 GB experiment: VideoMAEv2-Base or Large, pretrained on motion-sensitive video data, with the backbone frozen or adapted using temporal adapters/LoRA.
- Alternative streaming backbone: StreamFormer-style causal temporal attention.
- Output: global clip tokens plus intermediate dense tokens if available.

**Interaction/state stream**

- Detect and track hands, tools, parts, fasteners, connectors, and mating points.
- Pool ROI features from the RGB encoder; fuse depth, hand joints, gaze, and CAD/pose cues when present.
- Construct a dynamic relation graph with edges such as `holding`, `touching`, `aligned_with`, `inserted_into`, `connected_to`, and `fastening`.
- Encode the graph with a small graph transformer or relation GNN.
- Train the detector and state predicates with real images plus CAD-rendered synthetic images, domain randomization, and hard occlusion augmentation.

### A2. Cross-modal state fusion

At each update, cross-attend global motion tokens to object/relation tokens. Emit:

- atomic action distribution;
- current phase and boundary probabilities;
- current state predicates;
- predicted state delta;
- visibility/occlusion scores;
- embedding for normality and open-set detection.

The state stream must remain independently inspectable. A correct action label cannot override a contradictory physical state.

### A3. Streaming temporal segmenter

Use a **causal FACT-like temporal head**:

- causal dilated temporal convolutions on frame/clip features;
- a compact set of learned action/step tokens;
- frame-to-action and action-to-frame cross-attention;
- memory cache for the current procedure run;
- boundary, phase, action, step, and completion heads.

Train an offline FACT or DiffAct model as a teacher if desired, then distill it into the causal student. This preserves the strong offline model as an upper bound while optimizing the deployed model for latency.

### A4. Task-graph belief decoder

Replace the linear validator with a probabilistic task graph:

- nodes: correct steps, allowed background actions, inspection states, recovery states;
- edges: allowed precedence, optional/parallel branches, undo/redo, and recovery transitions;
- node predicates: required preconditions and expected effects;
- online belief: probability over current node and assembly state;
- decoding: causal semi-Markov beam search or differentiable task-graph alignment.

The decoder should never force an observation into a legal step. It must preserve an `unknown/deviation` hypothesis.

### A5. Mistake detection and attribution

Use three complementary residuals:

1. **Procedure residual:** observed action versus valid next-step set.
2. **Execution residual:** observed segment embedding versus multiple normal prototypes conditioned on history.
3. **Effect residual:** observed post-action state versus expected state delta.

Fuse them in a calibrated attribution head that predicts mistake family, subtype, violated role, PNR, and evidence region. Train the open-set branch on correct executions plus synthetic/counterfactual negatives; reserve real error examples for supervised attribution and evaluation.

### A6. Future prediction and recovery

- Predict a distribution over the next 1, 3, and 5 steps from the task-graph belief.
- Predict future state deltas in latent/predicate space, not pixels.
- Score risk: probability that each candidate continuation violates a constraint or makes recovery harder.
- Search the approved task graph for the shortest safe path from the observed state to a valid state.
- Recovery primitives: inspect, undo, remove, replace, reconnect, repeat, verify, or escalate to a human.
- Convert the selected plan into language and optionally retrieve a validated expert clip. A VLM may phrase or explain the plan, but it cannot invent an unapproved repair operation.

## Candidate B — P-JEPA procedural representation

This candidate targets label scarcity and long-range context.

1. Extract dense features from V-JEPA 2.1, VideoMAEv2, or a lighter video encoder.
2. Train a P-JEPA-style contextualizer on all available unlabeled procedure runs. It predicts masked future/intermediate latent segments and learns progress-sensitive, long-duration representations.
3. Add the same object-state graph and task-graph decoder as Candidate A.
4. Fine-tune compact probes for action, step, boundary, state delta, mistake residual, and anticipation.

**24 GB route:** cache backbone features and train only the procedural contextualizer and heads. This should fit comfortably because the expensive video backbone does not retain gradients.

**Scale-up route:** adapt the last video-backbone blocks jointly, use longer clips and multi-view contrast, and pretrain on Assembly101, Ego-Exo4D, MECCANO, HoloAssist, and the project's unlabeled video.

Candidate B is scientifically bolder and particularly attractive after the supervised Candidate A establishes trustworthy metrics and labels.

## Candidate C — event-triggered VLM collaboration

Use a VLM for sparse reasoning, attribution, and explanation:

1. The continuous core model raises a candidate event with timestamps, state residuals, task-graph context, and uncertainty.
2. An evidence selector chooses a small packet: before frame, contact frame, PNR candidate, effect frame, ROIs, and recent step history.
3. An AXG-style structured prompt asks the VLM to compare expected and observed semantic roles and output strict JSON.
4. The result is fused with the specialist model only after schema validation and confidence checks.

**24 GB model choice:** Qwen3-VL-4B in BF16/FP8 or Qwen3-VL-8B in 4-bit/FP8, run sequentially on event packets. A 2B model is an additional latency baseline. Fine-tune only LoRA adapters if domain adaptation is needed.

**Larger-compute alternative:** ProVideLLM-style streaming memory or a larger Qwen3-VL variant on a separate inference GPU/server.

The VLM should not continuously inspect every frame. Sparse evidence improves temporal focus, latency, cost, and auditability, and follows the motivation of AXG-Reasoner.

## Model-selection experiments

Run experiments in the following order so each result answers one architectural question.

| ID | Change from previous | Question answered |
|---|---|---|
| E0 | Reproduce current VideoMAEv2 + ASFormer/DiffAct result | Is the evaluation pipeline stable? |
| E1 | FACT on exactly the current cached features | Is segmentation architecture the limiting factor? |
| E2 | Smaller adaptable VideoMAEv2 + temporal adapters/LoRA | Does domain adaptation beat a frozen 1B encoder? |
| E3 | Add object/hand/tool relation graph | Do interaction cues improve fine-grained steps and wrong-object/tool errors? |
| E4 | Add predicate/state-delta heads | Does explicit outcome modeling improve completed-step recognition? |
| E5 | Replace Viterbi with branching task-graph decoder | Does structured decoding improve POS, omissions, and valid reordering? |
| E6 | Add multiple normal prototypes + open-set energy | Can the system detect unseen errors without collapsing recall? |
| E7 | Add P-JEPA contextualizer on cached features | Does unlabeled long-form pretraining improve small-data generalization? |
| E8 | Add event-triggered Qwen3-VL adjudicator | Does VLM evidence improve subtype/explanation without harming false-alert rate? |
| E9 | Distill offline teacher into causal student | What is the accuracy–latency frontier? |

## Training plan for one 24 GB GPU

### Phase 0 — data audit and split hygiene

- Split by operator, recording session, camera/environment, and assembly instance.
- Keep staged error patterns and recovery sequences isolated from training when measuring open-set behavior.
- Report class counts, duration, error frequency, visibility, and completion cues.
- Convert procedure JSON into a graph with preconditions/effects.

### Phase 1 — feature and detector baselines

- Cache current VideoMAEv2 features once.
- Train FACT and the task-graph decoder on cached features.
- Train part/tool/hand detector and predicate heads separately using crops and CAD synthetic data.
- Establish reproducible data loaders, checkpoints, and per-run manifests.

### Phase 2 — state-aware multi-task training

Train temporal and state modules with a weighted objective:

`L = L_action + L_step + 0.5 L_boundary + L_state + 0.5 L_effect + 0.2 L_graph + 0.5 L_open + 0.3 L_future + 0.05 L_calibration`

These are starting weights; tune them by gradient balance and validation performance. Use class-balanced sampling or focal loss for rare mistake types.

Recommended settings:

- BF16 mixed precision;
- gradient checkpointing only when adapting the backbone;
- batch 1–2 clips with gradient accumulation to effective batch 16–32;
- freeze most of the backbone for the first experiments;
- discriminative learning rates: heads `1e-4` to `3e-4`, adapters `1e-5` to `5e-5`;
- early stopping on a composite of F1@50, POS, completion delay, and false alerts/hour;
- operator-held-out validation, not random clip validation.

### Phase 3 — error and counterfactual curriculum

- Begin with normal-only prototypes and procedure violations.
- Generate controlled counterfactual labels: wrong part/tool/target/orientation, omission, swap, partial completion, and wrong effect.
- Use CAD renderings and compositing for selection/spatial variants; do not rely on synthetic video alone for final evaluation.
- Fine-tune attribution on real mistakes while keeping unseen mistake types held out.
- Calibrate every error family separately using temperature scaling or conformal thresholds.

### Phase 4 — anticipation and recovery

- Train next-step and next-state prediction jointly with task-graph belief.
- Add action-conditioned latent prediction only after current-state metrics are stable.
- Build a deterministic recovery graph from approved work instructions and engineering constraints.
- Human-review every recovery template before it is eligible for live recommendation.

### Phase 5 — causal deployment optimization

- Distill offline teacher logits, boundaries, state embeddings, and future predictions into the causal student.
- Quantize only after accuracy and calibration are stable.
- Benchmark TensorRT/ONNX or the selected runtime at the actual input rate.
- Add watchdogs for stale frames, camera loss, out-of-distribution scenes, and missing evidence.

## Evaluation protocol

### Step and procedure recognition

- frame accuracy/MoF;
- Edit score;
- F1@10/25/50;
- Procedure Order Similarity (POS);
- step completion F1;
- completion delay and early-completion false positives;
- per-step and per-operator breakdowns.

### Mistake detection and attribution

- frame AUROC/AUPR and segment EDA;
- macro-F1 by mistake family and subtype;
- seen versus unseen error AUROC;
- false alerts per hour;
- alert delay from PNR and from first visible evidence;
- semantic-role attribution accuracy;
- PNR timestamp error;
- evidence localization IoU or pointing accuracy;
- expected calibration error and abstention coverage/risk.

### Anticipation and recovery

- next-step top-1/top-3 recall at 1, 3, and 5-step horizons;
- state-delta macro-F1;
- time-to-action error;
- valid recovery plan rate;
- minimality versus expert plan;
- simulated/recorded recovery success;
- expert acceptance rate;
- unsafe recommendation rate, which must be treated as a release blocker.

### Deployment

- FPS and p50/p95 latency per module;
- peak VRAM and memory drift over a full shift;
- end-to-end alert latency;
- camera and occlusion robustness;
- VLM trigger rate and adjudication latency;
- performance under lighting, viewpoint, gloves, tool, operator, and assembly-variant shifts.

## Twelve-week execution plan

| Weeks | Deliverable | Exit gate |
|---|---|---|
| 1–2 | Reproducible E0/E1; clean split; task graph schema; metric dashboard | Baseline numbers reproduced within tolerance |
| 3–4 | Part/tool/hand tracking and predicate dataset; CAD augmentation | State predicates beat image-only baseline on held-out operators |
| 5–6 | Candidate A fusion + causal FACT head | Step F1/POS improve without unacceptable delay |
| 7–8 | Effect verifier, normal prototypes, open-set detector | Meaningful unseen-error AUROC and false-alert target trend |
| 9 | Attribution labels and MATT-style outputs | Parent error type, role, PNR, and evidence all measurable |
| 10 | Future-step/state heads and recovery graph search | High valid-plan rate on offline traces |
| 11 | Qwen3-VL event adjudicator | Subtype/explanation improves without raising unsafe alerts |
| 12 | Causal distillation, profiling, ablation report | Candidate selected for pilot implementation |

## Scaling path after more GPUs arrive

- Jointly adapt the last blocks of V-JEPA 2.1/VideoMAEv2 and the object-state branch.
- Train P-JEPA on longer windows and all unlabeled project video.
- Add synchronized egocentric/exocentric views with consistency losses.
- Pretrain on Assembly101, MECCANO, HoloAssist, Ego-Exo4D, and mistake datasets, then domain-adapt.
- Use a larger VLM or ProVideLLM-style streaming memory on a separate GPU.
- Explore action-conditioned latent world models for future state prediction, but retain the symbolic recovery graph for safety and auditability.

## Research sources

1. Schoonbeek et al., [IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors](https://openaccess.thecvf.com/content/WACV2024/html/Schoonbeek_IndustReal_A_Dataset_for_Procedure_Step_Recognition_Handling_Execution_Errors_WACV_2024_paper.html), WACV 2024.
2. Schoonbeek et al., [Learning to Recognize Correctly Completed Procedure Steps through Spatio-Temporal Modeling (STORM-PSR)](https://arxiv.org/abs/2510.12385), 2025.
3. Lu and Elhamifar, [FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html), CVPR 2024.
4. Liu et al., [Diffusion Action Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Diffusion_Action_Segmentation_ICCV_2023_paper.html), ICCV 2023.
5. Liu et al., [End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html), CVPR 2024.
6. Lee et al., [Error Detection in Egocentric Procedural Task Videos](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html), CVPR 2024.
7. Flaborea et al., [PREGO: Online Mistake Detection in Procedural Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2024/html/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.html), CVPR 2024.
8. Huang et al., [Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Modeling_Multiple_Normal_Action_Representations_for_Error_Detection_in_Procedural_CVPR_2025_paper.html), CVPR 2025.
9. Lee and Elhamifar, [Error Recognition in Procedural Videos using Generalized Task Graph](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.html), ICCV 2025.
10. Lee and Elhamifar, [AXG-Reasoner: Error Detection and Explanation in Long Task Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_AXG-Reasoner_Error_Detection_and_Explanation_in_Long_Task_Videos_with_CVPR_2026_paper.html), CVPR 2026.
11. Li et al., [Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Mistake_Attribution_Fine-Grained_Mistake_Understanding_in_Egocentric_Videos_CVPR_2026_paper.html), CVPR 2026.
12. Guo et al., [Procedural Mistake Detection via Action Effect Modeling](https://arxiv.org/abs/2512.03474), 2025.
13. Kung et al., [State-Change Counterfactuals for Procedure-Aware Video Representation Learning](https://openaccess.thecvf.com/content/ICCV2025/html/Kung_What_Changed_and_What_Could_Have_Changed_State-Change_Counterfactuals_for_ICCV_2025_paper.html), ICCV 2025.
14. Seminara et al., [Differentiable Task Graph Learning](https://arxiv.org/abs/2406.01486), 2024.
15. Tristram et al., [P-JEPA: Procedural Video Representation Learning via Joint Embedding Predictive Architecture](https://arxiv.org/abs/2606.23256), 2026.
16. Mur-Labadia et al., [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482), 2026.
17. Chatterjee et al., [Streaming VideoLLMs for Real-Time Procedural Video Understanding](https://openaccess.thecvf.com/content/ICCV2025/html/Chatterjee_Streaming_VideoLLMs_for_Real-Time_Procedural_Video_Understanding_ICCV_2025_paper.html), ICCV 2025.
18. Bai et al., [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), 2025.
19. Sener et al., [Assembly101](https://openaccess.thecvf.com/content/CVPR2022/html/Sener_Assembly101_A_Large-Scale_Multi-View_Video_Dataset_for_Understanding_Procedural_Activities_CVPR_2022_paper.html), CVPR 2022.
20. Peddi et al., [CaptainCook4D](https://arxiv.org/abs/2312.14556), NeurIPS 2024 Datasets and Benchmarks.
21. Wang et al., [HoloAssist](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.html), ICCV 2023.
22. Ashutosh et al., [ExpertAF: Expert Actionable Feedback from Video](https://openaccess.thecvf.com/content/CVPR2025/html/Ashutosh_ExpertAF_Expert_Actionable_Feedback_from_Video_CVPR_2025_paper.html), CVPR 2025.
