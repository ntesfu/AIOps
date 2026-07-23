# Assembly101 realtime benchmark

This benchmark separates the causal visual feature pipeline from the trainable
StateGraph temporal head. Measurements use a Slurm GPU, bf16 inference, batch
size 1, one `e1` ego view, 30 FPS source video, 32-frame causal clips, and an
8-frame update stride. The temporal-head proxy is the selected I0 architecture,
which has the same 34.92M-parameter shape as the final staged model.

## Measured latency

| Component | Workload | Latency / throughput | Peak allocated VRAM |
|---|---|---:|---:|
| Visual features | Decode + causal clip + Swin3D-S + JPEG current frame + ConvNeXt-T; 577 genuine clips | 81.35 ms/clip mean | 0.971 GiB |
| StateGraph head | 512 cached feature steps, batch 1, 100 iterations | 12.87 ms mean; 12.82 ms p95 | 0.483 GiB |
| Approximate serial compute | Feature mean + head p95 | 94.17 ms/update | not additive in practice |

The source produces one feature update every `8/30 = 266.67 ms`, so measured
serial compute consumes about 35% of the update budget. The feature benchmark
processed 153.87 seconds of genuine video in 46.94 seconds, or 3.28x realtime.
The temporal head processes about 39,788 cached feature steps per second.

The earlier batch-8 cache extraction recorded a higher 5.846 GiB peak, and full
training peaks near 2.7 GiB (1.641 GiB when the action branch is frozen). All are
comfortably below the required 23 GiB ceiling.

## Delay interpretation

Compute latency is not the full alert delay:

- A causal 32-frame clip at 30 FPS carries 1.067 seconds of past visual context.
- Updates arrive every 0.267 seconds.
- Visual and temporal compute add roughly 0.094 seconds on the benchmark GPU.
- Learned event localization adds the validation matched-delay metric. The best
  staged seed-17 ablation is 0.93 seconds; the staged procedural ablation is
  4.98 seconds. The final 80-epoch model must be reported separately on held-out
  test rather than inheriting either validation number.

Thus the architecture can execute in realtime, but realtime throughput does not
guarantee prompt mistake detection. The final operational latency is the causal
context plus compute plus the model's measured event-localization delay.

## Reproduction

The temporal measurement is produced by
`scripts/benchmark_stategraph_checkpoint.py`; the genuine video/feature
measurement is produced by `scripts/benchmark_assembly101_features.py`. Both
scripts print JSON and distinguish their measurement scope so cached-head timing
cannot be mistaken for end-to-end video timing.
