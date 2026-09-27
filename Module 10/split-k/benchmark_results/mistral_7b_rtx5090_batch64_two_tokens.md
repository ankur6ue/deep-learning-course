# Mistral 7B, Batch 64, One Decode Pass On RTX 5090

This sweep tests whether the extra M-axis parallelism from a batch of 64 makes
split-K stop helping at lower LoRA ranks. The raw samples are in
`mistral_7b_rtx5090_batch64_two_tokens.json`.

## Configuration

```text
model            mistralai/Mistral-7B-Instruct-v0.3
GPU              NVIDIA GeForce RTX 5090, 170 SMs, compute capability 12.0
vLLM             0.30.0
Transformers     5.17.0
PyTorch          2.13.0+cu130
dtype            bfloat16
stream mode      single
async scheduler  disabled for one-step decode timing
batch size       64
prompt tokens    128 per request
output tokens    2: one from prefill and one from one decode pass
ranks            1, 8, 16, 32, 64, 128, 256, 320, 512
split factors    1, 2, 4, 8, 16, 32, 64
warmups          3 per point
timed trials     25 per point, with split-factor order randomized each trial
```

The selected split factor applied to both prefill and decode. Observed shrink
shapes confirmed `M=8192` during prefill and `M=64` during decode. vLLM's
asynchronous scheduler was disabled because it can queue both model steps and
process their outputs close together; under that mode, request timestamps do
not isolate the latency of a single decode pass.

## Decode Results

| Rank | Best median K | K=1 ms | Best ms | Nominal speedup | Paired trials faster than K=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 23.310 | 23.310 | 1.000x | - |
| 8 | 8 | 20.652 | 20.347 | 1.015x | 13/25 |
| 16 | 8 | 23.787 | 22.979 | 1.035x | 17/25 |
| 32 | 64 | 22.954 | 22.704 | 1.011x | 12/25 |
| 64 | 1 | 22.874 | 22.874 | 1.000x | - |
| 128 | 16 | 23.214 | 22.953 | 1.011x | 15/25 |
| 256 | 1 | 22.461 | 22.461 | 1.000x | - |
| 320 | 1 | 23.361 | 23.361 | 1.000x | - |
| 512 | 4 | 22.859 | 22.716 | 1.006x | 13/25 |

None of the nominal `K>1` winners is distinguishable from `K=1` in these
samples. Paired bootstrap 95% intervals for the median latency improvement all
include zero. Even the largest nominal result, rank 16 with `K=8`, has an
interval of `[-0.028, 0.713] ms`. The exact best K in the table should
therefore be treated as sample noise rather than a tuning recommendation.

All median decode times, in milliseconds:

| Rank | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.310 | 23.574 | 23.924 | 23.431 | 23.816 | 23.950 | 23.498 |
| 8 | 20.652 | 22.595 | 22.796 | 20.347 | 22.685 | 20.621 | 22.989 |
| 16 | 23.787 | 23.341 | 23.442 | 22.979 | 23.457 | 23.503 | 23.453 |
| 32 | 22.954 | 22.869 | 23.450 | 23.103 | 23.215 | 23.134 | 22.704 |
| 64 | 22.874 | 23.166 | 22.916 | 23.398 | 23.324 | 23.484 | 23.174 |
| 128 | 23.214 | 23.567 | 23.290 | 23.285 | 22.953 | 23.852 | 23.090 |
| 256 | 22.461 | 22.689 | 22.988 | 22.623 | 22.975 | 23.615 | 23.203 |
| 320 | 23.361 | 23.465 | 23.466 | 23.762 | 23.502 | 23.621 | 23.532 |
| 512 | 22.859 | 23.293 | 22.716 | 22.995 | 23.347 | 22.963 | 23.229 |

## Interpretation

At batch 64, split-K provides no measurable full-model decode benefit at any
tested rank. By rank 32 its nominal benefit is only 1.1%, rank 64 selects
`K=1`, and ranks 256 and 320 also select `K=1`. This is the practical crossover
the experiment was looking for: after roughly rank 16, extra splitting no
longer improves decode latency for this large batch.

The shrink kernel uses `BLOCK_M=32` and `BLOCK_N=16`. Batch 64 supplies two M
tiles before split-K is applied, so the approximate unsplit output grid for a
single projection slice is:

```text
ceil(64 / 32) * ceil(rank / 16) = 2 * ceil(rank / 16)
```

This is twice the M-axis work available at batch 8. At low rank, the shrink
operation is too small a share of the complete model step for extra programs
to produce a reliable end-to-end gain. As rank grows, N-axis tiles provide
more independent work, while split-K retains its launch and atomic-reduction
cost. The measured result is consequently a broad flat region followed by a
penalty for aggressive splitting, not a sharp mathematical threshold.

This result is consistent with the batch-size hypothesis, but it is not a
controlled comparison with the earlier batch-8 sweep: that run used an RTX
5080 and vLLM's asynchronous scheduler. A same-GPU, synchronous batch-8 run is
needed to attribute the difference solely to batch size.

The subsequent
[`batch-32 RTX 5090 sweep`](mistral_7b_rtx5090_batch32_two_tokens.md) is a
controlled comparison that halves the number of M-axis tiles. It shows more
nominal `K>1` winners, but no decode improvement was statistically
distinguishable from `K=1` in 25 trials.

## Guardrail Latency

Prefill dominates this two-token workload, and one split factor is used for
both phases. The largest nominal wall-time improvement over `K=1` is only 0.6%
at any rank:

| Rank | Best wall-time K | Wall speedup vs K=1 |
|---:|---:|---:|
| 1 | 16 | 1.002x |
| 8 | 4 | 1.005x |
| 16 | 2 | 1.006x |
| 32 | 8 | 1.005x |
| 64 | 2 | 1.004x |
| 128 | 2 | 1.000x |
| 256 | 2 | 1.003x |
| 320 | 2 | 1.001x |
| 512 | 1 | 1.000x |

Large split factors become increasingly expensive during prefill. At rank 512,
`K=64` raises median prefill latency from `697.10` to `790.74 ms` and wall time
from `725.13` to `819.20 ms`. For this batch-64 guardrail workload, `K=1` is
the conservative setting across ranks; the data does not justify paying the
prefill cost of a larger split for an unproven decode gain.
