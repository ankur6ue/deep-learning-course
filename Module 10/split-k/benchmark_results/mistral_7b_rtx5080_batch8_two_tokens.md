# Mistral 7B, Batch 8, One Decode Pass On RTX 5080

This sweep tests whether split-K stops helping LoRA decode once rank exceeds 8
or 16. The raw samples are in
`mistral_7b_rtx5080_batch8_two_tokens.json`.

> **Timing note:** This run used vLLM 0.30's default asynchronous scheduler.
> With only two output tokens, vLLM can queue prefill and decode and process
> their outputs close together, so the request-level inter-token interval does
> not reliably isolate decode model execution. Keep this result as historical
> data; rerun with the current harness before using its decode speedups.

## Configuration

```text
model            mistralai/Mistral-7B-Instruct-v0.3
GPU              NVIDIA GeForce RTX 5080, 84 SMs, compute capability 12.0
vLLM             0.30.0
Transformers     5.17.0
PyTorch          2.13.0+cu130
dtype            bfloat16
stream mode      single
async scheduler  enabled implicitly by vLLM 0.30
batch size       8
prompt tokens    128 per request
output tokens    2: one from prefill and one from one decode pass
ranks            1, 8, 16, 32, 64, 128, 256, 320, 512
split factors    1, 2, 4, 8, 16, 32, 64
warmups          3 per point
timed trials     25 per point, with split-factor order randomized each trial
```

The selected split factor applied to both prefill and decode. Observed shrink
shapes confirmed `M=1024` during prefill and `M=8` during decode. Ranks through
128 used `gpu_memory_utilization=0.95`; ranks 256 through 512 used 0.98 so the
larger adapters fit on the 16 GB card. This changes KV-cache reservation, not
the kernels used by this fixed workload.

## Decode Results

| Rank | Best median K | K=1 ms | Best ms | Speedup | Trials faster than K=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 6.055 | 5.363 | 1.129x | 25/25 |
| 8 | 16 | 6.597 | 5.609 | 1.176x | 25/25 |
| 16 | 32 | 6.303 | 5.432 | 1.160x | 25/25 |
| 32 | 32 | 6.665 | 5.851 | 1.139x | 25/25 |
| 64 | 32 | 7.140 | 6.210 | 1.150x | 25/25 |
| 128 | 8 | 7.441 | 6.693 | 1.112x | 24/25 |
| 256 | 8 | 7.219 | 6.653 | 1.085x | 25/25 |
| 320 | 4 | 7.896 | 7.321 | 1.078x | 24/25 |
| 512 | 8 | 8.433 | 7.885 | 1.069x | 24/25 |

The exact winning K should not be over-interpreted. Paired comparisons between
the best and second-best K had confidence intervals spanning zero through rank
320. At rank 512, K=8 was distinguishable from the second-place K=16 in this
sample. The useful region is generally a plateau around K=8 through K=32 for
lower ranks and K=4 through K=8 at the largest ranks. Paired bootstrap
intervals for the listed best K versus K=1 excluded zero at every rank.

All median decode times, in milliseconds:

| Rank | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.055 | 5.665 | 5.488 | 5.380 | 5.372 | 5.363 | 5.409 |
| 8 | 6.597 | 6.620 | 6.098 | 6.001 | 5.609 | 5.801 | 5.849 |
| 16 | 6.303 | 5.928 | 5.654 | 5.495 | 5.578 | 5.432 | 5.469 |
| 32 | 6.665 | 6.403 | 6.099 | 5.916 | 5.879 | 5.851 | 5.940 |
| 64 | 7.140 | 6.778 | 6.414 | 6.268 | 6.347 | 6.210 | 6.270 |
| 128 | 7.441 | 7.034 | 6.769 | 6.693 | 6.774 | 6.793 | 6.908 |
| 256 | 7.219 | 6.906 | 6.657 | 6.653 | 6.752 | 6.809 | 7.121 |
| 320 | 7.896 | 7.652 | 7.321 | 7.467 | 7.383 | 7.516 | 8.119 |
| 512 | 8.433 | 8.115 | 8.124 | 7.885 | 8.023 | 8.290 | 9.162 |

## Interpretation

The proposed crossover at rank 8 or 16 did not occur. Batch 8 still produces
only one M tile because the shrink kernel uses `BLOCK_M=32`. Rank 8 and rank 16
also both produce one N tile because `BLOCK_N=16`. The approximate unsplit
decode grid for a projection slice is therefore:

```text
ceil(8 / 32) * ceil(rank / 16)
```

Even rank 512 supplies only 32 output tiles for a single-slice projection on an
84-SM GPU. The fused QKV shrink has three slices and more total programs, but
the single-slice output projection remains under-filled. Split-K can therefore
still expose useful work at high rank. The trend appears in the expected
direction only gradually: the best observed speedup falls from 1.176x at rank
8 to 1.069x at rank 512, corresponding to latency reductions of 15.0% and
6.5%. Excessive splitting becomes harmful: at rank 512, K=64 was 8.6% slower
than K=1.

## Guardrail Latency

There is only one decode pass, so prefill dominates end-to-end latency. The
best wall-time improvements were 2.2% to 2.4% through rank 16, 2.0% at rank 32,
1.4% at rank 64, and less than 1% from rank 128 onward. The K that minimizes
wall time is consequently smaller than the decode-only winner at high ranks:

| Rank | Best wall-time K | Wall speedup vs K=1 |
|---:|---:|---:|
| 1 | 16 | 1.022x |
| 8 | 16 | 1.023x |
| 16 | 16 | 1.024x |
| 32 | 8 | 1.020x |
| 64 | 4 | 1.014x |
| 128 | 8 | 1.009x |
| 256 | 4 | 1.005x |
| 320 | 4 | 1.003x |
| 512 | 4 | 1.004x |

For this workload, K=8 is a robust general decode choice, while K=4 is safer
when optimizing complete guardrail latency at high rank. The production choice
should still be measured at the actual batch-size distribution: batches above
32 begin adding M tiles and should reduce the need for split-K.
