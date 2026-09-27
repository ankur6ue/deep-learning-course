# Mistral 7B, Batch 32, One Decode Pass On RTX 5090

This sweep repeats the batch-64 experiment with 32 requests. The raw samples
are in `mistral_7b_rtx5090_batch32_two_tokens.json`.

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
batch size       32
prompt tokens    128 per request
output tokens    2: one from prefill and one from one decode pass
ranks            1, 8, 16, 32, 64, 128, 256, 320, 512
split factors    1, 2, 4, 8, 16, 32, 64
warmups          3 per point
timed trials     25 per point, with split-factor order randomized each trial
```

The selected split factor applied to both prefill and decode. Hook metadata
confirmed `M=4096` during prefill and `M=32` during decode for every rank.

## Decode Results

| Rank | Best median K | K=1 ms | Best ms | Nominal speedup | Paired trials faster than K=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 23.347 | 22.815 | 1.023x | 12/25 |
| 8 | 1 | 22.481 | 22.481 | 1.000x | - |
| 16 | 64 | 23.592 | 22.780 | 1.036x | 14/25 |
| 32 | 64 | 22.392 | 22.004 | 1.018x | 13/25 |
| 64 | 8 | 22.931 | 22.639 | 1.013x | 15/25 |
| 128 | 8 | 23.607 | 22.833 | 1.034x | 14/25 |
| 256 | 4 | 23.279 | 22.643 | 1.028x | 17/25 |
| 320 | 8 | 22.787 | 22.451 | 1.015x | 14/25 |
| 512 | 16 | 23.100 | 22.714 | 1.017x | 17/25 |

Batch 32 has more nominal `K>1` winners than batch 64, with maximum median
speedups of about 3.6%. None is statistically distinguishable from `K=1` in
these samples: every paired bootstrap 95% interval for median latency
improvement includes zero. The exact winning K values are also irregular, so
they should not be interpreted as a rank-dependent tuning rule.

All median decode times, in milliseconds:

| Rank | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.347 | 23.090 | 23.223 | 23.200 | 22.815 | 23.046 | 23.283 |
| 8 | 22.481 | 22.771 | 23.129 | 23.071 | 22.960 | 23.401 | 23.281 |
| 16 | 23.592 | 23.082 | 23.611 | 23.535 | 22.782 | 23.091 | 22.780 |
| 32 | 22.392 | 22.410 | 22.312 | 22.323 | 22.896 | 22.532 | 22.004 |
| 64 | 22.931 | 22.823 | 22.743 | 22.639 | 22.806 | 23.256 | 23.101 |
| 128 | 23.607 | 23.362 | 23.186 | 22.833 | 23.432 | 23.273 | 23.420 |
| 256 | 23.279 | 23.246 | 22.643 | 22.838 | 22.813 | 22.813 | 22.781 |
| 320 | 22.787 | 22.898 | 22.649 | 22.451 | 22.694 | 22.600 | 22.552 |
| 512 | 23.100 | 23.084 | 22.835 | 22.779 | 22.714 | 22.940 | 22.999 |

## Comparison With Batch 64

The shrink kernel uses `BLOCK_M=32` and `BLOCK_N=16`. The approximate unsplit
decode grid for one projection slice is therefore:

```text
batch 32: ceil(32 / 32) * ceil(rank / 16) =     ceil(rank / 16)
batch 64: ceil(64 / 32) * ceil(rank / 16) = 2 * ceil(rank / 16)
```

Batch 64 starts with twice as many M-axis programs. The observed nominal
decode speedups are consistent with this extra parallelism reducing the value
of split-K, although the individual differences remain below statistical
resolution:

| Rank | Batch 32 best K | Batch 32 speedup | Batch 64 best K | Batch 64 speedup |
|---:|---:|---:|---:|---:|
| 1 | 16 | 1.023x | 1 | 1.000x |
| 8 | 1 | 1.000x | 8 | 1.015x |
| 16 | 64 | 1.036x | 8 | 1.035x |
| 32 | 64 | 1.018x | 64 | 1.011x |
| 64 | 8 | 1.013x | 1 | 1.000x |
| 128 | 8 | 1.034x | 16 | 1.011x |
| 256 | 4 | 1.028x | 1 | 1.000x |
| 320 | 8 | 1.015x | 1 | 1.000x |
| 512 | 16 | 1.017x | 4 | 1.006x |

The comparison supports the direction of the batch-size hypothesis, but it
does not establish a precise decode crossover. Full-model decode contains
attention, base-model projections, LoRA expand projections, normalization,
sampling, and scheduler work. Any shrink-only improvement must be large enough
to rise above those fixed costs and the roughly millisecond-scale trial spread.

## Complete Guardrail Latency

Prefill dominates this two-token workload. Batch 32 prefill has
`ceil(4096 / 32) = 128` M tiles per N tile, compared with 256 at batch 64. For
a low-rank single-slice projection, 128 programs are fewer than the RTX 5090's
170 SMs, so moderate split-K can still add useful prefill work.

| Rank | Best wall-time K | K=1 ms | Best ms | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 32 | 327.86 | 324.79 | 1.009x |
| 8 | 8 | 329.36 | 325.98 | 1.010x |
| 16 | 4 | 328.47 | 326.53 | 1.006x |
| 32 | 8 | 328.95 | 325.73 | 1.010x |
| 64 | 4 | 330.38 | 329.16 | 1.004x |
| 128 | 2 | 341.28 | 336.93 | 1.013x |
| 256 | 4 | 351.61 | 351.07 | 1.002x |
| 320 | 8 | 359.55 | 357.90 | 1.005x |
| 512 | 1 | 377.42 | 377.42 | 1.000x |

Paired bootstrap intervals for the selected wall-time winner exclude zero at
ranks 1, 8, 32, and 128. These are exploratory comparisons selected from the
same sweep, so a targeted rerun should confirm a production setting. Even where
present, the complete-request improvement is only about 1%.

Large split factors remain a poor general policy. At rank 512, `K=64` raises
prefill latency from `351.24` to `399.24 ms` and wall time from `377.42` to
`424.66 ms`. For batch 32, a moderate `K=2..8` is worth testing for a fixed
production rank, but this sweep does not justify a universal `K>1` setting for
decode.
