# Mistral 7B On RTX 5090

`mistral_7b_rtx5090.json` contains a complete rank/split-K sweep run on
September 26, 2026.

This is a historical result from the earlier decode-focused harness, which
fixed prefill at `split_k=1`. The current harness applies one split factor to
both prefill and decode so it matches deployment behavior. Rerun the benchmark
before comparing against the current defaults.

The controlled RTX 5090 sweeps with synchronous one-step decode timing are
documented in
[`mistral_7b_rtx5090_batch32_two_tokens.md`](mistral_7b_rtx5090_batch32_two_tokens.md)
and
[`mistral_7b_rtx5090_batch64_two_tokens.md`](mistral_7b_rtx5090_batch64_two_tokens.md).

The earlier batch-8, two-token RTX 5080 sweep is documented in
[`mistral_7b_rtx5080_batch8_two_tokens.md`](mistral_7b_rtx5080_batch8_two_tokens.md).
It used vLLM's default asynchronous scheduler, so its request-level decode
intervals are not directly comparable with the synchronous batch-64 results.

## Configuration

```text
model            mistralai/Mistral-7B-Instruct-v0.3
GPU              NVIDIA GeForce RTX 5090, compute capability 12.0
vLLM             0.30.0
Transformers     5.17.0
PyTorch          2.13.0+cu130
dtype            bfloat16
batch size       1
prompt tokens    128
output tokens    32
ranks            8, 32, 128, 512
split factors    1, 2, 4, 8, 16, 32, 64
warmups          2 per point
timed trials     7 per point
```

Prefill used `split_k=1` for every run. Only decode-sized shrink calls used the
split factor under test. Numerical comparisons used eight greedy decode steps
and two executions per split factor.

## Performance Summary

The table selects the lowest median decode latency for each rank and stream
mode. Speedup is the median `K=1` latency divided by the best median latency.

| Stream | Rank | Best K | K=1 ms/token | Best ms/token | Speedup |
|---|---:|---:|---:|---:|---:|
| single | 8 | 1 | 12.752 | 12.752 | 1.000x |
| single | 32 | 32 | 13.510 | 13.468 | 1.003x |
| single | 128 | 32 | 13.168 | 12.993 | 1.013x |
| single | 512 | 32 | 15.245 | 13.891 | 1.097x |
| dual | 8 | 4 | 17.471 | 16.946 | 1.031x |
| dual | 32 | 16 | 17.465 | 17.380 | 1.005x |
| dual | 128 | 2 | 17.445 | 17.400 | 1.003x |
| dual | 512 | 4 | 17.620 | 17.505 | 1.007x |

The rank-512 single-stream result is the only large, clean effect: every one of
the seven `K=32` samples was below every `K=1` sample. The smaller-rank gains
and all dual-stream gains overlap their trial ranges, so their exact winning K
should not be treated as significant.

## Interpretation

The initial output-grid argument is incomplete at end-to-end scale. Increasing
rank does create more output tiles, but it also makes LoRA shrink a larger part
of the model step. At rank 8, shrink is highly under-parallelized but so small
that changing it barely moves full-model latency. At rank 512, an unsplit
single-request projection still has only `ceil(512 / 16) = 32` output tiles on
a 170-SM GPU, and shrink is now expensive enough that `K=32` produces a visible
9.7% decode improvement.

The experimental dual-stream path did not improve absolute latency on this
software/hardware combination. It was slower than single-stream at every rank,
and changing split-K had little additional effect. This does not establish a
general dual-stream rule: the per-point ranges overlap, and stream scheduling
is hardware- and vLLM-version-specific.

These results therefore do **not** support the simple prediction that the best
split factor must fall as rank increases. Two competing effects matter:

1. Higher rank supplies more independent output tiles, reducing the need to
   split the reduction.
2. Higher rank makes shrink more expensive, increasing the end-to-end value of
   accelerating it.

On this workload, the second effect dominates at rank 512.

## Numerical Summary

All first-step logits matched exactly, confirming that fixed prefill removed KV
cache construction as a confound. Later single-stream decode steps showed
maximum absolute raw-logit differences up to:

| Rank | Largest observed difference |
|---:|---:|
| 8 | 0.257812 |
| 32 | 0.375000 |
| 128 | 0.328125 |
| 512 | 0.218750 |

For rank 8, several split factors changed the eighth greedy token. Other ranks
kept the same eight-token trajectory in this run. Several configurations also
showed nonzero repeat-to-repeat drift, which is consistent with unordered
atomic accumulation.

Numerical tests were limited to single-stream workers. A seven-trial
dual-stream performance sweep was stable, but adding full-vocabulary logit
collection afterward caused a native crash in one seven-trial run on the tested
vLLM stack. The benchmark exposes `--numeric-stream-modes single,dual` for
retesting that combination when the runtime changes.

The JSON file contains every timing sample, every numerical comparison, the
generated token IDs, worker configuration, environment versions, and observed
vLLM shrink shapes.
