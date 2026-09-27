# LoRA Rank And Split-K In vLLM

This experiment measures how LoRA rank changes the useful split-K factor during
real Mistral inference. It uses vLLM's production LoRA implementation and its
existing Punica/Triton shrink kernel. There is no custom split-K kernel in this
directory.

The default model is `mistralai/Mistral-7B-Instruct-v0.3`. The benchmark applies
a deterministic synthetic LoRA to `q_proj`, `k_proj`, `v_proj`, and `o_proj` in
all 32 transformer layers, generates tokens through vLLM, and varies only the
`split_k` value selected by vLLM's shrink operator.

## Setup

Use a dedicated environment so upgrading vLLM does not change the environments
used by earlier course modules:

```bash
uv venv --python 3.12 ~/dev/.venv-vllm-split-k
uv pip install \
  --python ~/dev/.venv-vllm-split-k/bin/python \
  -r "Module 10/split-k/requirements.txt"
source ~/dev/.venv-vllm-split-k/bin/activate
```

The pinned vLLM 0.30.0 installation currently resolves to Transformers 5.17.0
and PyTorch 2.13.0. The Mistral checkpoint is about 15 GB and is downloaded by
Hugging Face on the first run.

Run the default sweep:

```bash
python "Module 10/split-k/lora_rank_vs_split_k.py" \
  --output "Module 10/split-k/benchmark_results/mistral_7b.json"
```

The default experiment uses:

```text
ranks          = 8, 32, 128, 512
split_k        = 1, 2, 4, 8, 16, 32, 64
batch size     = 8
prompt length  = 128 tokens
output length  = 2 tokens
warmups        = 2 per split factor
timed trials   = 7 per split factor
stream mode    = single
async scheduler = disabled
numeric mode   = single stream only
```

A short plumbing check is useful before a full sweep:

```bash
python "Module 10/split-k/lora_rank_vs_split_k.py" \
  --ranks 8 \
  --split-k 1,8 \
  --batch-size 2 \
  --output-tokens 2 \
  --trials 1 \
  --numeric-tokens 1
```

## What Is Being Varied

For one LoRA projection, vLLM first computes the narrow or *shrink* projection:

```text
hidden [M, H] @ lora_A.T [H, r] -> low_rank [M, r]
```

During pure decode, vLLM flattens one token from each active request, so `M=B`.
With vLLM's default shrink tile dimensions of `BLOCK_M=32` and `BLOCK_N=16`,
an unsplit invocation has:

```text
output tiles = ceil(M / 32) * ceil(r / 16)
```

Rank 8 or 16 therefore gives only one output tile per projection slice. Split-K
divides the reduction over `H` among several programs and atomically accumulates
their FP32 partial sums. That adds parallel work when the output grid is too
small to occupy the GPU, but also adds program-launch and atomic-reduction cost.

A batch larger than one does not immediately create more Triton programs. For
`1 <= B <= 32`, `ceil(B / BLOCK_M)` remains one: the batch fills more rows in
the existing tile. `B=33` creates the second M tile. A larger batch can still
change kernel efficiency and resource use, but the launch grid gains direct
M-axis parallelism only when the batch crosses a tile boundary.

Increasing rank adds output tiles directly. The expected trend is therefore:

```text
small rank  -> more benefit from splitting H
large rank  -> enough output tiles already, so a smaller split should win
```

This is a hypothesis to measure, not an assumption built into the report. The
crossover depends on the GPU, batch size, vLLM version, kernel configuration,
and how much of total inference time the shrink kernels occupy.

vLLM uses split-K only for the shrink projection. The following wide *expand*
projection fixes `split_k=1` because its output dimension already supplies many
tiles:

```text
low_rank [M, r] @ lora_B.T [r, H_out] -> delta [M, H_out]
```

## Short-Output Guardrails

For a next-token guardrail, the prompt forward pass produces logits for the
classification token. Output length has an important consequence:

```text
output tokens = 1  -> prefill produces the classification token; no decode pass
output tokens = 2  -> prefill produces token 1; one decode pass produces token 2
```

For one-token classification, prefill and end-to-end latency are the production
metrics; decode split-K is never exercised. During prefill, `M` is the total
number of scheduled prompt-token rows, often approximately `B * prompt_length`,
so it normally offers much more M-axis parallelism than decode. The default
uses eight requests and two output tokens to retain one production-shaped
decode step. Use `--output-tokens 1` to reproduce a strictly one-token
classifier.

## Why Each Rank Gets A New Engine

vLLM stores adapter weights and intermediate buffers at `max_lora_rank`. Loading
a rank-8 adapter into an engine configured with `max_lora_rank=512` would still
exercise rank-512 buffers and kernel shapes. That would not measure rank 8.

The coordinator starts one worker process per rank. A worker sets
`max_lora_rank` to the exact adapter rank, loads Mistral once, and sweeps all
split factors while the model remains resident. The process exits before the
next rank so GPU memory is released cleanly. When `--output` is set, the
coordinator updates the aggregate JSON after every worker; `complete=false`
identifies a partial result if a later worker fails.

## Real vLLM Kernel Path

The script wraps vLLM's `get_lora_op_configs` function and changes only the
`split_k` field when `op_type == "shrink"`. The selected value applies to every
shrink invocation in the measured generation, including prefill and decode.
This matches an inference deployment configured with one kernel policy rather
than giving prefill special benchmark-only treatment. Everything else is vLLM
inference:

```text
vLLM scheduler
Mistral forward pass
vLLM attention and base-model kernels
vLLM LoRA routing metadata
vLLM Punica shrink and expand kernels
vLLM sampler
```

Every measured generation verifies that the hook was called and that the
kernel's rank equals the requested rank. It saves all observed shrink shapes in
the JSON output. This makes a future vLLM internal API change fail explicitly
instead of silently benchmarking a fixed default. vLLM's initialization dummy
forwards run at the controller's initial `split_k=1`; `controller.begin()` then
sets the measured value and clears those initialization observations.

The engine uses eager execution because `split_k` is a compile-time Triton
constant. CUDA graphs or `torch.compile` could capture one value and replay it
after the Python-side value changes. Eager mode keeps the sweep valid, and all
split factors pay the same execution mode cost.

The benchmark also disables vLLM's asynchronous scheduler by default. That
scheduler can keep the prefill and decode batches in flight together, then
process both outputs close together. With only two generated tokens, the
request's first-to-second-token timestamps can consequently be much shorter
than the actual decode model step. Synchronous scheduling makes `decode
ms/tok` cover the one decode pass. Use `--async-scheduling` when measuring
production-style end-to-end throughput, but do not interpret its one-step
request interval as standalone decode latency.

## Single And Dual Streams

vLLM's default `single` mode executes each base projection and its LoRA work on
the same CUDA stream. The operations are serial, so other model kernels are not
simultaneously occupying SMs. A faster shrink kernel can still help, but its
gain is diluted by attention, base GEMMs, expand projections, normalization,
sampling, and scheduler overhead.

The optional `dual` mode sets `VLLM_LORA_ENABLE_DUAL_STREAM=1`. vLLM can then
overlap LoRA work on an auxiliary CUDA stream with the base projection. This is
the mode that tests whether other model work consumes enough SM capacity to
make a large split factor saturate sooner than it does in an isolated kernel:

```bash
python "Module 10/split-k/lora_rank_vs_split_k.py" \
  --stream-modes single,dual \
  --ranks 8,32,128,512 \
  --split-k 1,2,4,8,16,32,64
```

Dual-stream LoRA is an experimental vLLM path. Report both modes rather than
attributing a dual-stream result to default vLLM behavior.

## Synthetic Adapters

A trained adapter cannot be resized to several ranks without changing its
meaning. The benchmark therefore creates one deterministic PEFT-format adapter
per rank. It scales random weights as:

```text
std(A) = 1 / sqrt(input_size)
std(B) = adapter_scale / sqrt(rank)
lora_alpha / rank = 1
```

This keeps the expected update magnitude roughly stable across ranks. The
adapters are saved under `~/.cache/deep-learning-course/split-k` and reused on
later runs. With the default four ranks, they consume roughly 1.2 GB in total.

These weights are suitable for performance and numerical-drift experiments;
they are not a trained language adapter and generated text quality is not a
benchmark result.

## Reported Performance

Each split factor is warmed before timing so Triton compilation and first-time
adapter loading are excluded. Trial order is shuffled to reduce thermal and
ordering bias. The report includes:

- `wall ms`: full `LLM.generate` time, including prefill, decode, and host work.
- `wall tok/s`: all generated tokens divided by wall time.
- `prefill ms`: median scheduled-to-first-token interval reported by vLLM.
- `time_to_first_token_ms`: prefill plus any request queue time, saved in JSON.
- `decode ms/tok`: median request inter-token latency after the first token.
- `prefill speedup`: `split_k=1` prefill latency divided by current latency.
- `decode speedup`: `split_k=1` decode latency divided by the current latency.

With one output token, decode metrics are `null` because no decode forward took
place. With two output tokens, decode latency contains one interval per request
and is therefore noisier than a long-generation benchmark; use more timed
trials when small differences matter. The default synchronous scheduler is
required for this interval to represent the model's decode pass. For a
one-token guardrail, compare `prefill ms` and `wall ms`, not decode latency.

Batch size changes `M` and creates parallelism independently of split-K:

```bash
python "Module 10/split-k/lora_rank_vs_split_k.py" \
  --batch-size 8 \
  --output-tokens 2 \
  --ranks 8,32,128,512
```

## Numerical Drift

For each rank and split factor, the benchmark also asks vLLM for full-vocabulary
raw logits over a short greedy generation. Because the chosen split factor now
applies during prefill, numerical drift can appear in the first classification
logits and in the KV cache. The script compares each run with `split_k=1` and
reports:

- maximum and mean absolute logit difference;
- first-step and last-comparable-step differences;
- the first generated token that differs; and
- run-to-run drift for repeated execution at the same split factor.

Atomic FP32 additions can arrive in different orders for `split_k > 1`.
Floating-point addition is not associative, so both split-versus-unsplit drift
and run-to-run drift are possible. Applying LoRA in every attention layer lets
small layer-level differences propagate through the full 32-layer model and
become visible in final logits.

If greedy token trajectories diverge at step `i`, logits at step `i` are still
comparable because both runs saw the same preceding tokens. Later steps use
different inputs, so the script excludes them from the direct logit comparison.
Use `--numeric-tokens 0` to skip this pass.

Numerical comparison defaults to single-stream workers. On the tested vLLM
0.30.0 stack, requesting full-vocabulary logits after a long dual-stream sweep
caused a native process crash, although dual-stream performance-only sweeps were
stable. Use `--numeric-stream-modes single,dual` to opt into that experimental
combination on another stack.

## Reading The Result

For a one-token classifier, find the split factor with the lowest prefill and
wall latency. For a two-token run, inspect decode latency separately and ask
whether its best factor moves toward 1 as batch size or rank increases. Compare
single and dual streams separately. Also check the absolute gain: a large
shrink-kernel speedup can produce only a small end-to-end change when shrink is
a small part of the complete model step.

Keep the hardware, batch size, context length, output length, dtype, vLLM
version, and stream mode fixed when comparing ranks. The JSON output records
those settings, every raw timing sample, package versions, GPU model, compute
capability, adapter path, and observed vLLM kernel shapes.
