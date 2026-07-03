# v10: Architecture Adapters for Gemma 4

`v10` starts from `v9` and keeps the same serving-system ideas: chunked
prefill, paged KV cache, paged attention metadata, GPU-resident decode state,
and async output handling. The new teaching target is model architecture
variation. Gemma 4 text checkpoints are close enough to decoder-only serving to
fit this engine, but not close enough to be treated as "Llama with different
numbers."

## What Changed From v9

- `ModelConfig` now has an explicit `architecture` field.
- Attention head dimension and KV-head count can vary by layer.
- Layer types identify `sliding_attention` versus `full_attention`.
- RoPE parameters can vary by layer type.
- The loader can unwrap Gemma 4 unified configs and load a single
  `model.safetensors` file without an index.
- The model body has a dedicated `Gemma4Layer` path with Gemma-specific norms,
  residual order, MLP activation, tied embeddings, layer scalars, and logit
  softcapping.
- The default attention backend is `triton_paged`, using vLLM's Triton unified
  attention wrapper for mixed head sizes and sliding-window paged attention.
- CUDA graph replay supports both `flash_attn_paged` and `triton_paged` when
  `--enable-decode-cuda-graphs` is set. The graph path covers decode and the
  all-valid prefill bucket case.

## Why This Helps

Earlier versions intentionally assume one Llama/Mistral-style layer shape:

```text
num_attention_heads is fixed
num_key_value_heads is fixed
head_dim is fixed
one RoPE table works for every layer
Q/K/V are packed into one projection
```

Gemma 4 breaks those assumptions:

```text
sliding-attention layers: local window, head_dim=256, 8 KV heads
full-attention layers: global attention, head_dim=512, 1 KV head
attention_k_eq_v: full layers reuse K projection output as raw V input
per-layer-type RoPE: default for sliding layers, proportional for full layers
```

`v10` keeps those differences explicit instead of hiding them behind a large
generic abstraction. That makes it easier to compare with the HF/vLLM model
definitions while preserving the course's scheduler and KV-cache code.

## Gemma Optimization Switches

The Gemma path keeps several optimizations behind explicit environment flags so
benchmarks can compare accuracy and speed tradeoffs:

```bash
SIMPLE_VLLM_GEMMA_OPTIMIZE=1
SIMPLE_VLLM_GEMMA_FAST_RMSNORM=all
SIMPLE_VLLM_GEMMA_VLLM_GELU_GATE=1
```

- `SIMPLE_VLLM_GEMMA_OPTIMIZE=1` enables the grouped Gemma fast path switches.
- `SIMPLE_VLLM_GEMMA_FAST_RMSNORM=all` uses the fast RMSNorm variants for the
  Gemma norms covered by the implementation.
- `SIMPLE_VLLM_GEMMA_VLLM_GELU_GATE=1` uses vLLM's fused GELU-tanh gated MLP op
  when that runtime op is available.

## Validation

Use Transformers greedy output as the correctness reference:

```bash
/home/ankur/dev/.venv-trl/bin/python Module\ 8/class_8.1/simple_vllm_engine/scripts/generate_transformers_groundtruth.py \
  --model-path /home/ankur/dev/models/gemma-4-12B-it-BF16-Fresh \
  --workload-path Module\ 8/class_8.1/simple_vllm_engine/workloads/long_context_serving_workload.json \
  --max-new-tokens 100
```

Run `v10` and production vLLM on the same workload:

```bash
SIMPLE_VLLM_GEMMA_OPTIMIZE=1 \
SIMPLE_VLLM_GEMMA_FAST_RMSNORM=all \
SIMPLE_VLLM_GEMMA_VLLM_GELU_GATE=1 \
/home/ankur/dev/.venv-trl/bin/python Module\ 8/class_8.1/simple_vllm_engine/scripts/benchmark_versions_against_vllm.py \
  --model-path /home/ankur/dev/models/gemma-4-12B-it-BF16-Fresh \
  --workload-path Module\ 8/class_8.1/simple_vllm_engine/workloads/long_context_serving_workload.json \
  --versions v10 \
  --attention-backend triton_paged \
  --max-batch-tokens 4096 \
  --max-prefill-chunk-tokens 4096 \
  --max-decode-batch-size 8 \
  --max-model-len 4096 \
  --max-new-tokens 100 \
  --enable-decode-cuda-graphs \
  --vllm-gpu-memory-utilization 0.95 \
  --results-dir Module\ 8/class_8.1/simple_vllm_engine/benchmark_results_gemma4_v10_long_context_100
```

Compare generated token IDs:

```bash
/home/ankur/dev/.venv-trl/bin/python Module\ 8/class_8.1/simple_vllm_engine/scripts/compare_generation_outputs.py \
  --reference Module\ 8/class_8.1/simple_vllm_engine/workloads/groundtruth/gemma-4-12B-it-BF16-Fresh_long_context_serving_workload_transformers_greedy_100.json \
  --candidate Module\ 8/class_8.1/simple_vllm_engine/benchmark_results_gemma4_v10_long_context_100/v10.json \
  --candidate Module\ 8/class_8.1/simple_vllm_engine/benchmark_results_gemma4_v10_long_context_100/vllm.json
```

See `GEMMA_V10_WORKLOG.md` in the parent directory for current run notes and
resume commands.
