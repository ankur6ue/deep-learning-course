# v6: torch.compile Around the Optimized Model Body

`v6` keeps the `v5` packed projections, cached RoPE, local kernels, and FlashAttention backend. It adds `torch.compile` for selected pure tensor regions of the transformer body.

The goal of this step is to teach the safe compile boundary:

```text
safe to compile:    deterministic tensor math
not compiled here:  scheduler, request objects, KV-cache allocation, KV-cache mutation, FlashAttention call
```

## What Changed From v5

- `EngineConfig` enables `enable_torch_compile_model_body` by default.
- `model.py` defines compile-safe helper functions for tensor-only work.
- The default compile scope is `mlp`, which compiles the packed SwiGLU `mlp` region.
- You can also choose larger compile scopes: `input_qkv`, `tail`, or `all`.
- The FlashAttention call and K/V cache writes are not passed to `torch.compile`.

## Compile Boundary

The serving forward contains Python objects and mutable cache state:

```text
RequestState objects
PagedKVCache allocator state
block tables and slot mappings
FlashAttention external kernel call
sampling and output bookkeeping
```

Those are intentionally not hidden behind `torch.compile` in this teaching step.

Instead, compiled helpers accept ordinary tensors and weights.

Example for the default `mlp` scope:

```text
input:  normed hidden states [tokens, hidden_size]
weights: gate_up_proj.weight, down_proj.weight
output: mlp result [tokens, hidden_size]
```

The helper computes:

```text
gate_up = linear(normed, gate_up_proj)
gate, up = split(gate_up)
activated = silu(gate) * up
out = linear(activated, down_proj)
```

## Compile Scopes

```text
mlp       -> packed gate/up projection, SwiGLU, down projection
input_qkv -> input RMSNorm + packed QKV projection; RoPE stays outside
tail      -> output projection + residual/RMSNorm + mlp
all       -> input_qkv + tail
```

The default is `mlp` because it is the least surprising: it is pure tensor math and does not interact with paged attention metadata.

## Edge Cases

- Different prefill chunk widths can create different compile specializations.
- Attention is not compiled here because it calls FlashAttention with block-table metadata.
- K/V writes are not compiled here because they update the shared paged cache.
- Larger compile scopes may be faster for one model/GPU and slower for another, so they stay as opt-in choices.

## Next Version

`v7` keeps the optimized and compiled model body, then reduces decode-side CPU synchronization with GPU-resident token state and asynchronous token copies.

## Validation

Smoke test:

```bash
python "Module 8/class_8.1/simple_vllm_engine/scripts/smoke_test_versions.py" --versions v6
```

Short real-model sanity run, Llama 3.1 8B, shared six-request workload, `max_new_tokens=8`, prefix cache disabled:

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v6 | 0.439 | 109.39 | yes | no | +65.5% | +38.0% |

This is the largest step in the new sequence for this workload: the model body is still explicit, but selected pure tensor regions are compiled by Inductor.
