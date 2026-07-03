# Teaching Sequence Notes

The sequence is cumulative, but each version should introduce one main idea. Avoid carrying old fallback paths forward unless the fallback is itself the teaching target.

## Current Sequence

```text
v1  serving control plane
v2  direct paged reference attention
v3  gathered SDPA attention
v4  paged FlashAttention backend
v5  packed projections, cached RoPE, local kernels, prefill slot mapping
v6  torch.compile for pure tensor model-body regions
v7  async GPU/CPU control-plane communication
v8  decode and all-valid prefill CUDA graph replay
v9  GPU-resident decode metadata
v10 architecture adapters for Gemma 4 text checkpoints
```

## v1: Serving Control Plane

The first version teaches the serving loop before optimizing kernels.

Core concepts:

- `RequestSpec`: immutable input request.
- `RequestState`: mutable runtime state.
- Prefill: compute K/V for prompt tokens.
- Decode: generate one token per active request.
- Decode-first scheduling: finished prompts decode before more prefill is scheduled.
- Chunked prefill: long prompts are split so they do not monopolize the batch.
- Paged KV cache: each request owns a logical-to-physical block table.

## v2: Direct Paged Reference Attention

This version makes the paged KV abstraction explicit.

Core concepts:

- Each request has `block_ids`: logical block index -> physical cache block.
- Attention walks the paged cache directly instead of first creating dense K/V.
- `past_lens`, `query_lens`, and `key_lens` define what each request can see.
- The implementation is intentionally slow but maps directly to the mental model.

## v3: Gathered SDPA Attention

This version shows the common intermediate implementation: keep the persistent KV cache paged, but temporarily gather dense tensors before attention.

Core concepts:

- Gather cached K/V pages into dense `k_past` / `v_past` tensors.
- Assemble `k_full` / `v_full` as `[cached prefix | current chunk | padding]`.
- Build a causal/padding mask.
- Call PyTorch `scaled_dot_product_attention`.

This step is easier to understand than an optimized paged-attention kernel, but it adds extra memory movement.

## v4: Paged FlashAttention Backend

This version changes only the attention mechanism.

Core concepts:

- FlashAttention reads K/V through `block_tables_i32` instead of dense gathered K/V.
- `cu_seqlens_q` describes packed query offsets.
- `valid_query_rows` maps padded batch layout to FlashAttention's flat query layout.
- `decode_slot_mapping` lets decode write one token's K/V directly to its physical cache slot, still using PyTorch `index_copy_`.
- Prefill intentionally still uses the general `write_batch()` -> `write_tokens()` path.
- Projections remain separate: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`.

What is intentionally not here:

- Packed QKV projection.
- Packed gate/up projection.
- Cached/fused RoPE.
- Prefill slot mapping.
- `torch.compile`.

## v5: Packed Projections, Cached RoPE, and Local Kernels

This version optimizes model-body work around the FlashAttention boundary.

Core concepts:

- Pack Q/K/V into one `qkv_proj` matrix.
- Pack MLP gate/up into one `gate_up_proj` matrix.
- Build a reusable RoPE cos/sin cache.
- Apply RoPE with a local Triton kernel when CUDA is available.
- Use local kernels for RMSNorm, fused residual-add RMSNorm, SwiGLU, and slot-mapped K/V writes.
- Add `prefill_slot_mapping`, a padded 2D physical-slot map for prompt chunks, computed once per batch before the layer loop.
- Use a Triton cache-write kernel for slot-mapped K/V copies when CUDA inputs are compatible.

What is intentionally not here:

- `torch.compile`. That starts in `v6`.

## v6: `torch.compile` Model Body

This version introduces graph compilation where it is safe and explainable.

Core concepts:

- Compile pure tensor blocks only.
- Keep attention, KV cache mutation, scheduler metadata, and request objects outside compile.
- Default compile scope is `mlp` because it is the least surprising pure tensor subgraph.
- Wider scopes exist for experiments: `input_qkv`, `tail`, and `all`.

## v7: Async GPU/CPU Control Plane

This version removes sampled-token CPU synchronization from the decode hot path.

Core concepts:

- Store latest sampled token IDs in a GPU table.
- Map request IDs to stable GPU slots.
- Build next decode inputs from GPU state.
- Copy sampled tokens back to CPU asynchronously for output/EOS bookkeeping.

## v8: CUDA Graph Replay

This version adds graph replay after the eager path is already optimized.

Core concepts:

- Fixed-shape decode graph buckets for one-token decode batches.
- Fixed-shape prefill graph buckets for the simple all-valid prompt-chunk case.
- Ragged prefill chunks still run eager, because padding and masking those
  variable shapes would obscure the graph concept.
- Static input/output/metadata tensors.
- Scratch rows and scratch KV blocks for padding.
- Replay mutates metadata values in place while tensor addresses stay stable.

## v9: GPU-Resident Decode Metadata

This version removes a remaining Python/H2D decode cost after graph replay.

Core concepts:

- Keep request block tables in a persistent GPU tensor keyed by request slot.
- Update a request's GPU block-table row only when `req.block_ids` grows.
- Stage only request slots and cached sequence lengths each decode step.
- Use a Triton kernel to gather the active block-table rows into FlashAttention's batch metadata.

## v10: Architecture Adapters for Gemma 4

This version keeps the v9 serving/dataflow structure but makes the model layer
shape configurable enough for Gemma 4 text checkpoints.

Core concepts:

- Carry an explicit `architecture` field in `ModelConfig`.
- Allow per-layer attention head dimensions and KV-head counts.
- Represent Gemma layer types so sliding-attention and full-attention layers can use different RoPE parameters.
- Add Gemma's Q/K/V normalization, four RMSNorms per layer, tied embeddings, logit softcapping, GELU-tanh gated MLP, and learned layer scalars.
- Use vLLM's Triton unified attention wrapper for the mixed head-size/sliding-window paged attention cases that the FlashAttention path does not cover cleanly.

## Current Short Benchmark Snapshot

Short Llama 3.1 8B sanity run, six requests, `max_new_tokens=8`, prefix cache disabled:

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v4 | 1.365 | 35.17 | n/a | no | n/a | +329.2% |
| v5 | 1.272 | 37.73 | yes | no | +6.8% | +300.1% |
| v6 | 0.439 | 109.39 | yes | no | +65.5% | +38.0% |
| v7 | 0.439 | 109.24 | yes | no | -0.1% | +38.2% |
| v8 | 0.431 | 111.40 | yes | no | +1.9% | +35.5% |

The big measured jump in this short workload is `v6`, not `v5`. That is useful pedagogically: the local layout/kernel changes are concrete and general, but the compiler removes a much larger amount of model-body overhead here.

## Pending Follow-Up: GPU Prefill Slot Mapping

`v5` builds `prefill_slot_mapping`, but it still builds it with Python loops. Moving that mechanical mapping to a Triton kernel is a good future optimization.

Current CPU-side idea:

```text
for each request:
  for each query token in this prefill chunk:
    token_pos = past_len + query_offset
    slot = physical_slot(block_table, token_pos)
```

GPU-side idea:

```text
token_pos    = past_len[row] + query_offset
block_index  = token_pos // block_size
block_offset = token_pos % block_size
physical_blk = block_table[row, block_index]
slot         = physical_blk * block_size + block_offset
slot         = -1 if query_offset >= query_len[row]
```

This is a general optimization, not benchmark cheating: it keeps the same block-table abstraction while moving mechanical per-token mapping out of Python.
