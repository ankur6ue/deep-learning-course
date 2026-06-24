# v5: Packed Projections, Cached RoPE, and Local Kernels

`v5` keeps the `v4` paged FlashAttention backend and optimizes the transformer body around it.

The goal of this step is to reduce overhead outside attention without introducing `torch.compile` yet.

```text
v4: separate Q/K/V and gate/up projections, readable RoPE, general prefill KV writes
v5: packed projections, cached RoPE, fused local kernels, slot-mapped prefill KV writes
```

`torch.compile` starts in `v6`, not here.

## What Changed From v4

- Q/K/V are stored as one packed `qkv_proj` weight.
- Gate/up MLP projections are stored as one packed `gate_up_proj` weight.
- RoPE uses a reusable cos/sin cache instead of rebuilding frequencies each step.
- A Triton RoPE kernel applies cached RoPE to Q/K when CUDA is available.
- RMSNorm, fused add+RMSNorm, SwiGLU, and slot-mapped K/V writes use local Triton kernels when available.
- The custom RoPE path introduces an important CUDA/Triton concept: Python can
  move on after launching a GPU kernel, while that kernel may still be reading
  its input tensors. `v5` therefore uses cloned position tensors and a small
  keepalive list for the tensors passed to the RoPE kernels.
- Prefill can write K/V through `prefill_slot_mapping` instead of looping through the generic request/chunk writer.

## Packed QKV Projection

Earlier versions use three separate linear layers:

```text
q_proj(hidden) -> Q
k_proj(hidden) -> K
v_proj(hidden) -> V
```

`v5` packs those into one matrix:

```text
qkv_proj(hidden) -> [Q | K | V]
```

Example:

```text
num_q_heads  = 8
num_kv_heads = 2
head_dim     = 64

Q width = 8 * 64 = 512
K width = 2 * 64 = 128
V width = 2 * 64 = 128

qkv_proj output width = 512 + 128 + 128 = 768
```

The output is split like this:

```python
q_flat, k_flat, v_flat = qkv.split([512, 128, 128], dim=-1)
```

Edge cases:

- Grouped-query attention has fewer KV heads than query heads. That is why K and V are narrower than Q.
- The checkpoint loader detects whether a version uses separate or packed weights by looking at the model state dict, then loads Hugging Face weights accordingly.

## Packed Gate/Up Projection

The Llama/Mistral MLP computes:

```text
gate = gate_proj(x)
up   = up_proj(x)
mlp  = down_proj(silu(gate) * up)
```

`v5` packs `gate_proj` and `up_proj` into one matrix:

```text
gate_up_proj(x) -> [gate | up]
```

Example:

```text
intermediate_size = 14336

gate_up_proj output width = 2 * 14336
```

Then:

```python
gate, up = gate_up.chunk(2, dim=-1)
```

The fused SwiGLU kernel can compute `silu(gate) * up` without launching extra tiny elementwise kernels.

## RoPE Cos/Sin Cache

Earlier versions rebuild RoPE frequencies for the positions in each step. `v5` builds reusable cosine/sine tables once and indexes them by absolute token position.

Example:

```text
positions = [5, 6, 7]
head_dim = 128

cos_cache[positions] -> [3, 64]
sin_cache[positions] -> [3, 64]
```

The RoPE kernel uses those rows to rotate Q/K.

The tradeoff is that `v5` now has an explicit maximum served sequence length:

```text
prompt_len + max_new_tokens <= max_position_embeddings
```

Example:

```text
max_position_embeddings = 4096
prompt_len              = 3800
max_new_tokens          = 256

3800 + 256 = 4056  -> allowed
```

But:

```text
max_position_embeddings = 4096
prompt_len              = 4000
max_new_tokens          = 256

4000 + 256 = 4256  -> rejected at request submission
```

In `v4`, RoPE values were rebuilt from the actual positions used in each step,
so there was no prebuilt RoPE table to run out of. The practical limit was
mostly KV-cache capacity: `num_blocks * block_size` total cache slots shared
across active requests. In `v5`, cached RoPE is faster, but the cache must be
large enough for the largest prompt-plus-generation length this engine serves.

Edge case if this validation were missing:

```text
position >= cos_cache.shape[0]
```

The cache is too short for the requested context length. The explicit request
check prevents this from becoming an indexing failure inside the RoPE kernel.

## Async Triton Kernel Lifetime

Earlier versions mostly use PyTorch operations. PyTorch CUDA operations are also
asynchronous, but PyTorch owns the operator implementation and normally manages
the obvious tensor lifetime details for us. In `v5`, we start launching our own
Triton kernels, so the code has to be more explicit about what the GPU may still
be reading after Python has returned from the launch call.

The important rule:

```text
Launching a CUDA/Triton kernel queues GPU work.
It does not mean the GPU has already finished reading the input tensors.
```

In the RoPE path:

```python
q_triton = triton_apply_rope_from_cache(q_2d, flat_positions, cos_sin_cache, ...)
k_triton = triton_apply_rope_from_cache(k_2d, flat_positions, cos_sin_cache, ...)
```

The Triton kernels read:

```text
q_2d
k_2d
flat_positions
cos_sin_cache
```

and write:

```text
q_triton
k_triton
```

The CPU does not wait for those kernels to finish. It continues through the
model code and may soon schedule the next step.

### Why `flat_positions.clone()`?

This helper is used by both prefill and decode.

In prefill, the positions tensor is freshly allocated for that prefill batch:

```python
positions = torch.zeros((len(work_items), max_chunk), device=device, dtype=torch.long)
```

That tensor does not live across scheduler steps.

In decode, the positions tensor is a slice of a reusable workspace:

```python
self.positions_workspace = torch.empty((max_decode_batch_size, 1), ...)
positions = self.positions_workspace[:batch_size]
```

A workspace is a tensor buffer that gets filled with new values every scheduler
step.

Example:

```text
step 10 positions workspace = [40, 91, 12]
step 11 positions workspace = [41, 92, 13]
```

If a custom kernel from step 10 is still reading the positions buffer when step
11 overwrites that same buffer, the kernel can read the wrong positions. To make
the shared RoPE helper safe for the decode case, `v5` always makes a private
copy before launching the Triton kernel:

```python
flat_positions = positions.reshape(-1).contiguous().clone()
```

For prefill, this clone is conservative but not primarily about workspace reuse.
For decode, it means the step 10 RoPE kernel reads the step 10 positions even if
the decode workspace is later reused for step 11.

### Why `_rope_keepalive`?

Some tensors passed to the kernel are views or short-lived temporaries:

```python
q_2d = q_flat.reshape(...)
k_2d = k_flat.reshape(...)
```

The kernel receives raw device pointers. Keeping Python tensor references makes
it clear that this storage must stay alive while recently launched kernels may
still be using it:

```python
self._rope_keepalive.extend(
    (q_2d, k_2d, flat_positions, cos_sin_cache, q_triton, k_triton)
)
```

This list is not part of the RoPE math. It is a simple teaching-engine lifetime
guard. After many launches, old entries are removed:

```python
if len(self._rope_keepalive) > 4096:
    del self._rope_keepalive[:2048]
```

By then, those older kernels have completed.

### Same Stream vs Other Streams

CUDA preserves ordering within one stream:

```text
kernel A
kernel B
```

If both are queued on the same stream, `kernel B` will not run before `kernel A`.
That ordering helps a lot.

But as soon as code uses reusable buffers, custom kernels, CUDA graphs, side
streams, or non-blocking copies, it is easy to create situations where Python
state has advanced while GPU work from an earlier step is still in flight.
The safe mental model is:

```text
If a queued GPU operation still needs a tensor, keep that tensor alive.
If a queued GPU operation needs the old values, do not pass a mutable workspace;
pass an owned copy.
```

Runnable tutorials for this concept live in:

```text
Module 8/class_8.1/tutorials
```

## `prefill_slot_mapping`

`v4` introduced `decode_slot_mapping`, where each request writes exactly one new K/V row. Prefill is different: each request may write several prompt-token rows, and different requests may contribute different chunk lengths.

Example with `block_size = 16`:

```text
request 0 past_len = 14, query_len = 3, block_ids = [7, 12]
request 1 past_len =  4, query_len = 1, block_ids = [3]

max_query_len = 3
```

Physical slots:

```text
request 0 token positions 14, 15, 16 -> slots 126, 127, 192
request 1 token position  4         -> slot 52
```

Request 0 crosses a block boundary in the normal way: token positions `14` and
`15` live in physical block `7`, while token position `16` starts the next
logical block, which maps to physical block `12`.

So:

```python
prefill_slot_mapping = torch.tensor([
    [126, 127, 192],
    [ 52,  -1,  -1],
])
```

`-1` means: this padded query cell is not a real token, so do not write it to KV cache.

Edge cases:

- If all requests have the same chunk length, there are no `-1` cells and the write kernel can skip the validity mask.
- During decode, this structure is not used. Decode stays with the simpler one-dimensional `decode_slot_mapping` from `v4`.

## Next Version

`v6` keeps these local optimizations and adds `torch.compile` around selected pure tensor regions.

## Validation

Smoke test:

```bash
python "Module 8/class_8.1/simple_vllm_engine/scripts/smoke_test_versions.py" --versions v5
```

Short real-model sanity run, Llama 3.1 8B, shared six-request workload, `max_new_tokens=8`, prefix cache disabled:

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v5 | 1.272 | 37.73 | yes | no | +6.8% | +300.1% |

The local model-body changes are directionally useful but modest on this short workload. The larger jump comes in `v6` when `torch.compile` is added around pure tensor regions.
