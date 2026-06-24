# v4: Paged FlashAttention Backend

`v4` changes the attention mechanism, and intentionally leaves the rest of the transformer body close to `v3`.

The goal of this step is narrow:

```text
v3: gather paged K/V into dense tensors, then call PyTorch SDPA
v4: keep K/V paged and let FlashAttention read through block tables
```

Projection and MLP weights are still separate in this version:

```text
q_proj, k_proj, v_proj
 gate_proj, up_proj
```

Those are packed in `v5`, not here. Keeping them separate makes `v4` about one idea: replacing gathered SDPA with a paged FlashAttention backend.

## What Changed From v3

- `attention_backends.py` now exposes only `FlashAttentionPagedBackend`.
- The backend writes the current K/V chunk into the paged cache before attention.
- It flattens valid query rows and calls `flash_attn_varlen_func` with `cu_seqlens_q`, `key_lens`, and `block_tables_i32`.
- Decode batches build `decode_slot_mapping` so one-token K/V writes can go directly to physical cache slots.
- `model.py` still uses separate Q/K/V projections and separate gate/up MLP projections.

## Attention Flow

```text
Q, new K/V, paged cache, block table, lengths
        |
        v
write new K/V into paged cache
        |
        v
FlashAttention reads old + new K/V through block_table
        |
        v
attention output [B, Tq, Hq, D]
```

This matches the high-level shape of vLLM's normal CUDA path: one selected attention backend, with metadata describing where each request's tokens live in the paged KV cache.

## New Data Structures

### `cu_seqlens_q`

FlashAttention does not want a padded `[batch, max_query_len, heads, head_dim]` query tensor. It wants all valid query tokens packed into one flat tensor. `cu_seqlens_q` tells FlashAttention where each request starts in that flat query tensor.

Example:

```text
request 0 query_len = 3
request 1 query_len = 5

q padded shape = [2, 5, H, D]
q flat shape   = [8, H, D]

cu_seqlens_q = [0, 3, 8]
```

Read that as:

```text
request 0 queries live in q_flat[0:3]
request 1 queries live in q_flat[3:8]
```

Decode is the simplest edge case:

```text
request 0 query_len = 1
request 1 query_len = 1
request 2 query_len = 1

cu_seqlens_q = [0, 1, 2, 3]
```

### `valid_query_rows`

The model still builds padded query tensors because the rest of this teaching version is easier to read in `[batch, max_query_len, ...]` layout. FlashAttention only sees the valid rows.

Example:

```text
request 0 query_len = 3
request 1 query_len = 5

q shape = [2, 5, H, D]
```

`valid_query_rows` is:

```python
torch.tensor([
    [True,  True,  True,  False, False],
    [True,  True,  True,  True,  True ],
])
```

So:

```python
q_flat = q[valid_query_rows]
```

has shape `[8, H, D]` and row order:

```text
q[0, 0]
q[0, 1]
q[0, 2]
q[1, 0]
q[1, 1]
q[1, 2]
q[1, 3]
q[1, 4]
```

FlashAttention writes `out_flat` in the same packed order. We scatter it back:

```python
out = torch.zeros_like(q)
out[valid_query_rows] = out_flat
```

The final `out` shape is again `[2, 5, H, D]`:

```text
out[0, 0:3] = real request 0 attention outputs
out[0, 3:5] = zeros for padded rows
out[1, 0:5] = real request 1 attention outputs
```

The padded rows are later masked out before the residual update, so zeros are safe.

Important edge case:

```text
request 0 query_len = 4
request 1 query_len = 4
```

There is no padding. In that case `valid_query_rows` is `None`, and the backend can return:

```python
out_flat.view_as(q)
```

without allocating a zero-filled padded output tensor.

### `block_tables_i32`

The paged KV cache stores a request's logical blocks in arbitrary physical cache blocks. `block_tables_i32` is the int32 version of that mapping for the FlashAttention kernel.

Example with `block_size = 16`:

```text
request 0 block_ids = [7, 12]
request 1 block_ids = [3]

block_tables =
[
    [7, 12],
    [3, -1],
]
```

Read row 0 as:

```text
logical tokens  0..15 live in physical block 7
logical tokens 16..31 live in physical block 12
```

The `-1` padding in row 1 is never read because `key_lens` tells the kernel how many tokens are actually visible for that request.

### `decode_slot_mapping`

During decode, each request contributes exactly one new token. Instead of calling the slower general chunk writer, we precompute the exact physical cache slot where that one token's K/V vectors should be written.

Formula:

```text
logical_block = cached_seq_len // block_size
block_offset  = cached_seq_len % block_size
physical_slot = block_ids[logical_block] * block_size + block_offset
```

Example with `block_size = 16`:

```text
request 0 block_ids = [7, 12]
request 0 cached_seq_len = 19

logical_block = 19 // 16 = 1
block_offset  = 19 % 16  = 3
physical_slot = block_ids[1] * 16 + 3
              = 12 * 16 + 3
              = 195
```

For a decode batch:

```text
request 0 cached_seq_len = 19, block_ids = [7, 12]      -> slot 195
request 1 cached_seq_len =  4, block_ids = [3]          -> slot 52
request 2 cached_seq_len = 32, block_ids = [9, 10, 11]  -> slot 176

decode_slot_mapping = [195, 52, 176]
```

The backend uses this as:

```python
flat_k.index_copy_(0, decode_slot_mapping, k_tokens[:, 0])
flat_v.index_copy_(0, decode_slot_mapping, v_tokens[:, 0])
```

Edge cases:

- If `cached_seq_len` lands exactly on a block boundary, such as `32` with `block_size = 16`, the write goes to offset `0` of the next logical block.
- The scheduler must allocate enough blocks before decode runs. If the request has no physical block for `logical_block`, that is a scheduler/allocation bug.
- `decode_slot_mapping` is only valid for decode. Prefill may write multiple query tokens per request, so `v4` still uses the general `write_batch()` path for prefill.

## Why No `prefill_slot_mapping` Yet?

A natural question is: if decode can use a slot mapping, why not prefill too?

In decode, each request writes exactly one token:

```text
request row -> one physical slot
```

In prefill, each request may write several tokens, and different requests may have different chunk lengths:

```text
request 0 query_len = 3
request 1 query_len = 1
```

That needs a padded 2D mapping with invalid cells, for example:

```python
prefill_slot_mapping = torch.tensor([
    [126, 127, 192],
    [ 52,  -1,  -1],
])
```

That is useful, but it is a separate optimization. `v4` intentionally avoids it so this step stays focused on the attention backend change. `v5` introduces the prefill slot mapping together with the other local model-body/cache-write optimizations.

## Next Version

`v5` keeps FlashAttention and optimizes the transformer body around it: packed QKV projection, packed gate/up projection, cached RoPE, fused local kernels, and prefill slot-mapped K/V writes.

## Validation

Smoke test:

```bash
python "Module 8/class_8.1/simple_vllm_engine/scripts/smoke_test_versions.py" --versions v4
```

Short real-model sanity run, Llama 3.1 8B, shared six-request workload, `max_new_tokens=8`, prefix cache disabled:

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|
| v4 | 1.365 | 35.17 | n/a | no | +329.2% |

`v4` through `v8` matched each other exactly in this run. vLLM differed on one request.
