# v3: Gathered SDPA Attention

`v3` shows the common intermediate step: keep the paged KV cache for storage, but temporarily leave the paged layout before attention.

## What Changed From v2

- `attention_backends.py` introduces one backend: `GatheredSDPAAttention`.
- The backend gathers cached K/V pages into dense padded tensors.
- It appends the current K/V chunk into `[cached_prefix | current_chunk]`.
- It calls PyTorch `torch.nn.functional.scaled_dot_product_attention` through `batched_sdpa_attention()`.
- The model loop writes current K/V to the paged cache after attention because this backend returns `wrote_kv=False`.

## Attention Flow

```text
paged cache + block ids
        |
        v
gather cached K/V into dense [B, max_past, Hkv, D]
        |
        v
assemble [cached prefix | current chunk]
        |
        v
build causal/padding mask and call PyTorch SDPA
```

Example:

```text
request A: past_len=5, query_len=2 -> key_len=7
request B: past_len=9, query_len=1 -> key_len=10
```

The dense K/V batch has width 10. A uses columns 0..6 and pads 7..9; B uses 0..9.

## New Data Structures

### `k_past` and `v_past`

`v3` still stores KV in paged form, but before attention it gathers each
request's cached prefix into dense padded tensors:

```text
k_past shape = [B, max_past_len, Hkv, D]
v_past shape = [B, max_past_len, Hkv, D]
```

Example:

```text
request A past_len = 5
request B past_len = 9

max_past_len = 9
```

Rows look like:

```text
k_past[A, 0:5] = real cached keys
k_past[A, 5:9] = padding
k_past[B, 0:9] = real cached keys
```

Edge case:

```text
past_len = 0
```

There is no cached prefix for that request, so the gathered row is all padding
and only the current chunk contributes keys.

### `k_full` and `v_full`

After gathering the cached prefix, the backend appends the current K/V chunk:

```text
k_full = [cached prefix | current chunk | padding]
```

Example:

```text
request A: past_len=5, query_len=2 -> key_len=7
request B: past_len=9, query_len=1 -> key_len=10

k_full shape = [2, 10, Hkv, D]
```

Rows look like:

```text
k_full[A, 0:5]  = cached prefix
k_full[A, 5:7]  = current chunk
k_full[A, 7:10] = padding

k_full[B, 0:9]  = cached prefix
k_full[B, 9:10] = current chunk
```

This is easy to understand, but expensive: every attention call temporarily
materializes dense K/V even though the permanent cache is paged.

### Causal and Padding Mask

The SDPA mask combines two ideas:

```text
padding mask: ignore columns beyond key_len
causal mask:  query token i cannot see future current-chunk tokens
```

Example:

```text
past_len = 5
query_len = 3
key_len = 8
```

The three query rows see:

```text
query 0 sees keys 0..5
query 1 sees keys 0..6
query 2 sees keys 0..7
```

Edge case:

If another request in the batch has a larger `key_len`, this request's row has
extra padded columns. Those columns receive the same large negative mask value
as causally invisible future keys, because both mean "attention must not read
this key."

## Next Version

`v4` stops materializing dense K/V for attention and switches to a FlashAttention paged backend that consumes the block table directly.

## Validation

Smoke test: `v3` satisfies generation invariants on the synthetic workload with the single `gathered_sdpa` backend.

```bash
python "Module 8/class_8.1/simple_vllm_engine/scripts/smoke_test_versions.py" --versions v3
```
