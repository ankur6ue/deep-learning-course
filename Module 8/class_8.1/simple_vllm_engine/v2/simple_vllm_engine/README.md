# v2: Direct Paged Reference Attention

`v2` is the first real-checkpoint version and the slowest attention implementation on purpose. It teaches what paged attention means before introducing optimized kernels.

## What Changed From v1

- `common/hf_loader.py` can load local Llama/Mistral-style checkpoints into this version.
- `run_real.py` can run a shared JSON workload with a Hugging Face tokenizer.
- `model.py` uses separate Q, K, V, gate, and up projections so the transformer body remains easy to inspect.
- `kernels.py` defines `paged_sdpa_attention()`, a direct paged reference implementation.

## Attention Flow

Each layer does this:

```text
1. Project Q/K/V for the current prompt chunk or decode token.
2. Write the new K/V rows into the paged KV cache.
3. Read visible K/V back by walking each request's block table.
4. Compute attention block by block with an online softmax update.
```

Example with `block_size = 16`:

```text
req.block_ids = [7, 12]
logical token 18 -> block_ids[18 // 16] = 12, offset 2
physical slot   -> 12 * 16 + 2
```

`v2` does not gather a dense `[past | current]` K/V tensor. That makes the paged-cache mapping explicit, but it is much slower than later versions.

## New Data Structures

### `block_ids`

Each request owns a list of physical KV-cache pages. The list position is the
logical block number; the value is the physical block number.

Example with `block_size = 16`:

```text
req.block_ids = [7, 12, 31]
```

Read that as:

```text
logical tokens  0..15 live in physical block 7
logical tokens 16..31 live in physical block 12
logical tokens 32..47 live in physical block 31
```

To find the physical cache slot for logical token 18:

```text
logical_block = 18 // 16 = 1
block_offset  = 18 % 16  = 2
physical_slot = block_ids[1] * 16 + 2
              = 12 * 16 + 2
              = 194
```

Edge cases:

- If the sequence length is exactly 16, only logical block 0 is visible.
- If the next token is position 16, the scheduler must allocate logical block 1
  before the K/V write.
- Physical block numbers do not need to be contiguous. `[7, 12, 31]` is valid.

### `past_lens`, `query_lens`, and `key_lens`

The attention code needs three lengths per request:

```text
past_lens  = tokens already in KV cache before this step
query_lens = tokens being processed in this step
key_lens   = past_lens + query_lens
```

Example:

```text
request A: past_len=10, query_len=4 -> key_len=14
request B: past_len= 0, query_len=6 -> key_len=6
```

Request A is continuing from cached tokens. Request B is in its first prefill
chunk and has no cached prefix yet.

Edge case:

```text
decode query_len = 1
```

Decode is just the same structure with exactly one new query token per active
request.

## Next Version

`v3` keeps the same paged KV cache, but gathers cached pages into dense tensors and calls PyTorch `scaled_dot_product_attention` for the attention math.

## Validation

Smoke test: `v2` batched execution matches the serial reference exactly on the synthetic workload.

```bash
python "Module 8/class_8.1/simple_vllm_engine/scripts/smoke_test_versions.py" --versions v2
```
