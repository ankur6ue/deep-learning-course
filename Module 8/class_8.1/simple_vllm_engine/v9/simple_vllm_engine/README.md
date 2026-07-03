# v9: GPU-Resident Decode Metadata

`v9` keeps the `v8` decode and all-valid prefill CUDA graph paths and removes
one remaining Python/H2D hot-path cost: rebuilding and copying the decode block
table every decode step.

## What Changed From v8

- `DecodeGpuState` now owns a persistent GPU table named `request_block_tables`.
- Each row in that table belongs to a stable request slot, not to a temporary batch row.
- The table is updated only when `req.block_ids` grows because the request received a new KV cache block.
- Decode steps now stage only two small per-request arrays from CPU to GPU: request slots and cached sequence lengths.
- A Triton metadata kernel gathers the active block-table rows from `request_block_tables` and fills the batch-shaped FlashAttention metadata tensors.
- The all-valid prefill graph path from `v8` is unchanged; `v9`'s new work is
  specifically decode metadata movement.

## Why This Helps

In `v8`, every decode step did this in Python:

```text
for each request in the decode batch:
  walk req.block_ids
  copy physical block ids into a pinned CPU batch table
copy the whole rectangular batch table to GPU
run a metadata kernel
```

That table is mostly repetitive. A request's block table changes only when it crosses a KV block boundary.

Example with `block_size=16`:

```text
cached_seq_len = 42
req.block_ids  = [7, 12, 31]
```

For decode steps at positions 42, 43, 44, and 45, the request still uses the same logical block table:

```text
logical block 0 -> physical block 7
logical block 1 -> physical block 12
logical block 2 -> physical block 31
```

Only when the request reaches the next block boundary does `ensure_capacity()` append a new physical block id. `v9` updates the GPU row at that point, not on every token.

## New Data Structure

### `DecodeGpuState.request_block_tables`

This is a rectangular GPU tensor:

```text
request_block_tables[request_slot, logical_block] -> physical_block
```

Example:

```text
request_id_to_slot["req-A"] = 3
req-A.block_ids             = [7, 12, 31]
scratch_block_id            = 2047
```

The GPU table row becomes:

```text
request_block_tables[3] = [7, 12, 31, 2047, 2047, ...]
```

A later decode batch does not copy `[7, 12, 31, ...]` from CPU. It copies only:

```text
req_slots[batch_row]       = 3
cached_seq_lens[batch_row] = 42
```

Then the Triton prep kernel derives:

```text
input_ids[batch_row]    = last_token_ids[3]
positions[batch_row]    = 42
key_lens[batch_row]     = 43
slot_mapping[batch_row] = 31 * block_size + (42 % block_size)
block_tables[batch_row] = request_block_tables[3]
```

## Edge Cases

### Request Gets a New KV Block

If a request crosses a block boundary, `kv_cache.ensure_capacity()` appends a physical block id to `req.block_ids`.

Example:

```text
before: req.block_ids = [7, 12, 31]
after:  req.block_ids = [7, 12, 31, 44]
```

`DecodeGpuState.sync_request_blocks(req)` copies only the new suffix:

```text
request_block_tables[slot, 3] = 44
```

It does not rewrite columns 0, 1, and 2.

### CUDA Graph Warmup

Graph warmup performs a dry run to capture decode and all-valid prefill graph
buckets. Because v9 adds GPU request-table state, warmup must snapshot and
restore that table along with KV allocator state and request-slot state.
Otherwise the timed run could inherit stale rows from the warmup requests.

## Validation

Run this version with:

```bash
/home/ankur/dev/.venv-trl/bin/python Module\ 8/class_8.1/simple_vllm_engine/v9/simple_vllm_engine/run_real.py \
  --model-path /dev/models/Llama-3.1-8B-Instruct \
  --workload Module\ 8/class_8.1/simple_vllm_engine/workloads/long_context_serving_workload.json \
  --max-new-tokens 100 \
  --ignore-eos \
  --disable-prefix-cache
```

Long-context real-model run, Llama 3.1 8B, eight requests, ~490-497 prompt tokens each, `max_new_tokens=100`, prefix cache disabled, arrival steps ignored for offline-style batching.

| Version | Wall time (s) | Tok/s | Exact IDs vs v8 | Exact IDs vs vLLM | Improvement vs v8 | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v9 | 1.807 | 442.72 | yes | no | +1.5% | +1.7% |

Same-run baselines:

| Runner | Wall time (s) | Tok/s |
|---|---:|---:|
| v8 | 1.835 | 435.90 |
| vLLM | 1.776 | 450.37 |

The vLLM token IDs differ from `v8`/`v9`, but `v9` is exactly identical to `v8`. That is the correctness signal for this version: only decode metadata movement changed, not model math or attention semantics.

## Ministral 14B Check

Same workload on `Ministral-3-14B-Instruct-2512-TextOnly-BF16-Fresh`, using `num_blocks=512` so the BF16 14B model plus KV cache fits on the RTX 5090.

| Version | Wall time (s) | Tok/s | Exact IDs vs v8 | Exact IDs vs vLLM | Improvement vs v8 | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v9 | 2.906 | 275.34 | yes | no | +2.1% | +0.3% |

Same-run baselines:

| Runner | Wall time (s) | Tok/s |
|---|---:|---:|
| v8 | 2.969 | 269.49 |
| vLLM | 2.896 | 276.28 |

The fresh checkpoint has `model_type=ministral3`, which this installed vLLM/Transformers stack does not recognize. The vLLM timing above uses the `TextOnly-BF16-Fresh-vllm-compat` config shim; its safetensors files are hardlinks to the fresh checkpoint. vLLM also emitted a tokenizer-regex warning and produced visibly degraded text, so use the vLLM row mainly as a timing baseline here. `v9` output looked normal and exactly matched `v8`.
