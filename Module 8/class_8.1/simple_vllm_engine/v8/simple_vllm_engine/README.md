# v8: Decode CUDA Graphs

`v8` keeps the optimized eager path from `v7` and adds fixed-shape decode CUDA graph replay.

## What Changed From v7

- `DecodeGraphBucket` owns static decode tensors for one decode batch size.
- Before replay, Python/Triton mutates those tensors in place with current input ids, positions, sequence lengths, slot mappings, and block tables.
- The captured graph replays the model decode path with stable tensor addresses.
- Prefill remains eager because prompt chunk shapes vary.
- `attention_backends.py` still contains only `FlashAttentionPagedBackend`.

## Graph Bucket Mental Model

```text
capture bucket size 8 once
        |
        v
for a real 8-request decode step:
  copy new metadata into static tensors
  replay graph
```

This teaches the same CUDA graph principle vLLM uses: shapes and tensor addresses stay fixed, while metadata values change in-place.

## New Data Structures

### `DecodeGraphBucket`

A CUDA graph can only replay work with the same tensor shapes and tensor
addresses used during capture. `DecodeGraphBucket` owns those fixed tensors for
one decode batch size.

Example:

```text
bucket batch size = 8

static tensors:
input_ids       [8, 1]
positions       [8, 1]
key_lens        [8]
slot_mapping    [8]
block_tables    [8, max_blocks]
```

Before each replay, the engine mutates the values in those tensors:

```text
request ids changed?      update req_slots
sequence lengths changed? update key_lens
cache pages changed?      update block_tables
new token slots changed?  update slot_mapping
```

The graph still sees the same tensor objects, so replay is legal.

### Scratch Rows

If the real batch has fewer requests than the graph bucket size, the unused rows
must still contain safe values because the captured graph will execute all rows.

Example:

```text
bucket size = 8
real decode batch = 6 requests

rows 0..5 = real requests
rows 6..7 = scratch rows
```

Scratch rows use harmless metadata:

```text
input_ids    = pad token
key_lens     = 0
slot_mapping = slot inside a scratch KV block
block_tables = scratch block id
```

Edge cases:

- If a real batch is larger than every captured bucket, this version falls back
  to eager decode for that step.
- If a request needs more block-table columns than the bucket was captured with,
  that bucket is invalid for the step and eager decode is safer.
- Prefill is not captured because prompt chunk sizes vary too much. Capturing
  highly variable prefill shapes would create many graph buckets and make the
  teaching implementation harder to follow.

## Validation

Short real-model sanity run, Llama 3.1 8B, shared six-request workload, `max_new_tokens=8`, prefix cache disabled.

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v8 | 0.431 | 111.40 | yes | no | +1.9% | +35.5% |

vLLM baseline for the same run: `0.318s`, `150.96 tok/s`. `v8` matched `v7` exactly. vLLM differed on one request in this short run.
