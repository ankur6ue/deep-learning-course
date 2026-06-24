# v7: Async GPU/CPU Control Plane

`v7` keeps FlashAttention, the v5 local kernels, and the v6 compile boundary, then reduces decode-side CPU synchronization.

## What Changed From v6

- `DecodeGpuState` stores the latest sampled token for each active request on GPU.
- Stable integer request slots map Python request IDs to GPU rows.
- `prepare_decode_inputs()` fills decode `input_ids`, positions, lengths, slot mappings, and int32 block tables on GPU.
- `update_decode_state()` writes sampled tokens back into the GPU table.
- `AsyncTokenCopy` copies sampled tokens to CPU without blocking the next decode step.
- `attention_backends.py` still contains only `FlashAttentionPagedBackend`.

## Decode Flow

```text
sample token on GPU
        |
        v
update last_token_ids[request_slot] on GPU
        |
        v
start async CPU copy for output/EOS bookkeeping
        |
        v
next decode step consumes last_token_ids directly on GPU
```

## New Data Structures

### `DecodeGpuState`

`DecodeGpuState` is a small GPU table for active decode requests. It avoids a
synchronous GPU-to-CPU transfer just to feed the next token back into the model.

It has two key pieces:

```text
request_id_to_slot: Python dict, request id -> stable GPU row
last_token_ids:     GPU tensor, latest sampled token per slot
```

Example:

```text
request_id_to_slot = {
    "req-a": 2,
    "req-b": 0,
}

last_token_ids =
slot 0 -> token for req-b
slot 1 -> unused
slot 2 -> token for req-a
```

When the scheduler forms a decode batch `[req-a, req-b]`, it copies the request
slots `[2, 0]` to GPU. A small kernel then reads:

```text
input_ids[0] = last_token_ids[2]
input_ids[1] = last_token_ids[0]
```

Edge cases:

- Request slots are stable while a request is active, but they are recycled when
  the request finishes.
- The dictionary is still on CPU because scheduling decisions are CPU-side in
  this teaching engine.
- The token values stay on GPU so the next decode step does not need `.item()`.

### `AsyncTokenCopy`

The CPU still eventually needs generated token ids for output text and EOS
handling. `AsyncTokenCopy` starts a GPU-to-CPU copy on a side stream and returns
immediately.

Example:

```text
step N samples tokens on GPU
step N starts async copy to CPU
step N+1 can already launch using GPU-resident last_token_ids
later, CPU calls tolist() and waits only if the copy is not finished
```

Edge case:

If the CPU needs the token immediately, `tolist()` synchronizes and there is no
latency hiding. The win comes from delaying that wait until after more GPU work
has been launched.

## Next Version

`v8` adds decode CUDA graph replay around the optimized FlashAttention decode path.

## Validation

Short real-model sanity run, Llama 3.1 8B, shared six-request workload, `max_new_tokens=8`, prefix cache disabled.

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v7 | 0.439 | 109.24 | yes | no | -0.1% | +38.2% |

`v7` matched `v6` exactly. On this short workload the async control-plane change is correctness-neutral but not measurably faster; its value tends to show up more when decode has enough steps to hide CPU readback latency.
