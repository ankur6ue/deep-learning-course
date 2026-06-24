# v1: Serving Control Plane

`v1` is the readable baseline. It teaches the serving loop and KV-cache ownership before introducing real checkpoints or optimized kernels.

## What This Version Contains

- `requests.py`: immutable `RequestSpec` plus mutable `RequestState`.
- `scheduler.py`: decode-first continuous batching and chunked prefill scheduling.
- `kv_cache.py`: paged KV allocation with per-request logical block tables.
- `engine.py`: `PrefillWorker`, `DecodeWorker`, `SimpleVLLMEngine`, and a `SerialEngine` reference.
- `model.py`: a small synthetic Llama-like model used for deterministic toy tests.
- `bench.py`: a synthetic workload comparing serial and batched execution.

The only shared support file used here is `common/tokenizer.py` for the toy tokenizer.

## Core Flow

A request starts with prompt tokens but no KV pages. During prefill, the scheduler assigns a prompt chunk and the KV cache allocates enough physical pages for that request.

Example with `block_size = 16`:

```text
request prompt length = 35
request block_ids = [7, 12, 31]
logical tokens 0..15   live in physical block 7
logical tokens 16..31  live in physical block 12
logical tokens 32..34  live in physical block 31
```

Decode then repeatedly schedules one token per active request. The sampled token becomes `next_input_token_id` for the next decode step.

## Next Version

`v2` keeps the same scheduler, request lifecycle, and paged KV ownership, then swaps the synthetic model for a real checkpoint-backed model. That next step adds Hugging Face tokenizer support, checkpoint loading, and a Transformers comparison script so correctness can be tested against a real Llama/Mistral-style model.

## Benchmark

`v1` uses a deterministic toy model, so its benchmark answers a narrow control-plane question: does continuous batching beat running the same toy requests serially? The real-checkpoint correctness and vLLM timing table starts in `v2`.

| Runner | Workload | Wall time (s) | Tok/s | Comparison |
|---|---|---:|---:|---|
| serial reference | synthetic six-request toy workload | 0.530 | 271.53 | baseline |
| v1 batched engine | synthetic six-request toy workload | 0.130 | 1103.54 | 4.06x faster than serial |

| Version | Exact IDs vs previous | Exact IDs vs vLLM | Wall-time improvement vs previous | Remaining wall-time gap vs vLLM |
|---|---|---|---:|---:|
| v1 | n/a | n/a | n/a | n/a |

## Run
```bash
python "Module 8/class_8.1/simple_vllm_engine/v1/simple_vllm_engine/bench.py"
```
