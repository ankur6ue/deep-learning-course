# Simple vLLM-Style Engine Teaching Sequence

This directory is organized as a clean 10-step teaching progression. The old exploratory versions live in `legacy_experiments/`; the active `v1`-`v10` folders should tell one coherent story.

The toy engine does not use vLLM model code. Later versions intentionally call
vLLM-provided attention/runtime kernels in a few places so the course can focus
on scheduler, metadata, KV-cache, and model-body mechanics. Production vLLM
remains useful as an external benchmark in `vllm_baseline/`.

## Version Map

| Version | Topic | Main Idea |
| --- | --- | --- |
| `v1` | Serving control plane | Requests, scheduler, chunked prefill, decode, and paged KV cache. |
| `v2` | Direct paged reference attention | Walk paged KV blocks directly; slow, explicit, and useful for understanding block tables. |
| `v3` | Gathered SDPA attention | Gather paged K/V into dense `[past | current]` tensors and call PyTorch SDPA. |
| `v4` | Paged FlashAttention backend | FlashAttention reads paged K/V through block tables; decode gets `decode_slot_mapping`, while cache writes still use PyTorch paths. |
| `v5` | Local model-body kernels | Packed QKV, packed gate/up, cached RoPE, fused local kernels, precomputed prefill slot mappings, and Triton K/V cache writes. |
| `v6` | `torch.compile` model body | Compile selected pure tensor regions while keeping attention and KV mutation explicit. |
| `v7` | Async GPU/CPU control plane | Keep next decode tokens on GPU and copy CPU-visible output asynchronously. |
| `v8` | CUDA graphs | Fixed-shape decode buckets, all-valid prefill buckets, static workspaces, scratch rows, and graph replay. |
| `v9` | GPU-resident decode metadata | Keep request block tables on GPU and gather active metadata rows with Triton. |
| `v10` | Architecture adapters for Gemma 4 | Keep the serving engine explicit while adding Gemma 4 text layer shapes, norms, RoPE, and Triton paged attention. |

## Shared Test Assets

- `workloads/`: shared prompt workloads.
- `scripts/`: smoke and real-model benchmark orchestration.
- `vllm_baseline/`: external production-vLLM timing harness.
- `common/`: shared loader, tokenizer, benchmark, and utility code used by multiple versions.
- `legacy_experiments/`: preserved old implementation history.
- `GEMMA_V10_WORKLOG.md`: durable notes for the Gemma 4 validation work.

## Current Short Benchmark Snapshot

Short Llama 3.1 8B sanity run, six requests, `max_new_tokens=8`, prefix cache disabled:

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v4 | 1.365 | 35.17 | n/a | no | n/a | +329.2% |
| v5 | 1.272 | 37.73 | yes | no | +6.8% | +300.1% |
| v6 | 0.439 | 109.39 | yes | no | +65.5% | +38.0% |
| v7 | 0.439 | 109.24 | yes | no | -0.1% | +38.2% |
| v8 | 0.431 | 111.40 | yes | no | +1.9% | +35.5% |

vLLM baseline for the same run: `0.318s`, `150.96 tok/s`.

`v4` through `v8` matched each other exactly. vLLM differed on one request in this short run.

## Documentation Style

Each version README should answer:

1. What changed from the previous version?
2. What problem does the change solve?
3. What new data structures or metadata were introduced?
4. What is the core data flow, with examples and edge cases?
5. How do you run a small correctness/timing check?

Inline comments should explain data movement and invariants. Avoid comments that merely restate code.
