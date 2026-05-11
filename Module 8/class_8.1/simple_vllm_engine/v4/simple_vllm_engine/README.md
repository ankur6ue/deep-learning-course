# Simple vLLM-Style Engine v4

This directory contains a teaching implementation of a small vLLM-like
inference engine for a decoder-only Llama-style model.

The goal is not to reproduce vLLM exactly. The goal is to make these ideas
concrete and readable:

- paged KV cache
- continuous batching
- chunked prefill
- prefill/decode disaggregation
- prefix caching on full KV blocks
- the boundary between scheduler logic and CUDA kernel execution
- packed batched execution over multiple requests inside each layer
- loading a real local Hugging Face Llama checkpoint into the teaching engine
- a staging point for the next two runtime changes that move the toy
  implementation closer to vLLM:
  - replacing dense KV gather + assemble with a more direct paged-attention
    interface
  - avoiding explicit materialization of repeated KV heads

## Layout

- `config.py`: model and engine config dataclasses
- `tokenizer.py`: tiny demo tokenizer plus a thin Hugging Face tokenizer wrapper
- `requests.py`: request state machine used by the engine
- `kernels.py`: attention helpers and CUDA-kernel-facing utilities
- `kv_cache.py`: paged KV cache allocator and prefix cache
- `model.py`: a compact Llama-style decoder with explicit KV-cache hooks and one batched attention call per layer
- `scheduler.py`: decode-first continuous batching scheduler
- `engine.py`: serial baseline and simplified vLLM-style engine
- `hf_loader.py`: Hugging Face config/tokenizer/checkpoint loader
- `bench.py`: toy benchmark entrypoint
- `run_real.py`: real-model workload harness comparing serial execution and the `v4` engine
- `compare_vs_transformers.py`: compare `v4` greedy generation against a standard Hugging Face model

## What this engine uses from CUDA

Even though the engine logic is written in Python, the expensive numerical work
still lands in optimized kernels:

- embeddings and linear layers dispatch GEMM kernels through PyTorch
- attention uses `torch.nn.functional.scaled_dot_product_attention`, which on
  CUDA can dispatch FlashAttention-style kernels when supported
- KV page gather/write is expressed with indexing and tensor writes, which is
  where serving-specific logic enters the stack

`v4` starts from the `v3` codebase and introduces an explicit attention-backend
split:

- `dense_reference`: preserves the older `v3` behavior of gathering past KV
  into one padded dense tensor, assembling `[past | current chunk]`, and
  explicitly repeating KV heads to match query heads
- `paged_sdpa`: builds the `[past | current chunk]` tensor directly from the
  paged cache in KV-head space and lets PyTorch SDPA handle grouped-query
  attention internally

The goal is to preserve the same teachable structure while reducing the two
biggest sources of extra memory movement in the current implementation.

In `v4`, the scheduler still builds prefill and decode batches separately, but
the model no longer loops over requests for the attention kernel call. Each
layer gathers paged KV for the whole batch, builds `query_lens`,
`seq_lens_k`, `cu_seqlens_q`, and a batched block table, and then issues one
batched SDPA call. `v4` inherits the real local checkpoint support from `v3` by
loading Hugging Face configs, tokenizers, and sharded safetensors weights.

The key teaching point is that serving systems are not "just kernels". They are
also schedulers, memory managers, and cache managers that decide how kernels are
fed.

## Running

```bash
python "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v4/simple_vllm_engine/bench.py" --device cuda
```

CPU also works for smoke testing:

```bash
python "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v4/simple_vllm_engine/bench.py" --device cpu
```

To compare the two `v4` attention paths directly on the toy workload:

```bash
python "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v4/simple_vllm_engine/bench.py" \
  --device cuda \
  --attention-backend dense_reference
```

```bash
python "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v4/simple_vllm_engine/bench.py" \
  --device cuda \
  --attention-backend paged_sdpa
```

`bench.py` still uses the small synthetic teaching model and workload. It is
useful for understanding the scheduler and batching behavior, but it is not the
entrypoint for validating a real checkpoint load.

To run a representative real-model workload with shared prefixes and staggered
arrivals, comparing serial execution against the `v3` engine, use the existing
vLLM virtualenv so the Hugging Face dependencies are available:

```bash
/home/ankur/dev/.venv-llm/bin/python \
  "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v4/simple_vllm_engine/run_real.py" \
  --model-path /home/ankur/dev/models/Llama-3.1-8B-Instruct \
  --device cuda
```

To compare the `v4` engine against a standard Hugging Face greedy generation
run:

```bash
/home/ankur/dev/.venv-llm/bin/python \
  "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v4/simple_vllm_engine/compare_vs_transformers.py" \
  --model-path /home/ankur/dev/models/Llama-3.1-8B-Instruct \
  --device cuda
```

## Important simplifications

- both `v4` attention backends still end up calling dense SDPA; `v4` has not
  reached a true block-table-driven paged-attention kernel yet
- `paged_sdpa` reduces one dense intermediate and removes explicit KV-head
  repetition, but it still materializes a dense `[B, Tk, Hkv, D]` tensor before
  SDPA
- prefill and decode workers are separate software stages that share one model
  instance and one paged KV store; this teaches the disaggregated control flow
  without requiring a distributed runtime

That said, the implementation is still faithful to the core serving ideas.
