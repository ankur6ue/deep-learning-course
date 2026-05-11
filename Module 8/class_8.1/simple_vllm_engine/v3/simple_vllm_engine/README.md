# Simple vLLM-Style Engine v3

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
- `run_real.py`: real-model workload harness comparing serial execution and the `v3` engine
- `compare_vs_transformers.py`: compare `v3` greedy generation against a standard Hugging Face model

## What this engine uses from CUDA

Even though the engine logic is written in Python, the expensive numerical work
still lands in optimized kernels:

- embeddings and linear layers dispatch GEMM kernels through PyTorch
- attention uses `torch.nn.functional.scaled_dot_product_attention`, which on
  CUDA can dispatch FlashAttention-style kernels when supported
- KV page gather/write is expressed with indexing and tensor writes, which is
  where serving-specific logic enters the stack

In `v3`, the scheduler still builds prefill and decode batches separately, but
the model no longer loops over requests for the attention kernel call. Each
layer gathers paged KV for the whole batch, builds `query_lens`,
`seq_lens_k`, `cu_seqlens_q`, and a batched block table, and then issues one
batched SDPA call. `v3` extends that path to a real local checkpoint by
loading Hugging Face configs, tokenizers, and sharded safetensors weights.

The key teaching point is that serving systems are not "just kernels". They are
also schedulers, memory managers, and cache managers that decide how kernels are
fed.

## Running

```bash
python "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v3/simple_vllm_engine/bench.py" --device cuda
```

CPU also works for smoke testing:

```bash
python "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v3/simple_vllm_engine/bench.py" --device cpu
```

`bench.py` still uses the small synthetic teaching model and workload. It is
useful for understanding the scheduler and batching behavior, but it is not the
entrypoint for validating a real checkpoint load.

To run a representative real-model workload with shared prefixes and staggered
arrivals, comparing serial execution against the `v3` engine, use the existing
vLLM virtualenv so the Hugging Face dependencies are available:

```bash
/home/ankur/dev/.venv-llm/bin/python \
  "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v3/simple_vllm_engine/run_real.py" \
  --model-path /home/ankur/dev/models/Llama-3.1-8B-Instruct \
  --device cuda \
  --timing
```

To compare the `v3` engine against a standard Hugging Face greedy generation
run:

```bash
/home/ankur/dev/.venv-llm/bin/python \
  "/home/ankur/dev/deep-learning-course/Module 8/class_8.1/v3/simple_vllm_engine/compare_vs_transformers.py" \
  --model-path /home/ankur/dev/models/Llama-3.1-8B-Instruct \
  --device cuda
```

## Important simplifications

- attention over paged KV is implemented by gathering logical pages into a
  temporary contiguous view before calling SDPA
- even in `v3`, attention over paged KV is still gathered into a dense view
  before calling SDPA rather than dispatched to a custom paged-attention kernel
- prefill and decode workers are separate software stages that share one model
  instance and one paged KV store; this teaches the disaggregated control flow
  without requiring a distributed runtime

That said, the implementation is still faithful to the core serving ideas.
