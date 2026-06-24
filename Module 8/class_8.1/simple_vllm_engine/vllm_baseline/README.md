# vLLM Baseline

This folder runs the shared `simple_vllm_engine` real-model workload on production vLLM.

The goal is not to reproduce the teaching scheduler exactly. It gives a practical gold standard for the model execution stack: production paged KV cache, FlashAttention/Triton attention backends, torch.compile integration, CUDA graph capture, and vLLM's metadata/cache management.

## Run

```bash
/home/ankur/dev/.venv-llm/bin/python \
  "Module 8/class_8.1/simple_vllm_engine/vllm_baseline/run_vllm_real.py" \
  --model-path /home/ankur/dev/models/Llama-3.1-8B-Instruct \
  --max-model-len 4096 \
  --max-new-tokens 32 \
  --gpu-memory-utilization 0.85 \
  --num-runs 2
```

`max_model_len` is intentionally capped because the local Llama 3.1 config advertises a 131k context window, which would require much more KV cache memory than this workload needs.

For toy-engine parity checks, the runner can also narrow vLLM's scheduler/cache settings:

```bash
/home/ankur/dev/.venv-llm/bin/python \
  "Module 8/class_8.1/simple_vllm_engine/vllm_baseline/run_vllm_real.py" \
  --model-path /home/ankur/dev/models/Llama-3.1-8B-Instruct \
  --max-new-tokens 48 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --disable-prefix-caching \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --block-size 16 \
  --num-runs 1
```

The script prints vLLM's effective dtype, KV-cache dtype setting, block size, prefix-cache state, chunked-prefill state, `max_num_batched_tokens`, and `max_num_seqs` after the engine is constructed.

## Interpreting Results

- Use `run=1` for a fresh workload comparison within one vLLM engine instance.
- Use later runs to understand warm steady-state behavior.
- Do not compare vLLM with `simple_vllm_engine --timing`; the teaching profiler synchronizes around many small sections and inflates runtime.
- vLLM offline generation ignores the teaching workload's `arrival_step`, so this is a model-execution gold standard, not an exact scheduler simulation.
- `kv_cache_dtype=auto` means vLLM stores KV cache in the model dtype unless an fp8/quantized cache dtype is explicitly requested.
