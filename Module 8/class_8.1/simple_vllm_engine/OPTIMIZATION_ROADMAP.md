# Simple vLLM Engine Optimization Roadmap

This note replaces the accidental v1/v2/... progression with a category-based
teaching plan. The old versions are useful history, but the implementation has
branched enough that the next pass should rebuild from a clean baseline and add
one optimization category at a time.

Current teaching sequence: `v1` through `v8`.
Historical implementation snapshots: `legacy_experiments/`.

The most optimized historical source used to seed the later teaching versions is
`legacy_experiments/v16_no_vllm_ops` plus the follow-up
`legacy_experiments/v17_prefill_slot_mapping_gpu` experiment.

## Category Count

I count **8 meaningful optimization categories**, plus a baseline step.

For a 6-7 lesson teaching sequence, some categories can be merged:

- Merge categories 2 and 3 into one "paged attention + metadata" lesson.
- Merge categories 4 and 5 into one "model body fusion/compilation" lesson.
- Keep categories 6, 7, and 8 separate because each teaches a distinct vLLM
  idea and has a clear before/after mental model.

## Baseline Step: Serving Control Plane

This is not an optimization category; it is the starting point.

Core ideas:

- Request lifecycle: prefill, decode, finished.
- Decode-first continuous batching.
- Chunked prefill so long prompts do not monopolize the engine.
- Paged KV cache with per-request block tables.
- Prefix cache for complete prompt blocks.
- Serial baseline for comparison.

Historical source: `v1`.

Teaching goal: make the serving architecture readable before worrying about
kernel efficiency.

## 1. Batched Model Execution and Real-Model Harness

Problem in baseline:

- The scheduler batches requests, but the model still loops over requests inside
  attention.
- Synthetic weights/tokenizer hide real-model correctness issues.

Optimizations:

- Move from per-request attention calls to one batched attention call per layer.
- Build batch metadata: query lengths, past lengths, key lengths, cumulative
  query offsets, and block tables.
- Add Hugging Face checkpoint loading, tokenizer wrapping, real workload scripts,
  and `transformers` comparison.

Historical sources: `legacy_experiments/v2`, `legacy_experiments/v3`.

Why it matters:

- Separates scheduler decisions from model-runner metadata construction.
- Gives us a real benchmark target instead of a toy-only speedup.

## 2. True Paged-Attention Boundary

Problem after batched SDPA:

- The KV cache is paged, but attention still materializes dense `[past | current]`
  K/V tensors and repeats KV heads before calling SDPA.

Optimizations:

- Add an attention backend interface.
- Use a readable dense/gathered path as an intermediate correctness reference.
- Add paged attention backends that consume block tables instead of dense
  gathered K/V.
- The active teaching sequence now standardizes on the FlashAttention paged
  backend. Older exploratory versions also tried FlashInfer and custom Triton
  attention paths; those remain useful provenance, not the main teaching path.

Historical sources: `legacy_experiments/v4`, `legacy_experiments/v7`,
`legacy_experiments/v13`, `legacy_experiments/v16_no_vllm_ops`.

Why it matters:

- This is the key conceptual vLLM transition: attention kernels should read KV
  through block tables, not through Python-assembled dense buffers.

## 3. Metadata, Slot Mapping, and KV-Write Hot Paths

Problem after adding paged attention:

- Even if attention is efficient, Python can spend too much time rebuilding
  metadata and writing K/V request by request.

Optimizations already implemented:

- Build layer-invariant attention/decode metadata once per decode batch, not
  once per layer.
- Compute `decode_slot_mapping` once per decode batch.
- Use slot-mapped vectorized KV writes instead of per-request decode KV writes.
- Reuse fixed decode `input_ids`, `positions`, block tables, and metadata
  workspaces in eager decode.
- Add a Triton decode metadata-prep kernel that fills input ids, positions,
  sequence lengths, slot mapping, and int32 block tables on device.

Historical sources: `legacy_experiments/v5`, `legacy_experiments/v6`,
`legacy_experiments/v12`, `legacy_experiments/v13`,
`legacy_experiments/v16_no_vllm_ops`.

Why it matters:

- vLLM's model runner keeps persistent metadata buffers and updates them in
  place. It does not rebuild every decode input from Python objects.

Pending optimization to revisit:

- Move **prefill slot mapping** from the nested Python loop in
  `model._build_attention_metadata()` to a GPU/Triton kernel.
- Current code loops over requests and query offsets:

  ```python
  for req, past_len, query_len in zip(requests, past_lengths, lengths, strict=True):
      row = []
      for offset in range(query_len):
          token_pos = past_len + offset
          row.append(kv_cache.physical_slot(req.block_ids, token_pos))
      row.extend([-1] * (max_query_len - query_len))
      slot_rows.append(row)
  ```

- Desired GPU calculation:

  ```text
  token_pos    = past_len[row] + offset
  block_index  = token_pos // block_size
  block_offset = token_pos % block_size
  block_id     = block_tables[row, block_index]
  slot         = block_id * block_size + block_offset
  slot         = -1 if offset >= query_len[row]
  ```

- vLLM precedent in the installed source:
  `/home/ankur/dev/.venv-trl/lib/python3.12/site-packages/vllm/v1/worker/gpu/block_table.py`
  has `BlockTables.compute_slot_mappings()` and a Triton
  `_compute_slot_mappings_kernel` that derives slot ids from positions and block
  tables. The relevant math is around lines 252-274 in that file.

This is a general optimization, not benchmark cheating. It keeps the same block
-table abstraction while moving mechanical per-token mapping out of Python.

## 4. Model Weight Layout and Fused Primitives

Problem after attention/metadata improvements:

- The transformer layer still launches too many small operations compared with
  production vLLM.

Optimizations:

- Pack Q/K/V projections into one `qkv_proj`.
- Pack MLP gate/up projections into one `gate_up_proj`.
- Use fused residual-add RMSNorm where possible.
- Use fused/native RMSNorm, SwiGLU, and KV cache write kernels in the no-vLLM
  runtime path.
- Remove runtime dependency on vLLM custom ops by replacing them with local
  Triton/PyTorch-compatible implementations where practical.

Historical sources: `legacy_experiments/v8`, `legacy_experiments/v9`,
`legacy_experiments/v16_no_vllm_ops`.

Why it matters:

- Reduces GEMM launches and small pointwise kernels.
- Makes the teaching implementation closer to vLLM's Llama layer structure.

## 5. `torch.compile` for Pure Tensor Model-Body Regions

Problem:

- Full model forward is not a clean compile target because it crosses attention
  kernels, KV cache mutation, Python metadata objects, and backend wrappers.

Optimizations:

- Compile only pure tensor pieces:
  - `mlp`: gate/up projection, SwiGLU, down projection.
  - `input_qkv`: input RMSNorm/residual plus QKV projection, with RoPE outside.
  - `tail`: O projection plus post-attention RMSNorm/residual plus MLP.
  - `all`: `input_qkv` plus `tail`.
- Keep `mlp` as the safest default because broader compile scopes were model- and
  workload-dependent.

Historical sources: `legacy_experiments/v10`, `legacy_experiments/v13`,
`legacy_experiments/v16_no_vllm_ops`.

Measured finding:

- On the realistic long-prompt Llama workload, strict eager was about `1.822s`,
  `mlp` compile was about `1.239s`, and vLLM eager was about `1.192s`.
- Expanding to `tail` helped Llama slightly but did not consistently help
  Ministral, so the default should remain conservative.

Why it matters:

- This captures the value of graph compilation without trying to compile the
  dynamic attention/cache boundary.

## 6. RoPE Cache and Fused RoPE Kernel

Problem:

- The toy PyTorch RoPE path repeatedly built frequency/cos/sin tensors and
  launched many tiny elementwise/cat/copy kernels.

Optimizations:

- Build a reusable RoPE cache compatible with Llama/Mistral scaling rules.
- Apply RoPE with a fused out-of-place Triton kernel that emits contiguous Q/K
  tensors.
- Remove the dependency on vLLM's RoPE module by reproducing the needed cache
  layout locally.

Historical sources: `legacy_experiments/v14_rope`,
`legacy_experiments/v15_rope_native`, `legacy_experiments/v16_no_vllm_ops`.

Measured finding:

- Nsight showed toy v13 launched about `142,553` CUDA kernels, while v14/v15
  dropped to about `19,673`, in the same broad range as vLLM eager.
- The largest confirmed difference was RoPE-related: v13 had over `115k` generic
  elementwise kernels plus thousands of `sin`, `cos`, and copy kernels. The
  optimized RoPE path replaced that with one fused RoPE kernel per layer/step.

Why it matters:

- This was the optimization that moved the toy engine from a large eager gap to
  within run noise of vLLM eager on the measured short workload.

## 7. Asynchronous GPU-to-CPU Control-Plane Communication

Problem:

- Reading sampled token ids with `.item()` or immediate CPU materialization puts
  GPU-to-CPU synchronization on the per-token critical path.

Optimizations:

- Add a GPU decode-state table storing each active request's latest sampled
  token.
- Feed the next decode input from GPU state instead of CPU-visible token ids.
- Copy sampled tokens to CPU asynchronously for output/EOS bookkeeping.
- Resolve only older async copies, allowing CPU stop checks to lag one step
  behind the GPU decode path.

Historical sources: `legacy_experiments/v11`,
`legacy_experiments/v16_no_vllm_ops`.

Why it matters:

- Mirrors vLLM's separation between GPU-resident next-token state and CPU-visible
  request/output bookkeeping.
- The standalone timing gain may be small on workloads without EOS, but it
  prevents EOS-capable runs from reintroducing avoidable synchronization.

## 8. Decode CUDA Graph Replay

Problem:

- Even after eager optimization, many small decode launches still incur CPU
  launch overhead.

Optimizations:

- Add fixed-shape decode graph buckets.
- Use static input/output/metadata tensors per bucket.
- Pad inactive rows with a scratch KV block.
- Validate captured/replayed outputs against eager execution.
- Keep attention metadata in stable tensors so graph replay can mutate values
  without changing tensor addresses.

Historical sources: `legacy_experiments/v7`, `legacy_experiments/v10`,
`legacy_experiments/v12`, `legacy_experiments/v16_no_vllm_ops`.

Historical measured finding:

- An earlier FlashInfer-based v16 experiment measured about `0.742s`,
  `387.95 tok/s` eager on the shared short workload.
- The same historical path with decode CUDA graph replay reached about `0.692s`,
  `415.98 tok/s`.

Caveat:

- Prefill CUDA graph capture was tried and did not become a useful default. The
  variable prompt/chunk shapes and attention planning made it more complex and
  slower/harder to teach. Keep graph replay focused on decode unless a later
  design has graph-stable prefill inputs.

## Things We Learned Are Not Primary Optimizations

These were useful investigations, but should not be taught as major speedup
steps:

- Dtype mismatch: vLLM and the toy engine are both effectively bf16 for the
  measured real-model runs.
- `max_num_batched_tokens` / `max_num_seqs`: important for fair comparison, but
  not the reason for the measured gap on the small shared workload.
- Prefix caching: a real production vLLM feature advantage, but disable it on
  both sides when the goal is model-runner/kernel comparison.
- Full-forward `torch.compile`: unstable/wrong around attention wrappers and KV
  mutation; compile pure tensor subgraphs instead.
- Prefill CUDA graphs: investigated, but not a useful teaching/performance step
  in the current design.

## Current Rebuild Sequence

The top-level folders now use the cleaned teaching sequence:

```text
v1  serving control plane
v2  direct paged reference attention
v3  gathered SDPA attention
v4  paged FlashAttention backend
v5  packed projections + cached RoPE + local kernels
v6  torch.compile for pure tensor model-body regions
v7  async GPU/CPU control-plane communication
v8  decode CUDA graph replay
```

Older exploratory folders were moved to `legacy_experiments/`. Keep them as
provenance and benchmark history, but do not ask students to read them first.
