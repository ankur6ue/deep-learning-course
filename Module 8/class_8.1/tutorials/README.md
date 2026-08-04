# Attention And Serving Systems Tutorials

The first six tutorials isolate the attention progression used by
`simple_vllm_engine`. They all run on CPU by default and use the same deterministic
attention-only transformer, requests, weights, and block tables. The remaining
tutorials cover CUDA execution, Triton, `torch.compile`, and CUDA graphs.

## Quick Start

Run the attention sequence from the repository root:

```bash
python "Module 8/class_8.1/tutorials/01_per_request_attention.py"
python "Module 8/class_8.1/tutorials/02_dense_causal_attention.py"
python "Module 8/class_8.1/tutorials/03_paged_online_softmax.py"
python "Module 8/class_8.1/tutorials/04_gathered_sdpa_attention.py"
python "Module 8/class_8.1/tutorials/05_paged_varlen_and_slot_mapping.py"
python "Module 8/class_8.1/tutorials/06_sliding_window_paged_attention.py"
```

Each script validates its result against an independent dense recomputation for
every request. Tutorials 1-6 need only PyTorch; they do not require CUDA, Triton,
FlashAttention, or vLLM for their default paths. Tutorials 5 and 6 can
optionally call the vLLM FlashAttention binding on CUDA.

The systems sequence is:

```bash
python "Module 8/class_8.1/tutorials/07_cuda_launch_is_async.py"
python "Module 8/class_8.1/tutorials/08_triton_launch_is_async.py"
python "Module 8/class_8.1/tutorials/09_clone_and_keepalive_for_async_kernels.py"
python "Module 8/class_8.1/tutorials/10_materialization_vs_synchronization.py"
python "Module 8/class_8.1/tutorials/11_torch_compile_kernel_fusion.py"
python "Module 8/class_8.1/tutorials/12_torch_compile_static_vs_dynamic.py"
python "Module 8/class_8.1/tutorials/13_cuda_graph_buckets.py"
```

Those scripts print a short skip message when their required CUDA or Triton
runtime is unavailable.

## Version Map

| Tutorial | Main concept | Toy engine mapping |
| --- | --- | --- |
| `01_per_request_attention.py` | Calculate attention independently at each request's natural length | Conceptual baseline before batching and paging |
| `02_dense_causal_attention.py` | Combine requests in one padded dense attention batch | Conceptual baseline for batched serving |
| `03_paged_online_softmax.py` | Direct page walks and block-wise online softmax | `v2` |
| `04_gathered_sdpa_attention.py` | Gather paged K/V, assemble dense K/V, call PyTorch SDPA | `v3` |
| `05_paged_varlen_and_slot_mapping.py` | Packed queries, paged-kernel metadata, and ragged slot mappings | `v4` attention contract plus the `v5` prefill mapping/write generalization |
| `06_sliding_window_paged_attention.py` | Per-layer full versus local visibility, with an optional paged FlashAttention comparison | `v10` Triton paged attention semantics |
| `07`-`10` | CUDA launch ordering, tensor lifetime, and materialization | Primarily `v5`; side-stream lifetime also applies to `v7` |
| `11`-`12` | Compiled tensor regions and dynamic shapes | `v6` |
| `13_cuda_graph_buckets.py` | Static tensor addresses and fixed-shape graph buckets | `v8`; the static metadata model continues into `v9` |

There is no new core attention backend in `v6`-`v9`. Those versions optimize
the model body, CPU/GPU control plane, graph replay, and metadata movement around
the paged FlashAttention call. That is why tutorials 7-13 change execution
mechanics rather than attention math.

## Shared Attention Example

The shared support file `_attention_tutorial_common.py` contains only fixtures
and validation code. The attention mechanism itself remains visible in each
numbered tutorial.

The model is deliberately small:

```text
layers       = 2 attention-only layers
hidden size  = 16
heads        = 2
head dim     = 8
block size   = 16 tokens
```

Each layer contains separate Q, K, V, and output projections plus a residual
connection. It omits RMSNorm, RoPE, and the MLP so changes in output can be
attributed to attention data movement and masking.

The same three requests appear throughout:

```text
request_a: context=1..18     next chunk=[19, 20]
request_b: context=22..53    next chunk=[54]
request_c: context=40..54    next chunk=[56, 57, 58]
```

Their context, query, and resulting key lengths are:

```text
             past_len   query_len   key_len
request_a       18          2         20
request_b       32          1         33
request_c       15          3         18
```

The paged tutorials use deliberately non-contiguous block tables:

```text
request_a: [8, 1, 10, 3]
request_b: [0, 9, 4, 11]
request_c: [6, 2, 7, 5]
```

For `request_a`, `block_size=16` means:

```text
logical positions  0..15 -> physical block 8 -> flat slots 128..143
logical positions 16..19 -> physical block 1 -> flat slots  16..19
```

Physical blocks need not be adjacent. The block-table position is the logical
block number; its value identifies the physical cache block.

The tutorials use three related terms deliberately:

- **Per request** in tutorial 1 means each request has its own naturally sized
  tensors. There is no batch dimension.
- **Padded dense batch** in tutorials 2 and 4 means requests share rectangular
  tensors sized to the longest request. Invalid rows and columns are masked.
- **Packed queries** in tutorial 5 means valid query rows are concatenated
  without padding and separated by `cu_seqlens_q`.

## Tutorial 1: Per-Request Dense Attention

File: `01_per_request_attention.py`

The first implementation processes each request independently:

```text
for request in requests:
    prefill its context K/V
    project its current Q/K/V
    attend with Q_current over [K_context | K_current]
```

There is no batch dimension and no padding:

```text
request_a: Q=[2, 2, 8]  K/V=[20, 2, 8]
request_b: Q=[1, 2, 8]  K/V=[33, 2, 8]
request_c: Q=[3, 2, 8]  K/V=[18, 2, 8]
```

For `request_a`, `past_len=18`. Its current query rows have absolute positions
18 and 19, so their masks are:

```text
query row 0 -> keys 0..18
query row 1 -> keys 0..19
```

This is the most direct implementation because each softmax sees only real
tokens. Its limitation is execution: three requests require three separate
attention calculations. Tutorial 2 combines those calculations into one batch.

## Tutorial 2: Batched Dense Causal Attention

File: `02_dense_causal_attention.py`

This tutorial has two explicit serving stages:

```text
1. Prefill each request's context into a dense per-layer K/V cache.
2. Project a separate current query chunk and calculate:

   Q_current @ [K_context | K_current]^T / sqrt(head_dim)
                            |
                            v
                   causal/padding mask
                            |
                            v
                          softmax
                            |
                            v
                 probabilities @ [V_context | V_current]
```

The context, current-query, and resulting key lengths are:

```text
past_lens  = [18, 32, 15]
query_lens = [2, 1, 3]
key_lens   = [20, 33, 18]

Q_current = [3, 3, 2, 8]
K_full    = [3, 33, 2, 8]
V_full    = [3, 33, 2, 8]
mask      = [3, 1, 3, 33]
```

For `request_a`, the two current queries have absolute positions 18 and 19:

```text
query row 0 -> keys 0..18
query row 1 -> keys 0..19
```

The context does not produce Q rows again during stage 2. It contributes only
cached K/V. Current query tokens produce Q, K, and V because later rows in the
same current chunk may attend to earlier rows causally.

The cache is still non-paged: each layer keeps one ordinary ragged K tensor and
V tensor per request. The current Q rows and full K/V histories are copied into
rectangular tensors for the batched attention calculation. Tutorial 3 changes
the persistent cache storage to physical pages and block tables.

## Tutorial 3: Direct Paged Online Softmax

File: `03_paged_online_softmax.py`

The modification is storage and traversal:

```text
dense K/V
    becomes
paged K/V + per-request block table
```

For every valid current token, the tutorial computes:

```text
token_position = past_len + query_offset
logical_block  = token_position // block_size
block_offset   = token_position % block_size
physical_block = block_table[logical_block]
physical_slot  = physical_block * block_size + block_offset
```

It writes current K/V first, then each query walks the visible physical blocks.
Instead of storing logits for all keys, it keeps three values per head:

```text
m = running maximum score
l = running sum of exp(score - m)
n = running weighted V numerator
```

For a new score block `s_block`:

```text
m_new = max(m, max(s_block))

l_new = exp(m - m_new) * l
      + sum(exp(s_block - m_new))

n_new = exp(m - m_new) * n
      + sum(exp(s_block - m_new) * V_block)

output = n_new / l_new
```

Rescaling the old state whenever the maximum changes keeps the calculation
numerically stable. This is the same running-statistics idea used by tiled
FlashAttention kernels, although this tutorial performs it slowly in Python.

## Tutorial 4: Gathered K/V And PyTorch SDPA

File: `04_gathered_sdpa_attention.py`

The permanent cache remains paged, but the attention call temporarily returns
to a dense representation:

```text
paged cache + block tables
        |
        v
k_past / v_past = [batch, max_past_len, heads, head_dim]
        |
        v
k_full / v_full = [cached prefix | current chunk | padding]
        |
        v
additive causal/padding mask
        |
        v
torch.nn.functional.scaled_dot_product_attention
```

For the shared example:

```text
past_lens = [18, 32, 15]
query_lens = [2, 1, 3]
key_lens = [20, 33, 18]

k_past = [3, 32, 2, 8]
k_full = [3, 33, 2, 8]
mask   = [3, 1, 3, 33]
```

For `request_a`, query row 0 is absolute position 18 and can see keys 0..18.
Query row 1 is position 19 and can see keys 0..19. Columns beyond `key_len=20`
are padding.

This version is much simpler than direct Python attention and lets PyTorch
select an optimized SDPA backend. Its cost is extra page gathering, dense
temporary allocation, padding, and memory traffic.

## Tutorial 5: Paged Varlen Contract And Slot Mapping

File: `05_paged_varlen_and_slot_mapping.py`

Tutorial 5 removes the rectangular K/V temporary again. Valid Q rows are packed:

```text
query_lens   = [2, 1, 3]
q_flat       = [6, 2, 8]
cu_seqlens_q = [0, 2, 3, 6]
seqused_k    = [20, 33, 18]
```

Read `cu_seqlens_q` as:

```text
request_a queries -> q_flat[0:2]
request_b queries -> q_flat[2:3]
request_c queries -> q_flat[3:6]
```

`seqused_k` is the number of valid cached keys for each request. The causal
kernel aligns each request's query rows to the end of its K/V sequence:

```text
first query position = seqused_k - query_len
```

For `request_a`, `seqused_k=20` and `query_len=2`, so its packed query rows are
absolute positions 18 and 19.

The ragged prefill slot map is:

```text
tensor([[ 18,  19,  -1],
        [ 64,  -1,  -1],
        [111,  32,  33]])
```

`-1` means that the padded query row does not represent a token and must not
write K/V. Every layer reuses this mapping because request positions and block
tables do not change inside the layer loop.

A decode iteration does not need a different attention implementation. It is
another call to the same chunk operation with one query token per request:

```text
decode = chunked prefill with query_lens = [1, 1, 1]
```

Tutorial 5 therefore runs only the initial context and one ragged continuation
chunk. It does not add a third forward call solely to label it as decode.

The tutorial combines two engine steps:

- `v4` introduces paged FlashAttention metadata and a one-dimensional decode
  slot mapping, with PyTorch cache copies.
- `v5` adds the padded prefill slot mapping and attempts the mapped K/V write
  with Triton.

The tutorial uses one readable mapped write for both cases. Its inner attention
loop is a CPU reference for the paged varlen contract. The default remains:

```bash
python "Module 8/class_8.1/tutorials/05_paged_varlen_and_slot_mapping.py"
```

With CUDA and the vLLM FlashAttention bindings installed, two additional modes
exercise the same `flash_attn_varlen_func` call used by engine `v5`:

```bash
# Run only the CUDA kernel, then validate end to end against dense attention.
python "Module 8/class_8.1/tutorials/05_paged_varlen_and_slot_mapping.py" \
  --backend flash-attn

# Run the readable reference and CUDA kernel for every layer and compare them.
python "Module 8/class_8.1/tutorials/05_paged_varlen_and_slot_mapping.py" \
  --backend compare
```

The shared fixture now uses `block_size=16`, `head_dim=8`, and contexts long
enough to cross page boundaries. Kernel modes create the model, Q, and paged K/V
directly on CUDA in `float16`; no cache repacking or head padding is needed.

Tutorial 5 also sizes the block table for each scheduler batch:

```text
context key lengths = [18, 32, 15] -> block_tables shape [3, 2]
chunk key lengths   = [20, 33, 18] -> block_tables shape [3, 3]
```

The row count is the number of active requests. The column count is
`ceil(max_key_len / block_size)`, so unused block-table columns are not copied
or passed to `flash_attn_varlen_func`. The resulting slice is converted once to
a contiguous `int32` tensor and reused by every model layer in that step.

## Tutorial 6: Sliding-Window Paged Attention

File: `06_sliding_window_paged_attention.py`

The two toy layers now have different policies:

```text
layer 0: sliding window of 3 tokens
layer 1: full causal attention
```

This tutorial makes the distinction between validity and visibility explicit:

```text
seqused_k   = how many valid K/V rows the request has
window_size = which of those valid rows this layer may read
```

For `request_b` during decode:

```text
seqused_k = 34
valid key positions = 0..33

layer 0, window_size=3 -> visible keys 31..33
layer 1, full attention -> visible keys 0..33
```

The cache still stores every token for both layers. Sliding attention changes
the kernel's read range; it does not shorten `seqused_k`.

Tutorial 6 retains tutorial 5's paged metadata contract. Before every layer it
writes the current K/V rows through `slot_mapping`, packs valid Q rows, and
passes the resident paged cache plus `cu_seqlens_q`, `seqused_k`, and a sliced
`block_table` to attention. The only layer-specific metadata is the visibility
window:

```text
layer 0: window_size=3    -> FlashAttention window_size=(2, 0)
layer 1: window_size=None -> FlashAttention window_size=(-1, -1)
```

The tuple counts left and right neighboring keys. The current query is already
causally visible, so a three-token window needs two keys to its left and none
to its right.

The default is the readable CPU reference:

```bash
python "Module 8/class_8.1/tutorials/06_sliding_window_paged_attention.py"
```

With CUDA and vLLM's FlashAttention binding installed, use the resident paged
K/V cache directly or compare it with the reference on every layer:

```bash
python "Module 8/class_8.1/tutorials/06_sliding_window_paged_attention.py" \
  --backend flash-attn

python "Module 8/class_8.1/tutorials/06_sliding_window_paged_attention.py" \
  --backend compare
```

This fixed-shape toy model can express its local window through FlashAttention's
`window_size` argument. Production v10 uses vLLM's Triton paged kernel because
Gemma 4 also requires mixed per-layer head dimensions and KV-head counts, which
the FlashAttention path does not cover cleanly.

## Systems Tutorials 7-13

- `07_cuda_launch_is_async.py`: shows that normal PyTorch CUDA work is already
  asynchronous.
- `08_triton_launch_is_async.py`: shows the same launch behavior for a custom
  Triton kernel.
- `09_clone_and_keepalive_for_async_kernels.py`: demonstrates reusable-workspace
  hazards, cloned snapshots, tensor lifetime, and same-stream ordering.
- `10_materialization_vs_synchronization.py`: separates tensor/storage
  materialization from CPU-visible completion.
- `11_torch_compile_kernel_fusion.py`: compares eager PyTorch, static
  `torch.compile`, and a custom Triton kernel for an output projection followed
  by residual-add/RMSNorm.
- `12_torch_compile_static_vs_dynamic.py`: compares
  `torch.compile(dynamic=False)` and `dynamic=True` across token counts.
- `13_cuda_graph_buckets.py`: captures eager and compiled transformer tails,
  validates replay, demonstrates fixed-address/shape constraints, and selects
  between graph buckets.

## Nsight For Tutorial 11

Tutorial 11 wraps only its measured loop in a CUDA profiler range, keeping
setup, random initialization, and compile warmup out of the report.

Print ready-to-run commands:

```bash
python "Module 8/class_8.1/tutorials/11_torch_compile_kernel_fusion.py" \
  --print-nsight-commands
```

The profile variants are:

```text
tail_eager
tail_compile
tail_triton
```

Example:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --output /tmp/tail_triton \
  python "Module 8/class_8.1/tutorials/11_torch_compile_kernel_fusion.py" \
    --profile-variant tail_triton \
    --cuda-profiler-range

nsys stats --report cuda_gpu_kern_sum,nvtx_sum /tmp/tail_triton.nsys-rep
```

The projection remains a cuBLAS/CUTLASS GEMM. The residual-add/RMSNorm tail is
the region that Inductor or the custom Triton implementation can fuse.

## Tutorial 13: CUDA Graph Buckets

Tutorial 13 reuses tutorial 11's transformer-like tail:

```text
output projection -> residual add -> RMSNorm
```

It reports:

```text
eager
compile
graph eager replay
graph eager copy+replay
graph compile replay
graph compile copy+replay
```

The central distinction is:

```text
CUDA graph replay = one CPU-side cudaGraphLaunch
CUDA graph replay != one GPU kernel
```

The GPU still runs the captured kernels. Replay reduces repeated CPU launch
overhead. A graph bucket owns fixed-address tensors with fixed shapes, so a new
shape requires another bucket.

Run it or print profiling commands with:

```bash
python "Module 8/class_8.1/tutorials/13_cuda_graph_buckets.py"

python "Module 8/class_8.1/tutorials/13_cuda_graph_buckets.py" \
  --print-nsight-commands
```

## Mapping Back To v5 Async Execution

In `v5`, the custom RoPE kernels and slot-mapped cache writes are queued on a
CUDA stream. A Python function returning a tensor does not imply that the GPU
has finished reading its inputs or writing its output.

Same-stream ordering protects the current RoPE path:

```text
RoPE reads positions
        before
the next decode workspace update overwrites positions
```

Side-stream work requires explicit synchronization and lifetime management.
That becomes especially relevant for the asynchronous GPU-to-CPU token copies
introduced in `v7`.

The engine exposes debug synchronization flags:

```bash
SIMPLE_VLLM_SYNC_AFTER_ROPE=1
SIMPLE_VLLM_SYNC_AFTER_FORWARD=1
```

They are useful for isolating lifetime and ordering bugs, but they should remain
off during normal benchmarks because forced synchronization changes the
execution model being measured.
