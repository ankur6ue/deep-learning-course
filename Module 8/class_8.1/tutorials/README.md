# CUDA/Triton Async Execution Tutorials

These tutorials support the `simple_vllm_engine/v5` discussion about custom
Triton kernels, reusable buffers, cloned inputs, and keepalive references.

The key idea:

```text
A Python CUDA/Triton call usually queues GPU work and returns immediately.
The CPU can move on before the GPU has finished reading inputs or writing outputs.
```

Run them from the repository root with the same environment used for the course:

```bash
python "Module 8/class_8.1/tutorials/01_cuda_launch_is_async.py"
python "Module 8/class_8.1/tutorials/02_triton_launch_is_async.py"
python "Module 8/class_8.1/tutorials/03_clone_and_keepalive_for_async_kernels.py"
python "Module 8/class_8.1/tutorials/04_materialization_vs_synchronization.py"
python "Module 8/class_8.1/tutorials/05_torch_compile_kernel_fusion.py"
python "Module 8/class_8.1/tutorials/06_torch_compile_static_vs_dynamic.py"
python "Module 8/class_8.1/tutorials/07_cuda_graph_buckets.py"
```

If CUDA or Triton is unavailable, the scripts print a short skip message.

## Files

- `01_cuda_launch_is_async.py`: shows that normal PyTorch CUDA work is already asynchronous.
- `02_triton_launch_is_async.py`: shows the same behavior for a small custom Triton kernel.
- `03_clone_and_keepalive_for_async_kernels.py`: shows why a custom kernel should not read from a mutable workspace that Python may reuse, and why keeping tensor references around is a simple lifetime guard.
- `04_materialization_vs_synchronization.py`: explains which tensor operations are views, which allocate/copy/compute new storage, and which CPU reads force synchronization.
- `05_torch_compile_kernel_fusion.py`: compares eager PyTorch, static `torch.compile(dynamic=False)`, and a custom Triton kernel on a transformer-like output-projection/residual-add/RMSNorm tail; it also prints Nsight Systems / Nsight Compute commands for launch-count and memory-traffic inspection.
- `06_torch_compile_static_vs_dynamic.py`: compares `torch.compile(dynamic=False)` with `torch.compile(dynamic=True)` on a GELU/residual/RMSNorm tail, randomizing steady-state measurement order and separately showing first-call cost at alternate token counts.
- `07_cuda_graph_buckets.py`: captures the tutorial 5 transformer tail into CUDA graphs for both eager and compiled callables, validates graph replay against eager gold output, shows fixed-shape graph buckets, demonstrates shape-mismatch errors, and selects among multiple buckets.

## Nsight For Tutorial 5

Tutorial 5 has a profiling mode that wraps only the measured loop in a CUDA
profiler range. This keeps setup, random tensor initialization, and compile
warmup out of the Nsight report.

Print ready-to-run commands:

```bash
python "Module 8/class_8.1/tutorials/05_torch_compile_kernel_fusion.py" \
  --print-nsight-commands
```

The profile variants are:

```text
tail_eager
tail_compile
tail_triton
```

Example Nsight Systems run for the custom-tail variant:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --output /tmp/tail_triton \
  python "Module 8/class_8.1/tutorials/05_torch_compile_kernel_fusion.py" \
    --profile-variant tail_triton \
    --cuda-profiler-range

nsys stats --report cuda_gpu_kern_sum,nvtx_sum /tmp/tail_triton.nsys-rep
```

For three measured custom-tail iterations, the clean Nsight smoke test showed
three GEMM kernels and three `_add_rms_norm_kernel` launches: the projection
stays a cuBLAS/CUTLASS GEMM, while the residual-add/RMSNorm tail is one custom
kernel per iteration.

Tutorial 6 owns the static-vs-dynamic compile discussion. It warms both modes at
the base token count, randomizes steady-state measurement order, and then calls
both compiled functions at alternate token counts. `dynamic=False` may need a new
shape-specialized graph for a new token count; `dynamic=True` can often reuse an
existing graph because the token dimension was kept symbolic.

## Tutorial 7: CUDA Graph Buckets

Tutorial 7 uses the same transformer-like tail as tutorial 5:

```text
output projection -> residual add -> RMSNorm
```

It then captures that work into CUDA graph buckets:

```bash
python "Module 8/class_8.1/tutorials/07_cuda_graph_buckets.py"
```

The tutorial validates each graph replay against eager output and prints
steady-state timings for:

```text
eager
compile
graph eager replay
graph eager copy+replay
graph compile replay
graph compile copy+replay
```

The important distinction is:

```text
CUDA graph replay = one CPU-side cudaGraphLaunch
CUDA graph replay != one GPU kernel
```

The GPU still executes the captured kernels. The CPU saves repeated launch
overhead by replaying the captured graph as a unit.

The shape demo shows two different behaviors:

- Calling a `torch.compile(dynamic=False)` function at a new shape may compile
  and cache another shape-specialized graph.
- Replaying an already captured CUDA graph bucket at a new shape is an error.
  The bucket owns fixed-address tensors with fixed shapes.

The multiple-bucket demo captures two buckets and chooses the matching bucket by
the actual tensor shape:

```text
(tokens, hidden_size, attn_size) -> CUDA graph bucket
```

Print Nsight commands:

```bash
python "Module 8/class_8.1/tutorials/07_cuda_graph_buckets.py" \
  --print-nsight-commands
```

Use Nsight Systems to compare ordinary eager/compiled execution with graph
replay. The graph variants should show `cudaGraphLaunch` on the CPU side, while
the GPU kernel summary still lists the captured kernels.

## How This Maps To `v5`

In `v5`, `_apply_native_rope_to_flat_qk()` launches custom Triton RoPE kernels.
Those kernels read `q_2d`, `k_2d`, `flat_positions`, and the RoPE cos/sin cache.
RoPE runs on the current CUDA stream, and decode workspace updates are also
queued on that stream, so the current code relies on same-stream ordering:

```python
flat_positions = positions.reshape(-1).contiguous()
```

The next decode step may reuse the positions workspace, but its update is queued
after the previous RoPE kernels on the same stream. That means the update cannot
race ahead and overwrite positions before RoPE reads them.

The RoPE cos/sin cache is a model-owned buffer, so it already lives for the
model lifetime and does not need special lifetime handling.

The clone-and-keepalive pattern is still important for side-stream work, such as
async GPU-to-CPU token copies in later engine versions. In those cases the copy
stream can still be reading old values while the main stream has already moved
on, so the code needs explicit stream ordering and live tensor references.

`v5` also has two debug synchronization flags, both off by default:

```bash
SIMPLE_VLLM_SYNC_AFTER_ROPE=1
SIMPLE_VLLM_SYNC_AFTER_FORWARD=1
```

They are useful for narrowing down async lifetime bugs, but they should stay off
for normal benchmarking because forced synchronization changes the timing model.
