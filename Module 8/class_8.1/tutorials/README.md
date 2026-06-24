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
```

If CUDA or Triton is unavailable, the scripts print a short skip message.

## Files

- `01_cuda_launch_is_async.py`: shows that normal PyTorch CUDA work is already asynchronous.
- `02_triton_launch_is_async.py`: shows the same behavior for a small custom Triton kernel.
- `03_clone_and_keepalive_for_async_kernels.py`: shows why a custom kernel should not read from a mutable workspace that Python may reuse, and why keeping tensor references around is a simple lifetime guard.

## How This Maps To `v5`

In `v5`, `_apply_native_rope_to_flat_qk()` launches custom Triton RoPE kernels.
Those kernels read `q_2d`, `k_2d`, `flat_positions`, and the RoPE cos/sin cache.
The code does two safety-minded things:

```python
flat_positions = positions.reshape(-1).contiguous().clone()
```

This gives the kernel a private copy of positions instead of a view into a
workspace that the next decode step may reuse.

```python
self._rope_keepalive.extend((q_2d, k_2d, flat_positions, cos_sin_cache, q_triton, k_triton))
```

This keeps recently used tensor objects alive while queued kernels may still be
using their storage.
