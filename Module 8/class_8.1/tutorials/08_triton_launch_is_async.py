#!/usr/bin/env python3
"""Tutorial 8: a custom Triton kernel also launches asynchronously.

The kernel below performs a simple vector add. The operation is intentionally
simple because the point is not the math; the point is launch semantics:

    Python queues the Triton kernel and returns before the GPU is necessarily done.

This is the same behavior that matters in `v5` when the RoPE Triton kernel reads
Q/K and position tensors.
"""

from __future__ import annotations

import argparse
import time

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on local CUDA/Triton install.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elements",
        type=int,
        default=16_000_000,
        help="Number of float32 elements. Increase for a more visible async gap.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if triton is None:
        print("Triton is not available; install/import Triton to run this tutorial.")
        return
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    device = torch.device("cuda")
    x = torch.randn(args.elements, device=device)
    y = torch.randn(args.elements, device=device)
    out = torch.empty_like(x)
    grid = (triton.cdiv(args.elements, args.block_size),)

    for _ in range(3):
        _vector_add_kernel[grid](x, y, out, args.elements, BLOCK=args.block_size)
    torch.cuda.synchronize()

    print("Tutorial 8: Triton launch vs actual completion")
    print(f"elements: {args.elements:,}")
    print()

    for idx in range(args.repeats):
        torch.cuda.synchronize()
        done = torch.cuda.Event()
        t0 = time.perf_counter()
        _vector_add_kernel[grid](x, y, out, args.elements, BLOCK=args.block_size)
        done.record()
        t1 = time.perf_counter()
        immediately_done = done.query()
        done.synchronize()
        t2 = time.perf_counter()

        print(
            f"run {idx + 1}: Python enqueue returned in {(t1 - t0) * 1000.0:.3f} ms; "
            f"event done immediately? {immediately_done}; "
            f"event synchronized after {(t2 - t0) * 1000.0:.3f} ms"
        )

    torch.testing.assert_close(out[:1024].cpu(), (x[:1024] + y[:1024]).cpu())
    print()
    print("Correctness check passed after synchronization.")
    print("Takeaway: Triton launch returns before the GPU has necessarily read/written all data.")


if __name__ == "__main__":
    main()
