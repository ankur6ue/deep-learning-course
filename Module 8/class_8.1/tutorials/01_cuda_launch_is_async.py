#!/usr/bin/env python3
"""Tutorial 1: CUDA launches are asynchronous even with plain PyTorch.

This script does not use Triton yet. It shows the baseline CUDA behavior that
Triton inherits: Python queues GPU work quickly, then continues. The GPU work is
only guaranteed complete after an explicit synchronization or after the CPU asks
for a GPU result in a way that forces synchronization, such as `.item()`.
"""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Square matrix size. Increase for a more visible async gap.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    device = torch.device("cuda")
    a = torch.randn((args.size, args.size), device=device, dtype=torch.float16)
    b = torch.randn((args.size, args.size), device=device, dtype=torch.float16)

    # Warm up cuBLAS and CUDA context creation so the timings below focus on
    # launch vs execution rather than one-time setup.
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    print("Tutorial 1: PyTorch CUDA launch vs actual completion")
    print(f"matrix size: {args.size} x {args.size}")
    print()

    for idx in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        t1 = time.perf_counter()
        # At this point Python has a Tensor object `c`, but the matmul may still
        # be running on the GPU. Synchronize to measure actual completion.
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        enqueue_ms = (t1 - t0) * 1000.0
        full_ms = (t2 - t0) * 1000.0
        print(
            f"run {idx + 1}: Python enqueue returned in {enqueue_ms:.3f} ms; "
            f"GPU completion took {full_ms:.3f} ms"
        )

    print()
    print("Now force a CPU read with `.item()`:")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    value = float(c[0, 0].item())
    t1 = time.perf_counter()
    print(
        f"matmul + c[0, 0].item() took {(t1 - t0) * 1000.0:.3f} ms "
        f"because `.item()` waits for the GPU result; sample value={value:.4f}"
    )

    print()
    print("Takeaway: seeing a Tensor in Python does not mean the GPU work is done.")


if __name__ == "__main__":
    main()
