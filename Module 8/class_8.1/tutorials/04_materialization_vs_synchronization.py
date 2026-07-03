#!/usr/bin/env python3
"""Tutorial 4: materialization, views, and synchronization.

The word "materialization" is easy to overload. In this tutorial it means
"storage materialization": PyTorch has produced a Tensor object, and for
copy/math ops that Tensor has output storage on the GPU. It does not mean the
CPU can already read the computed values.

When the CPU actually observes a CUDA value, for example through .item(),
.tolist(), .cpu(), or numpy(), PyTorch must wait until the queued GPU work that
produces that value has finished. That wait is synchronization, not just storage
materialization.

Useful mental model:

    1. view-like ops usually create metadata only
    2. copy/contiguous/clone/to usually allocate storage and enqueue a copy
    3. math ops usually allocate output storage and enqueue GPU work
    4. CPU reads such as .item(), .tolist(), .cpu(), and numpy() wait for values

This is the distinction that matters in simple_vllm_engine/v5: the forward pass
creates real tensors before the final sync flag, but most CUDA work is still
queued asynchronously until a synchronization point.
"""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=2048,
        help="Square matrix size for the async timing example.",
    )
    return parser.parse_args()


def storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def describe(label: str, tensor: torch.Tensor, *, root: torch.Tensor | None = None) -> None:
    shares = ""
    if root is not None:
        shares = f", shares root storage={storage_ptr(tensor) == storage_ptr(root)}"
    print(
        f"{label:<30} shape={tuple(tensor.shape)!s:<16} "
        f"stride={tuple(tensor.stride())!s:<16} "
        f"contig={str(tensor.is_contiguous()):<5} "
        f"storage=0x{storage_ptr(tensor):x}{shares}"
    )


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def show_views_vs_copies(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    x = torch.arange(args.rows * args.cols, device=device, dtype=torch.float16).reshape(
        args.rows,
        args.cols,
    )
    torch.cuda.synchronize()

    print_section("1. View-like operations: metadata, same storage")
    describe("x", x)
    describe("x.view(-1)", x.view(-1), root=x)
    describe("x.reshape(-1)", x.reshape(-1), root=x)
    describe("x[:, :cols//2]", x[:, : args.cols // 2], root=x)
    first, second = x.split(args.cols // 2, dim=1)
    describe("split first", first, root=x)
    describe("split second", second, root=x)
    describe("x.t()", x.t(), root=x)
    print("These operations create Tensor metadata/views. They do not need the GPU values.")

    print_section("2. Reshape can materialize when a view is impossible")
    transposed = x.t()
    describe("transposed", transposed, root=x)
    describe("transposed.reshape(-1)", transposed.reshape(-1), root=x)
    describe("transposed.contiguous()", transposed.contiguous(), root=x)
    print("A non-contiguous transpose cannot always be flattened as a view.")
    print("In that case, reshape behaves like a copy and returns storage of its own.")

    print_section("3. Allocation/copy/math operations: new GPU storage")
    describe("torch.empty_like(x)", torch.empty_like(x), root=x)
    describe("x.clone()", x.clone(), root=x)
    describe("x.to(float32)", x.to(torch.float32), root=x)
    describe("x + 1", x + 1, root=x)
    weights = torch.randn((args.cols, args.cols), device=device, dtype=torch.float16)
    describe("x @ weights", x @ weights, root=x)
    print("torch.empty_like only allocates output storage; it does not compute values.")
    print("These tensors are storage-materialized outputs.")
    print("Creating them did not force Python to wait for the GPU to finish")
    print("computing their values.")


def show_enqueue_vs_completion(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    size = args.matmul_size
    a = torch.randn((size, size), device=device, dtype=torch.float16)
    b = torch.randn((size, size), device=device, dtype=torch.float16)

    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    print_section("4. New storage is not proof that GPU work is complete")
    done = torch.cuda.Event()
    t0 = time.perf_counter()
    c = a @ b
    c_view = c.reshape(-1)
    done.record()
    t1 = time.perf_counter()
    immediately_done = done.query()
    done.synchronize()
    t2 = time.perf_counter()

    print(f"matmul output type: {type(c).__name__}")
    print(f"view output type:   {type(c_view).__name__}")
    print(
        f"Python returned after {(t1 - t0) * 1000.0:.3f} ms; "
        f"event done immediately? {immediately_done}; "
        f"actual completion after {(t2 - t0) * 1000.0:.3f} ms"
    )
    print("The Tensor objects and output storage existed before the GPU necessarily")
    print("finished the matmul.")

    print_section("5. CPU reads force synchronization")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d = a @ b
    gpu_scalar = d[0, 0]
    t_mid = time.perf_counter()
    sample = float(gpu_scalar.item())
    t1 = time.perf_counter()
    print(f"d[0, 0] returned a {type(gpu_scalar).__name__} on {gpu_scalar.device}.")
    print(
        f"matmul + d[0, 0] returned after {(t_mid - t0) * 1000.0:.3f} ms; "
        f"matmul + .item() took {(t1 - t0) * 1000.0:.3f} ms; "
        f"sample={sample:.4f}"
    )
    print("Indexing a CUDA tensor still gives a CUDA tensor. .item() needs the")
    print("actual value on the CPU, so it waits for the GPU and copies that value.")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    print("Tutorial 4: materialization vs synchronization")
    print()
    print("Storage-materialized means a Tensor/storage exists.")
    print("Synchronized means the CPU has waited for queued GPU work to finish.")

    show_views_vs_copies(args)
    show_enqueue_vs_completion(args)

    print()
    print("Mapping back to v5:")
    print("  q/k/v reshapes are usually metadata views.")
    print("  F.linear, Triton RMSNorm, Triton RoPE, FlashAttention, and lm_head")
    print("  produce real output tensors while their CUDA work can remain queued.")
    print("  SIMPLE_VLLM_SYNC_AFTER_ROPE=1 and SIMPLE_VLLM_SYNC_AFTER_FORWARD=1")
    print("  are debug flags that insert explicit waits; both are off by default.")


if __name__ == "__main__":
    main()
