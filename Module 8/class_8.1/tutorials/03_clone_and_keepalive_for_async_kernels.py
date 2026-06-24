#!/usr/bin/env python3
"""Tutorial 3: cloned inputs and keepalive references for async kernels.

This script demonstrates two related ideas from `v5` RoPE:

1. If a custom kernel needs the values currently in a reusable workspace, pass an
   owned snapshot (`clone()`), not a view of the workspace that may be overwritten.
2. If queued GPU work may still use a tensor, keep a Python reference to that
   tensor until the work is known to be complete.

The unsafe case below is deterministic: it models a delayed GPU read by reusing
the workspace before launching the copy. A real async race is harder to make
portable, but the important lesson is the same: if the kernel reads from mutable
workspace storage after that storage has been reused, it observes the new values.
"""

from __future__ import annotations

import argparse

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on local CUDA/Triton install.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _copy_kernel(src_ptr, dst_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, values, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def launch_delayed_copy(
    *,
    source: torch.Tensor,
    out: torch.Tensor,
    side_stream: torch.cuda.Stream,
    n: int,
    block_size: int,
    keepalive: list[torch.Tensor],
) -> torch.cuda.Event:
    """Queue a delayed copy on a side stream and keep source tensors alive.

    `keepalive` mirrors the teaching-engine pattern. It is deliberately simple:
    keep references to tensors that queued GPU work may still read or write.
    """
    grid = (triton.cdiv(n, block_size),)
    event = torch.cuda.Event()
    with torch.cuda.stream(side_stream):
        _copy_kernel[grid](source, out, n, BLOCK=block_size)
        event.record()
    keepalive.extend((source, out))
    return event


def run_case(*, use_clone: bool, args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    workspace = torch.arange(args.elements, device=device, dtype=torch.float32)
    out = torch.empty_like(workspace)
    side_stream = torch.cuda.Stream()
    keepalive: list[torch.Tensor] = []

    if use_clone:
        # Safe pattern: the side-stream kernel reads from a private snapshot.
        # Later workspace reuse cannot change this tensor's values.
        source = workspace.clone()
        label = "safe clone"
    else:
        # Unsafe pattern: the future kernel reads from the same workspace that
        # the main stream is about to reuse.
        source = workspace
        label = "unsafe workspace view"

    # Reuse the workspace for the next logical step before the delayed read.
    # If `source` is the workspace itself, the later kernel sees -1. If `source`
    # is a clone, the later kernel still sees the original [0, 1, 2, ...].
    workspace.fill_(-1.0)

    event = launch_delayed_copy(
        source=source,
        out=out,
        side_stream=side_stream,
        n=args.elements,
        block_size=args.block_size,
        keepalive=keepalive,
    )

    event.synchronize()
    first_values = out[:8].cpu().tolist()
    print(f"{label}: first copied values = {first_values}")
    if use_clone:
        print("  expected: [0, 1, 2, ...], because the kernel read the cloned snapshot")
    else:
        print("  expected: [-1, -1, ...], because the kernel read reused workspace storage")

    # Drop references only after the event has completed. In v5 we use a rolling
    # list instead of one event per RoPE launch because it is simpler for the
    # teaching engine.
    keepalive.clear()


def main() -> None:
    args = parse_args()
    if triton is None:
        print("Triton is not available; install/import Triton to run this tutorial.")
        return
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    print("Tutorial 3: cloned inputs and keepalive references")
    print()
    run_case(use_clone=False, args=args)
    torch.cuda.synchronize()
    print()
    run_case(use_clone=True, args=args)
    torch.cuda.synchronize()

    print()
    print("How this maps to v5 RoPE:")
    print("  decode positions workspace -> clone() -> flat_positions passed to Triton")
    print("  prefill positions are freshly allocated; the clone is mainly for the decode path")
    print("  q/k views + cloned positions + outputs -> _rope_keepalive list")
    print("The clone protects values. The keepalive list protects tensor/storage lifetime.")


if __name__ == "__main__":
    main()
