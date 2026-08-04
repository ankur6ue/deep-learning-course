#!/usr/bin/env python3
"""Tutorial 9: cloned inputs and keepalive references for async kernels.

This script demonstrates two related ideas from `v5` RoPE:

1. If a custom kernel needs the values currently in a reusable workspace, pass an
   owned snapshot (`clone()`), not a view of the workspace that may be overwritten.
2. If queued GPU work may still use a tensor, keep a Python reference to that
   tensor until the work is known to be complete.

The three cases are intentionally explicit:

1. `copy_without_clone_on_sidestream`: the hazard appears. The copy is queued on
   a side stream, then the main/default stream reuses the workspace before the
   side-stream copy reads.
2. `copy_with_clone_on_sidestream`: cloning fixes the value hazard. The cloned
   tensor is created inside the launch helper, so we also keep it alive until
   the side stream finishes.
3. `copy_without_clone_on_default_stream`: the hazard disappears because copy
   and workspace reuse are ordered on the same stream.

The side-stream copy includes a small artificial GPU delay. This makes the
ordering visible in a deterministic tutorial; real serving code does not need a
delay because other queued GPU work naturally creates these windows.
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
    def _copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, values, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--delay-cycles",
        type=int,
        default=20_000_000,
        help=(
            "Side-stream GPU delay before the copy. This gives the default "
            "stream time to reuse the workspace before the side-stream copy reads."
        ),
    )
    return parser.parse_args()


def synchronize_main_stream() -> None:
    """Keep setup ordering simple for the tutorial."""
    torch.cuda.current_stream().synchronize()


def copy_without_clone_on_sidestream(
    *,
    workspace: torch.Tensor,
    out: torch.Tensor,
    side_stream: torch.cuda.Stream,
    n: int,
    block_size: int,
    delay_cycles: int,
) -> None:
    """Queue a delayed side-stream copy that reads caller-owned workspace."""
    grid = (triton.cdiv(n, block_size),)
    source = workspace

    with torch.cuda.stream(side_stream):
        # The caller synchronizes setup before this function, keeping the stream
        # mechanics in the three printed cases instead of hiding them here.
        if delay_cycles > 0:
            torch.cuda._sleep(delay_cycles)
        _copy_kernel[grid](source, out, n, BLOCK=block_size)


def copy_with_clone_on_sidestream(
    *,
    workspace: torch.Tensor,
    out: torch.Tensor,
    side_stream: torch.cuda.Stream,
    n: int,
    block_size: int,
    delay_cycles: int,
    keepalive: list[torch.Tensor],
) -> None:
    """Clone inside the launch helper, keep the clone alive, then queue copy."""
    grid = (triton.cdiv(n, block_size),)
    source = workspace.clone()

    # The clone is queued on the main/default stream. Make sure the cloned snapshot is fully
    # materialized before the side-stream kernel can read it, otherwise the copy kernel code
    # might see an uninitialized source, with the clone operation underway or not having started
    # see production_copy_with_clone_on_sidestream for how we can use CUDA events to make the
    # sidestream wait, rather than forcing a CPU blocking sync
    synchronize_main_stream()

    # This is the lifetime guard. Without it, `source` would be a local tensor
    # object that can go out of Python scope when this function returns, while
    # the queued side-stream kernel may still need its storage.
    keepalive.append(source)

    with torch.cuda.stream(side_stream):
        if delay_cycles > 0:
            torch.cuda._sleep(delay_cycles)
        _copy_kernel[grid](source, out, n, BLOCK=block_size)


def production_copy_with_clone_on_sidestream(
    *,
    workspace: torch.Tensor,
    out: torch.Tensor,
    side_stream: torch.cuda.Stream,
    n: int,
    block_size: int,
    delay_cycles: int,
    keepalive: list[torch.Tensor],
) -> None:
    """Production-style version: use stream ordering instead of CPU sync.

    This helper is not used by the printed tutorial cases because it introduces
    `wait_stream()`. It shows what real code should do to avoid blocking the CPU
    after `workspace.clone()`.
    """
    grid = (triton.cdiv(n, block_size),)

    # The clone is queued on the current stream. The returned Python tensor
    # exists immediately, but its device data may still be in flight.
    source = workspace.clone()
    producer_stream = torch.cuda.current_stream()

    # Keep the cloned tensor object alive while side_stream may still read it.
    keepalive.append(source)

    with torch.cuda.stream(side_stream):
        # GPU-side dependency only. This does not block Python. It tells
        # side_stream: do not run the copy until producer_stream has completed
        # work already queued before this wait, including workspace.clone().
        side_stream.wait_stream(producer_stream)
        if delay_cycles > 0:
            torch.cuda._sleep(delay_cycles)
        _copy_kernel[grid](source, out, n, BLOCK=block_size)


def copy_without_clone_on_default_stream(
    *,
    workspace: torch.Tensor,
    out: torch.Tensor,
    n: int,
    block_size: int,
) -> None:
    """Queue a copy on the default stream."""
    grid = (triton.cdiv(n, block_size),)
    _copy_kernel[grid](workspace, out, n, BLOCK=block_size)


def make_workspace(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    workspace = torch.arange(args.elements, device="cuda", dtype=torch.float32)
    out = torch.empty_like(workspace)
    return workspace, out


def warmup_copy_kernel(args: argparse.Namespace) -> None:
    """Warm up CUDA/Triton work before the printed examples.

    Otherwise the first printed case can include one-time setup costs. During
    those costs, the side-stream delay can finish before Python reaches the
    visible `workspace.fill_(-1.0)` line, which hides the stream-ordering
    behavior we are trying to teach.
    """
    n = args.elements
    workspace = torch.arange(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(workspace)
    copy_without_clone_on_default_stream(
        workspace=workspace,
        out=out,
        n=n,
        block_size=args.block_size,
    )
    torch.cuda.synchronize()

    workspace = torch.arange(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(workspace)
    side_stream = torch.cuda.Stream()
    synchronize_main_stream()
    copy_without_clone_on_sidestream(
        workspace=workspace,
        out=out,
        side_stream=side_stream,
        n=n,
        block_size=args.block_size,
        delay_cycles=args.delay_cycles,
    )
    workspace.fill_(-1.0)
    side_stream.synchronize()
    torch.cuda.synchronize()


def print_first_values(label: str, out: torch.Tensor, expected: str) -> None:
    first_values = out[:8].cpu().tolist()
    print(f"{label}: first copied values = {first_values}")
    print(f"  expected: {expected}")


def run_copy_without_clone_on_sidestream(args: argparse.Namespace) -> None:
    workspace, out = make_workspace(args)
    side_stream = torch.cuda.Stream()

    # Make the initial arange values real before we start the side-stream demo.
    # This keeps the launch helper free of stream-dependency setup code.
    synchronize_main_stream()

    copy_without_clone_on_sidestream(
        workspace=workspace,
        out=out,
        side_stream=side_stream,
        n=args.elements,
        block_size=args.block_size,
        delay_cycles=args.delay_cycles,
    )

    # Main/default stream reuses the workspace while the side stream is delayed.
    # Because the side-stream copy reads from workspace, it sees these new values.
    workspace.fill_(-1.0)
    side_stream.synchronize()

    print_first_values(
        "copy_without_clone_on_sidestream",
        out,
        "[-1, -1, ...], because the delayed side-stream copy read reused workspace storage",
    )


def run_copy_with_clone_on_sidestream(args: argparse.Namespace) -> None:
    workspace, out = make_workspace(args)
    side_stream = torch.cuda.Stream()
    keepalive: list[torch.Tensor] = []

    synchronize_main_stream()

    copy_with_clone_on_sidestream(
        workspace=workspace,
        out=out,
        side_stream=side_stream,
        n=args.elements,
        block_size=args.block_size,
        delay_cycles=args.delay_cycles,
        keepalive=keepalive,
    )

    # Main/default stream reuses the workspace, but the side stream reads from
    # the cloned snapshot instead of workspace.
    workspace.fill_(-1.0)
    side_stream.synchronize()

    print_first_values(
        "copy_with_clone_on_sidestream",
        out,
        "[0, 1, 2, ...], because the side-stream copy read the cloned snapshot",
    )

    # Drop the clone only after the side stream is done.
    keepalive.clear()


def run_copy_without_clone_on_default_stream(args: argparse.Namespace) -> None:
    workspace, out = make_workspace(args)

    # No clone is needed for this particular hazard. Copy and fill are queued on
    # the same stream, so fill cannot run until copy has already read workspace.
    copy_without_clone_on_default_stream(
        workspace=workspace,
        out=out,
        n=args.elements,
        block_size=args.block_size,
    )
    workspace.fill_(-1.0)
    torch.cuda.synchronize()

    print_first_values(
        "copy_without_clone_on_default_stream",
        out,
        "[0, 1, 2, ...], because same-stream ordering makes fill wait behind copy",
    )


def main() -> None:
    args = parse_args()
    if triton is None:
        print("Triton is not available; install/import Triton to run this tutorial.")
        return
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    print("Tutorial 9: cloned inputs and keepalive references")
    warmup_copy_kernel(args)
    print()
    run_copy_without_clone_on_sidestream(args)
    torch.cuda.synchronize()
    print()
    run_copy_with_clone_on_sidestream(args)
    torch.cuda.synchronize()
    print()
    run_copy_without_clone_on_default_stream(args)

    print()
    print("How this maps to the toy vLLM engine:")
    print("  RoPE runs on the same stream as decode workspace updates, so it can")
    print("  use positions.reshape(-1).contiguous() without cloning positions.")
    print("  Async token GPU-to-CPU copies run on a side stream, so that path still")
    print("  needs wait_stream/events, record_stream, and live tensor references.")
    print("The clone protects values only when another stream can race with reuse.")


if __name__ == "__main__":
    main()
