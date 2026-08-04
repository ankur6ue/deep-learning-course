#!/usr/bin/env python3
"""Tutorial 11: torch.compile, CUDA async execution, and custom kernels.

This tutorial uses one transformer-like sequence to show the difference between
three execution styles:

1. Eager PyTorch: Python launches each PyTorch operation separately.
2. torch.compile: PyTorch/Inductor compiles a tensor subgraph and may fuse some
   operations, while library GEMMs still normally run as cuBLAS/cuBLASLt calls.
3. A custom Triton GPU kernel: we manually fuse a specific hot sequence.

    projected = output_projection(attn_heads)
    y = RMSNorm(projected + residual)

The projection is a GEMM. A good implementation should usually leave it to
cuBLAS/cuBLASLt. The fuseable part is the non-GEMM tail:

    residual add + RMSNorm

That is what torch.compile/Inductor can often fuse automatically, and what the
custom Triton variant fuses manually.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


EPS = 1e-6


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    x_stride: tl.constexpr,
    residual_stride: tl.constexpr,
    out_stride: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Triton programs are like CUDA thread blocks. This kernel launches one
    # program per token row. Each program processes the whole hidden dimension
    # for that token, so all reduction work for one row stays inside this one
    # GPU kernel launch.
    row = tl.program_id(0)

    # BLOCK_H must be a power of two for Triton's vectorized program. The mask
    # prevents reads/writes past the actual hidden size when hidden_size is not
    # itself a power of two.
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    # Loads are asynchronous from the CPU's perspective once this kernel is
    # queued. Inside the GPU kernel, these are vectorized per-row loads.
    x = tl.load(x_ptr + row * x_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * residual_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # We reduce in fp32, matching the common RMSNorm implementation pattern.
    # This changes the arithmetic order relative to eager PyTorch, so tiny
    # differences from eager are expected.
    added = x + residual
    variance = tl.sum(added * added, axis=0) / hidden_size
    out = added * tl.rsqrt(variance + eps) * weight
    tl.store(out_ptr + row * out_stride + offsets, out, mask=mask)


def triton_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Custom Triton kernel for the complete add + RMSNorm sequence."""
    if x.device.type != "cuda":
        raise RuntimeError("triton_add_rms_norm needs CUDA tensors")
    if x.shape != residual.shape or x.shape[-1] != weight.shape[0]:
        raise ValueError("x/residual/weight shapes do not match")
    # Keep the kernel simple by requiring contiguous rows. If the caller passed
    # a view with unusual strides, these calls may enqueue copies before the
    # custom kernel launch.
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    residual_2d = residual.reshape(-1, residual.shape[-1])
    out = torch.empty_like(x_2d)
    hidden_size = x_2d.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    # This is the CPU-side launch point for the custom GPU kernel. Like normal
    # PyTorch CUDA ops, this queues work on the current CUDA stream and returns
    # before the GPU necessarily finishes.
    _add_rms_norm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight,
        out,
        x_2d.stride(0),
        residual_2d.stride(0),
        out.stride(0),
        hidden_size,
        eps,
        BLOCK_H=block_h,
    )
    return out.view_as(x)


def eager_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    added = x + residual
    added_f32 = added.float()
    variance = added_f32.pow(2).mean(dim=-1, keepdim=True)
    out = added_f32 * torch.rsqrt(variance + eps) * weight.float()
    return out.to(x.dtype)


def eager_projection_add_rms_norm(
    attn_heads: torch.Tensor,
    residual: torch.Tensor,
    o_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    projected = F.linear(attn_heads, o_weight)
    return eager_add_rms_norm(projected, residual, norm_weight, eps)


def triton_tail_projection_add_rms_norm(
    attn_heads: torch.Tensor,
    residual: torch.Tensor,
    o_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    # This intentionally leaves the projection GEMM to cuBLAS/cuBLASLt. The
    # custom kernel fuses the complete non-GEMM tail: residual add + RMSNorm.
    projected = F.linear(attn_heads, o_weight)
    return triton_add_rms_norm(projected, residual, norm_weight, eps)


@dataclass(frozen=True)
class BenchResult:
    name: str
    median_ms: float
    min_ms: float
    max_abs_diff: float
    mean_abs_diff: float
    argmax_flips: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--attn-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--profile-variant",
        choices=[
            "none",
            "tail_eager",
            "tail_compile",
            "tail_triton",
        ],
        default="none",
        help=(
            "Run only one variant in a repeated measured loop. Use this when "
            "profiling so Nsight captures one implementation instead of the "
            "full benchmark."
        ),
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Mark only the measured profile loop with cudaProfilerStart/Stop. "
            "This does not profile by itself; pair it with Nsight options such "
            "as nsys --capture-range=cudaProfilerApi or ncu "
            "--profile-from-start off."
        ),
    )
    parser.add_argument(
        "--print-nsight-commands",
        action="store_true",
        help="Print example Nsight Systems / Nsight Compute commands.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        # CUDA events measure elapsed GPU time on the current stream. The CPU
        # still only queues work here; end.synchronize() is what waits for this
        # measured iteration to finish.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return out, timings


def compare_to_reference(
    name: str,
    out: torch.Tensor,
    reference: torch.Tensor,
    timings: list[float],
) -> BenchResult:
    diff = (out.float() - reference.float()).abs()
    ref_scores = reference.float()
    out_argmax = out.float().argmax(dim=-1)
    ref_argmax = ref_scores.argmax(dim=-1)
    flips = int((out_argmax != ref_argmax).sum().item())

    ordered = sorted(timings)
    return BenchResult(
        name=name,
        median_ms=ordered[len(ordered) // 2],
        min_ms=min(timings),
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        argmax_flips=flips,
    )


def print_table(title: str, results: list[BenchResult]) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'variant':<18} {'median ms':>10} {'min ms':>10} "
        f"{'max |diff|':>12} {'mean |diff|':>12} {'argmax flips':>13}"
    )
    for result in results:
        print(
            f"{result.name:<18} {result.median_ms:10.4f} {result.min_ms:10.4f} "
            f"{result.max_abs_diff:12.6g} {result.mean_abs_diff:12.6g} "
            f"{result.argmax_flips:13d}"
        )


def print_nsight_commands(script_path: Path, args: argparse.Namespace) -> None:
    base_nsys = (
        f'python "{script_path}" '
        f"--tokens {args.tokens} --hidden-size {args.hidden_size} "
        f"--attn-size {args.attn_size} --dtype {args.dtype} "
        "--warmup 5 --iters 20"
    )
    base_ncu = (
        f'python "{script_path}" '
        f"--tokens {args.tokens} --hidden-size {args.hidden_size} "
        f"--attn-size {args.attn_size} --dtype {args.dtype} "
        "--warmup 5 --iters 1"
    )
    print()
    print("Nsight examples")
    print("---------------")
    print("Two script flags matter for profiling:")
    print()
    print("  --profile-variant tail_compile")
    print("      Runs only that variant's repeated loop after warmup/compilation.")
    print("  --cuda-profiler-range")
    print("      Calls cudaProfilerStart() before the loop and cudaProfilerStop()")
    print("      after it. Nsight uses those calls as capture boundaries.")
    print()
    print("Kernel launch counts and timeline: use Nsight Systems.")
    print("The stats command summarizes how many GPU kernels launched and how much")
    print("time each kernel name consumed:")
    print()
    for variant in (
        "tail_eager",
        "tail_compile",
        "tail_triton",
    ):
        print(
            "  "
            f"nsys profile --force-overwrite=true --trace=cuda,nvtx "
            f"--capture-range=cudaProfilerApi "
            f"--output /tmp/{variant} {base_nsys} --profile-variant {variant}"
            f" --cuda-profiler-range"
        )
        print(f"  nsys stats --report cuda_gpu_kern_sum,nvtx_sum /tmp/{variant}.nsys-rep")
    print()
    print("Memory traffic for one kernel: use Nsight Compute.")
    print("This profiles only matching kernels inside the cudaProfilerStart/Stop")
    print("range and reports DRAM read/write bytes:")
    print()
    print(
        "  "
        f"ncu --target-processes all "
        f"--profile-from-start off "
        f"--kernel-name regex:.*add_rms_norm.* "
        f"--metrics dram__bytes_read.sum,dram__bytes_write.sum "
        f"{base_ncu} --profile-variant tail_triton --cuda-profiler-range"
    )
    print()
    print("For torch.compile, first identify the generated fused kernel name:")
    print()
    print(
        "  "
        f"TORCH_LOGS=output_code {base_ncu} --profile-variant tail_compile "
        "--cuda-profiler-range"
    )
    print()
    print("Then run ncu with a regex matching that generated name, for example:")
    print()
    print(
        "  "
        f"ncu --target-processes all "
        f"--profile-from-start off "
        f"--kernel-name regex:.*fused.* "
        f"--metrics dram__bytes_read.sum,dram__bytes_write.sum "
        f"{base_ncu} --profile-variant tail_compile --cuda-profiler-range"
    )
    print()
    print("For eager PyTorch there is no single fused tail kernel; use the nsys")
    print("kernel summary to choose individual kernels if you want ncu details.")
    print()
    print("If ncu fails with ERR_NVGPUCTRPERM, the NVIDIA driver is blocking")
    print("non-admin access to GPU performance counters. Nsight Systems launch")
    print("counts should still be useful, but Nsight Compute memory counters need")
    print("admin access or a driver setting change such as:")
    print()
    print("  options nvidia NVreg_RestrictProfilingToAdminUsers=0")
    print()
    print("in a root-owned /etc/modprobe.d/*.conf file, followed by reloading the")
    print("NVIDIA driver or rebooting. On shared systems, ask the administrator.")
    print()
    print("If sudo says 'ncu: command not found', sudo's PATH is missing the CUDA")
    print("tools directory. Use the absolute path to ncu, for example:")
    print()
    print("  sudo -E /usr/local/cuda/bin/ncu ...")


def run_profile_variant(
    variant: str,
    functions: dict[str, Callable[[], torch.Tensor]],
    warmup: int,
    iters: int,
    *,
    cuda_profiler_range: bool,
) -> None:
    fn = functions[variant]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    # The NVTX range shows up in Nsight Systems, and the optional profiler
    # start/stop lets nsys/ncu capture only this loop instead of compile/setup.
    torch.cuda.nvtx.range_push(f"{variant}_loop")
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.cuda.nvtx.range_pop()
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()
    print(f"{variant}: synchronized loop took {(t1 - t0) * 1000.0:.3f} ms")


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    if args.print_nsight_commands:
        print_nsight_commands(script_path, args)
        return
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(0)

    residual = torch.randn((args.tokens, args.hidden_size), device=device, dtype=dtype)
    norm_weight = torch.randn((args.hidden_size,), device=device, dtype=dtype)
    attn_heads = torch.randn((args.tokens, args.attn_size), device=device, dtype=dtype)
    o_weight = torch.randn((args.hidden_size, args.attn_size), device=device, dtype=dtype)

    # Tutorial 12 compares dynamic=True with dynamic=False. Here we use the
    # fixed-shape/static form so this file can focus on fusion.
    compiled_tail = torch.compile(
        eager_projection_add_rms_norm,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )

    functions: dict[str, Callable[[], torch.Tensor]] = {
        "tail_eager": lambda: eager_projection_add_rms_norm(
            attn_heads,
            residual,
            o_weight,
            norm_weight,
        ),
        "tail_compile": lambda: compiled_tail(
            attn_heads,
            residual,
            o_weight,
            norm_weight,
        ),
        "tail_triton": lambda: triton_tail_projection_add_rms_norm(
            attn_heads,
            residual,
            o_weight,
            norm_weight,
        ),
    }

    if args.profile_variant != "none":
        run_profile_variant(
            args.profile_variant,
            functions,
            args.warmup,
            args.iters,
            cuda_profiler_range=args.cuda_profiler_range,
        )
        return

    print("Tutorial 11: torch.compile vs custom kernel fusion")
    print(f"tokens={args.tokens}, hidden={args.hidden_size}, attn={args.attn_size}, dtype={args.dtype}")
    print()
    print("Reminder: all timings synchronize after each measured iteration.")
    print("Without that sync, Python would mostly measure enqueue time, not GPU completion.")

    tail_reference, tail_ref_timings = time_cuda(
        functions["tail_eager"],
        warmup=args.warmup,
        iters=args.iters,
    )
    tail_results = [
        compare_to_reference("eager", tail_reference, tail_reference, tail_ref_timings)
    ]
    for name, label in (
        ("tail_compile", "compile"),
        ("tail_triton", "custom tail"),
    ):
        out, timings = time_cuda(functions[name], warmup=args.warmup, iters=args.iters)
        tail_results.append(compare_to_reference(label, out, tail_reference, timings))

    print_table("Transformer tail: output projection + residual add + RMSNorm", tail_results)

    print()
    print("What to look for:")
    print("  - torch.compile can fuse the pointwise/reduction tail, but the projection")
    print("    GEMM normally remains a cuBLAS/cuBLASLt kernel.")
    print("  - The custom Triton variant still leaves the GEMM to cuBLAS, then uses")
    print("    one custom kernel for residual add + RMSNorm.")
    print("  - Small floating-point differences are expected because reductions and")
    print("    operation ordering differ. Argmax flips show when those differences")
    print("    changed the largest hidden-dimension index for a token row.")
    print()
    print("Run with --print-nsight-commands for Nsight Systems / Compute examples.")


if __name__ == "__main__":
    main()
