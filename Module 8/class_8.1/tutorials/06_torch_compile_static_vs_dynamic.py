#!/usr/bin/env python3
"""Tutorial 6: torch.compile(dynamic=False) vs dynamic=True.

Tutorial 5 keeps the comparison simple: eager, static compile, and a custom
kernel. This tutorial isolates a different question:

    What do we get, and what do we pay, when we use dynamic=True?

`dynamic=False` lets Dynamo/Inductor specialize aggressively to the concrete
shapes it sees. That is a good fit for fixed serving buckets.

`dynamic=True` asks Dynamo/Inductor to keep eligible dimensions symbolic. That
can let one compiled graph handle multiple token counts, reducing recompilation
pressure in variable-length serving workloads.

The workload below is a transformer-like non-GEMM tail:

    gate = GELU(x * gate_scale + gate_bias)
    up = x * up_scale + up_bias
    mixed = gate * up
    y = RMSNorm(mixed + residual)

It includes GELU so the compiled region has enough pointwise work to make
steady-state differences easier to see, but avoids GEMMs so cuBLAS heuristics do
not dominate the lesson.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


EPS = 1e-6


def gelu_residual_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate_scale: torch.Tensor,
    gate_bias: torch.Tensor,
    up_scale: torch.Tensor,
    up_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """A compile-friendly transformer tail with no GEMM.

    The operations are deliberately written as ordinary PyTorch ops. The point
    is to let Inductor decide how much of this it can fuse for static and
    dynamic shape modes.
    """
    gate = F.gelu(x * gate_scale + gate_bias, approximate="tanh")
    up = x * up_scale + up_bias
    mixed = gate * up
    added = mixed + residual
    added_f32 = added.float()
    variance = added_f32.pow(2).mean(dim=-1, keepdim=True)
    out = added_f32 * torch.rsqrt(variance + eps) * norm_weight.float()
    return out.to(x.dtype)


@dataclass(frozen=True)
class Inputs:
    x: torch.Tensor
    residual: torch.Tensor
    gate_scale: torch.Tensor
    gate_bias: torch.Tensor
    up_scale: torch.Tensor
    up_bias: torch.Tensor
    norm_weight: torch.Tensor

    def as_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.x,
            self.residual,
            self.gate_scale,
            self.gate_bias,
            self.up_scale,
            self.up_bias,
            self.norm_weight,
        )


@dataclass(frozen=True)
class BenchResult:
    name: str
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    max_abs_diff: float
    mean_abs_diff: float
    argmax_flips: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--shape-change-demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure first/second call wall time at alternate token counts.",
    )
    parser.add_argument(
        "--inspect-lowered-code",
        choices=["none", "static", "dynamic"],
        default="none",
        help=(
            "Compile and run only one mode at the base shape and one alternate "
            "token count. Use with TORCH_LOGS=output_code to inspect Inductor's "
            "lowered generated code."
        ),
    )
    parser.add_argument(
        "--print-lowered-code-commands",
        action="store_true",
        help="Print commands for comparing static and dynamic lowered code.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Inputs:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")

    torch.manual_seed(seed)
    return Inputs(
        x=torch.randn((tokens, hidden_size), device=device, dtype=dtype),
        residual=torch.randn((tokens, hidden_size), device=device, dtype=dtype),
        gate_scale=torch.randn((hidden_size,), device=device, dtype=dtype),
        gate_bias=torch.randn((hidden_size,), device=device, dtype=dtype),
        up_scale=torch.randn((hidden_size,), device=device, dtype=dtype),
        up_bias=torch.randn((hidden_size,), device=device, dtype=dtype),
        norm_weight=torch.randn((hidden_size,), device=device, dtype=dtype),
    )


def time_one_cuda_event(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    # CUDA events measure elapsed time on the current stream. Python queues the
    # work, and end.synchronize() waits only for this measured call to finish.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return out, float(start.elapsed_time(end))


def synchronized_wall_ms(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    """Measure one call including CPU-side compile/recompile overhead."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def compare_to_reference(
    name: str,
    out: torch.Tensor,
    reference: torch.Tensor,
    timings: list[float],
) -> BenchResult:
    ordered = sorted(timings)
    diff = (out.float() - reference.float()).abs()
    ref_scores = reference.float()
    out_argmax = out.float().argmax(dim=-1)
    ref_argmax = ref_scores.argmax(dim=-1)
    flips = int((out_argmax != ref_argmax).sum().item())

    return BenchResult(
        name=name,
        median_ms=statistics.median(ordered),
        p10_ms=percentile(ordered, 0.10),
        p90_ms=percentile(ordered, 0.90),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        argmax_flips=flips,
    )


def print_table(title: str, results: list[BenchResult]) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'variant':<16} {'median ms':>9} {'p10 ms':>9} {'p90 ms':>9} "
        f"{'min ms':>9} {'max ms':>9} {'max |diff|':>12} {'argmax flips':>13} "
    )
    for result in results:
        print(
            f"{result.name:<16} {result.median_ms:9.4f} {result.p10_ms:9.4f} "
            f"{result.p90_ms:9.4f} {result.min_ms:9.4f} {result.max_ms:9.4f} "
            f"{result.max_abs_diff:12.6g} {result.argmax_flips:13d} "
        )


def benchmark_interleaved(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iters: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    """Benchmark static/dynamic in randomized order to reduce ordering bias."""
    rng = random.Random(seed)
    labels = list(functions)

    for _ in range(warmup):
        order = labels[:]
        rng.shuffle(order)
        for label in order:
            functions[label]()
    torch.cuda.synchronize()

    schedule = ["static", "dynamic"] * iters
    rng.shuffle(schedule)

    last_outputs: dict[str, torch.Tensor] = {}
    timings: dict[str, list[float]] = {"static": [], "dynamic": []}
    for label in schedule:
        out, elapsed_ms = time_one_cuda_event(functions[label])
        last_outputs[label] = out
        timings[label].append(elapsed_ms)
    torch.cuda.synchronize()
    return last_outputs, timings


def alternate_token_counts(tokens: int) -> list[int]:
    # Use a smaller and a larger token count when possible. The exact values are
    # not important; they only need to differ from the warmed-up base shape.
    candidates = {
        max(1, tokens - max(1, tokens // 4)),
        tokens + max(1, tokens // 4),
        tokens + max(1, tokens // 2),
    }
    candidates.discard(tokens)
    return sorted(candidates)


def print_lowered_code_commands(script_path: str, args: argparse.Namespace) -> None:
    base = (
        f'TORCH_LOGS=output_code python "{script_path}" '
        f"--tokens {args.tokens} --hidden-size {args.hidden_size} "
        f"--dtype {args.dtype} --inspect-lowered-code"
    )
    print()
    print("Lowered-code inspection commands")
    print("--------------------------------")
    print("Run the two modes separately so the generated code is not interleaved:")
    print()
    print(f"  {base} static")
    print(f"  {base} dynamic")
    print()
    print("In the output, look for lines like:")
    print()
    print("  Output code written to: /tmp/torchinductor_.../...py")
    print()
    print("Static code usually contains shape-specialized guards such as:")
    print()
    print("  assert_size_stride(arg0_1, (2048, 4096), (4096, 1))")
    print()
    print("Dynamic code should keep eligible dimensions symbolic/runtime-dependent,")
    print("for example by using names like s0/s1 in shapes and strides. Inductor")
    print("may still specialize some dimensions when guards prove they are fixed.")


def run_lowered_code_inspection(
    *,
    mode: str,
    base_inputs: Inputs,
    base_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> None:
    dynamic = mode == "dynamic"
    compiled = torch.compile(
        gelu_residual_rms_norm,
        fullgraph=True,
        dynamic=dynamic,
        backend="inductor",
    )

    print(f"Inspecting lowered code for dynamic={dynamic}.")
    if "output_code" not in os.environ.get("TORCH_LOGS", ""):
        print("Tip: rerun with TORCH_LOGS=output_code to print generated Inductor code.")

    print(f"Running base shape: tokens={base_tokens}, hidden={hidden_size}")
    compiled(*base_inputs.as_args())
    torch.cuda.synchronize()

    alt_tokens = alternate_token_counts(base_tokens)[0]
    alt_inputs = make_inputs(
        tokens=alt_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
        seed=seed + 2000,
    )
    print(f"Running alternate shape: tokens={alt_tokens}, hidden={hidden_size}")
    compiled(*alt_inputs.as_args())
    torch.cuda.synchronize()

    print()
    print("What to inspect in the generated code:")
    print("  - dynamic=False should emit a separate specialized graph for the")
    print("    alternate token count.")
    print("  - dynamic=True should usually reuse one graph and represent the token")
    print("    dimension with runtime symbols/size arguments.")
    print("  - Hidden size is not varied by this inspection run, but dynamic=True")
    print("    may still represent it symbolically if Inductor chooses to do so.")


def run_shape_change_demo(
    *,
    compiled_static: Callable[..., torch.Tensor],
    compiled_dynamic: Callable[..., torch.Tensor],
    base_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> None:
    rng = random.Random(seed)

    print()
    print("Shape-change first-call cost")
    print("----------------------------")
    print(
        "Wall time includes CPU-side compilation when a new graph is needed. "
        "The order is randomized per shape so one mode does not always get the "
        "first touch of CUDA/cuBLAS/runtime caches."
    )
    print(
        f"{'tokens':>8} {'order':<16} {'static first ms':>15} "
        f"{'static second ms':>16} {'dynamic first ms':>16} "
        f"{'dynamic second ms':>17}"
    )

    for index, tokens in enumerate(alternate_token_counts(base_tokens)):
        inputs = make_inputs(
            tokens=tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            seed=seed + 1000 + index,
        )
        fns = {
            "static": lambda inputs=inputs: compiled_static(*inputs.as_args()),
            "dynamic": lambda inputs=inputs: compiled_dynamic(*inputs.as_args()),
        }
        order = ["static", "dynamic"]
        rng.shuffle(order)

        measured: dict[str, tuple[float, float]] = {}
        for label in order:
            _, first_ms = synchronized_wall_ms(fns[label])
            _, second_ms = synchronized_wall_ms(fns[label])
            measured[label] = (first_ms, second_ms)

        static_first, static_second = measured["static"]
        dynamic_first, dynamic_second = measured["dynamic"]
        print(
            f"{tokens:8d} {'/'.join(order):<16} {static_first:15.3f} "
            f"{static_second:16.3f} {dynamic_first:16.3f} "
            f"{dynamic_second:17.3f}"
        )

    print(
        "Note: torch.cuda.empty_cache() would only clear the allocator cache; it "
        "would not unload compiled kernels or fully reset CUDA library state. "
        "Randomizing order is a better low-friction guard against simple bias."
    )


def main() -> None:
    args = parse_args()
    script_path = os.path.abspath(__file__)
    if args.print_lowered_code_commands:
        print_lowered_code_commands(script_path, args)
        return
    if not torch.cuda.is_available():
        print("CUDA is not available; this tutorial needs a CUDA GPU.")
        return

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    base_inputs = make_inputs(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    if args.inspect_lowered_code != "none":
        run_lowered_code_inspection(
            mode=args.inspect_lowered_code,
            base_inputs=base_inputs,
            base_tokens=args.tokens,
            hidden_size=args.hidden_size,
            dtype=dtype,
            device=device,
            seed=args.seed,
        )
        return

    compiled_static = torch.compile(
        gelu_residual_rms_norm,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )
    compiled_dynamic = torch.compile(
        gelu_residual_rms_norm,
        fullgraph=True,
        dynamic=True,
        backend="inductor",
    )

    static_fn = lambda: compiled_static(*base_inputs.as_args())
    dynamic_fn = lambda: compiled_dynamic(*base_inputs.as_args())

    print("Tutorial 6: torch.compile static vs dynamic shapes")
    print(f"tokens={args.tokens}, hidden={args.hidden_size}, dtype={args.dtype}")
    print("Compiling both modes at the base shape...")

    static_fn()
    dynamic_fn()
    torch.cuda.synchronize()

    reference = gelu_residual_rms_norm(*base_inputs.as_args())
    last_outputs, timings = benchmark_interleaved(
        {"static": static_fn, "dynamic": dynamic_fn},
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )

    results = [
        compare_to_reference("dynamic=False", last_outputs["static"], reference, timings["static"]),
        compare_to_reference("dynamic=True", last_outputs["dynamic"], reference, timings["dynamic"]),
    ]
    print_table("Steady-state base-shape benchmark", results)

    if args.shape_change_demo:
        run_shape_change_demo(
            compiled_static=compiled_static,
            compiled_dynamic=compiled_dynamic,
            base_tokens=args.tokens,
            hidden_size=args.hidden_size,
            dtype=dtype,
            device=device,
            seed=args.seed,
        )

    print()
    print("What to look for:")
    print("  - dynamic=False should usually pay a compile cost for new token counts.")
    print("  - dynamic=True can often reuse one graph across token counts.")
    print("  - The steady-state winner is empirical. A symbolic graph is more flexible,")
    print("    but a static graph may allow more shape-specialized code.")
    print("  - Tiny numeric differences can still appear because generated kernels may")
    print("    reassociate arithmetic differently from eager PyTorch.")


if __name__ == "__main__":
    main()
