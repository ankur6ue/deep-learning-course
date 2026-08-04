#!/usr/bin/env python3
"""Tutorial 13: CUDA graph replay, validation, and graph buckets.

Tutorial 11 compared eager PyTorch, torch.compile, and a custom Triton kernel for
one transformer-like tail:

    projected = output_projection(attn_heads)
    y = RMSNorm(projected + residual)

This tutorial keeps that workload but changes the question:

    What changes when we capture the work into a CUDA graph?

CUDA graphs are about launch overhead and fixed execution shapes. A replay is a
single CPU-side graph launch, but it is not a single GPU kernel. The GPU still
executes the kernels captured in the graph: GEMM, pointwise kernels, reductions,
and so on. Nsight Systems will show the CPU issuing cudaGraphLaunch and the GPU
running the captured kernels.

The script demonstrates:

1. Capture and replay for eager PyTorch and torch.compile callables.
2. Graph validation by comparing replay output to eager gold output.
3. Fixed-shape graph buckets with static input tensors.
4. Shape mismatch behavior: a CUDA graph bucket does not recompile; replay with
   a different shape is an error in this teaching wrapper.
5. Multiple buckets and bucket selection based on the actual tensor shape.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


EPS = 1e-6


def projection_add_rms_norm(
    attn_heads: torch.Tensor,
    residual: torch.Tensor,
    o_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Transformer-like output projection + residual add + RMSNorm tail."""
    projected = F.linear(attn_heads, o_weight)
    added = projected + residual
    added_f32 = added.float()
    variance = added_f32.pow(2).mean(dim=-1, keepdim=True)
    out = added_f32 * torch.rsqrt(variance + eps) * norm_weight.float()
    return out.to(attn_heads.dtype)


@dataclass(frozen=True)
class Inputs:
    attn_heads: torch.Tensor
    residual: torch.Tensor
    o_weight: torch.Tensor
    norm_weight: torch.Tensor

    def as_args(self) -> tuple[torch.Tensor, ...]:
        return (self.attn_heads, self.residual, self.o_weight, self.norm_weight)

    @property
    def shape_key(self) -> tuple[int, int, int]:
        tokens = self.attn_heads.shape[0]
        attn_size = self.attn_heads.shape[1]
        hidden_size = self.residual.shape[1]
        return (tokens, hidden_size, attn_size)


@dataclass(frozen=True)
class ValidationResult:
    name: str
    max_abs_diff: float
    mean_abs_diff: float
    argmax_flips: int
    passed: bool


@dataclass(frozen=True)
class BenchResult:
    name: str
    median_ms: float
    min_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--attn-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--skip-compiled-graph",
        action="store_true",
        help="Only capture the eager callable. Useful if compiled capture is unsupported.",
    )
    parser.add_argument(
        "--profile-variant",
        choices=["none", "eager", "compile", "graph_eager", "graph_compile"],
        default="none",
        help="Run one repeated loop for Nsight profiling.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Call cudaProfilerStart/Stop around the measured profile loop. "
            "Use with nsys profile --capture-range=cudaProfilerApi."
        ),
    )
    parser.add_argument(
        "--print-nsight-commands",
        action="store_true",
        help="Print Nsight Systems commands for observing cudaGraphLaunch.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    attn_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Inputs:
    if tokens <= 0 or hidden_size <= 0 or attn_size <= 0:
        raise ValueError("tokens, hidden_size, and attn_size must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return Inputs(
        attn_heads=torch.randn((tokens, attn_size), device=device, dtype=dtype, generator=generator),
        residual=torch.randn((tokens, hidden_size), device=device, dtype=dtype, generator=generator),
        o_weight=torch.randn((hidden_size, attn_size), device=device, dtype=dtype, generator=generator),
        norm_weight=torch.randn((hidden_size,), device=device, dtype=dtype, generator=generator),
    )


def empty_like_inputs(inputs: Inputs) -> Inputs:
    return Inputs(
        attn_heads=torch.empty_like(inputs.attn_heads),
        residual=torch.empty_like(inputs.residual),
        o_weight=torch.empty_like(inputs.o_weight),
        norm_weight=torch.empty_like(inputs.norm_weight),
    )


def copy_inputs_(dst: Inputs, src: Inputs) -> None:
    if dst.shape_key != src.shape_key:
        raise ValueError(f"shape mismatch: bucket shape {dst.shape_key}, input shape {src.shape_key}")
    dst.attn_heads.copy_(src.attn_heads)
    dst.residual.copy_(src.residual)
    dst.o_weight.copy_(src.o_weight)
    dst.norm_weight.copy_(src.norm_weight)


def compare_to_gold(
    name: str,
    out: torch.Tensor,
    gold: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> ValidationResult:
    torch.cuda.synchronize()
    diff = (out.float() - gold.float()).abs()
    out_argmax = out.float().argmax(dim=-1)
    gold_argmax = gold.float().argmax(dim=-1)
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    argmax_flips = int((out_argmax != gold_argmax).sum().item())
    passed = bool(torch.allclose(out.float(), gold.float(), atol=atol, rtol=rtol))
    return ValidationResult(
        name=name,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        argmax_flips=argmax_flips,
        passed=passed,
    )


def print_validation(results: list[ValidationResult]) -> None:
    print()
    print("Graph validation against eager gold")
    print("-----------------------------------")
    print(
        f"{'variant':<16} {'status':<8} {'max |diff|':>12} "
        f"{'mean |diff|':>12} {'argmax flips':>13}"
    )
    for result in results:
        status = "PASS" if result.passed else "WARN"
        print(
            f"{result.name:<16} {status:<8} {result.max_abs_diff:12.6g} "
            f"{result.mean_abs_diff:12.6g} {result.argmax_flips:13d}"
        )


class CudaGraphBucket:
    """One fixed-shape CUDA graph replay bucket.

    The bucket owns static input tensors. Replays copy new values into those
    fixed-address tensors, then call graph.replay(). The shape is part of the
    bucket identity; a different shape needs a different bucket.
    """

    def __init__(
        self,
        *,
        name: str,
        fn: Callable[..., torch.Tensor],
        example_inputs: Inputs,
        warmup: int,
    ) -> None:
        self.name = name
        self.fn = fn
        self.shape_key = example_inputs.shape_key
        self.static_inputs = empty_like_inputs(example_inputs)
        copy_inputs_(self.static_inputs, example_inputs)
        self.graph = torch.cuda.CUDAGraph()
        self.static_output: torch.Tensor | None = None

        # Warm up on a side stream so CUDA libraries, compiled kernels, and the
        # caching allocator are initialized before capture. Capturing first-use
        # setup work often fails or gives a graph that is not representative.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(warmup):
                self.fn(*self.static_inputs.as_args())
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        # Capture on the current stream. The graph records the GPU work launched
        # by fn() for these exact tensor addresses and shapes.
        with torch.cuda.graph(self.graph):
            self.static_output = self.fn(*self.static_inputs.as_args())

    def copy_inputs(self, inputs: Inputs) -> None:
        copy_inputs_(self.static_inputs, inputs)

    def replay(self, inputs: Inputs | None = None) -> torch.Tensor:
        if inputs is not None:
            self.copy_inputs(inputs)
        self.graph.replay()
        if self.static_output is None:
            raise RuntimeError("graph output was not captured")
        return self.static_output

    def validate(
        self,
        *,
        validation_inputs: Inputs,
        gold_fn: Callable[..., torch.Tensor],
        atol: float,
        rtol: float,
    ) -> ValidationResult:
        gold = gold_fn(*validation_inputs.as_args())
        out = self.replay(validation_inputs)
        out_snapshot = out.detach().clone()
        return compare_to_gold(self.name, out_snapshot, gold, atol=atol, rtol=rtol)


class GraphBucketSet:
    """Collection of exact-shape graph buckets."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, int, int], CudaGraphBucket] = {}

    def add(self, bucket: CudaGraphBucket) -> None:
        self._buckets[bucket.shape_key] = bucket

    def replay(self, inputs: Inputs) -> torch.Tensor:
        bucket = self._buckets.get(inputs.shape_key)
        if bucket is None:
            known = ", ".join(str(shape) for shape in sorted(self._buckets))
            raise ValueError(f"no graph bucket for shape {inputs.shape_key}; known buckets: {known}")
        return bucket.replay(inputs)


def time_cuda(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    ordered = sorted(timings)
    return BenchResult(
        name="",
        median_ms=statistics.median(ordered),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def named_result(name: str, result: BenchResult) -> BenchResult:
    return BenchResult(
        name=name,
        median_ms=result.median_ms,
        min_ms=result.min_ms,
        max_ms=result.max_ms,
    )


def print_benchmarks(results: list[BenchResult]) -> None:
    print()
    print("Steady-state GPU timing")
    print("-----------------------")
    print(f"{'variant':<22} {'median ms':>10} {'min ms':>10} {'max ms':>10}")
    for result in results:
        print(f"{result.name:<22} {result.median_ms:10.4f} {result.min_ms:10.4f} {result.max_ms:10.4f}")
    print()
    print("Notes:")
    print("  - graph_* replay timings measure graph.replay() only.")
    print("  - graph_* copy+replay timings include device-to-device copies into the")
    print("    bucket's fixed-address input tensors before graph.replay().")
    print("  - CUDA graph replay is one CPU graph launch, not one GPU kernel.")


def wall_ms(fn: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def run_shape_and_bucket_demo(
    *,
    compiled_fn: Callable[..., torch.Tensor],
    base_eager_bucket: CudaGraphBucket,
    base_compiled_bucket: CudaGraphBucket | None,
    dtype: torch.dtype,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    alt_tokens = args.tokens + max(1, args.tokens // 2)
    alt_inputs = make_inputs(
        tokens=alt_tokens,
        hidden_size=args.hidden_size,
        attn_size=args.attn_size,
        dtype=dtype,
        device=device,
        seed=args.seed + 100,
    )

    print()
    print("Shape change behavior")
    print("---------------------")
    print(f"Base bucket shape: {base_eager_bucket.shape_key}")
    print(f"Alternate input shape: {alt_inputs.shape_key}")

    try:
        base_eager_bucket.replay(alt_inputs)
        torch.cuda.synchronize()
        print("Unexpected: base eager graph bucket accepted the alternate shape.")
    except ValueError as exc:
        print(f"CUDA graph bucket replay with alternate shape: ERROR ({exc})")

    first_ms = wall_ms(lambda: compiled_fn(*alt_inputs.as_args()))
    second_ms = wall_ms(lambda: compiled_fn(*alt_inputs.as_args()))
    print(
        "torch.compile callable at alternate shape: "
        f"first {first_ms:.3f} ms, second {second_ms:.3f} ms"
    )
    print(
        "Interpretation: torch.compile may compile/cache another shape-specialized "
        "graph, but an already captured CUDA graph bucket does not change shape."
    )

    print()
    print("Multiple graph buckets")
    print("----------------------")
    alt_eager_bucket = CudaGraphBucket(
        name=f"graph eager bucket {alt_inputs.shape_key}",
        fn=projection_add_rms_norm,
        example_inputs=alt_inputs,
        warmup=args.warmup,
    )
    bucket_set = GraphBucketSet()
    bucket_set.add(base_eager_bucket)
    bucket_set.add(alt_eager_bucket)

    for inputs in (
        make_inputs(
            tokens=args.tokens,
            hidden_size=args.hidden_size,
            attn_size=args.attn_size,
            dtype=dtype,
            device=device,
            seed=args.seed + 101,
        ),
        alt_inputs,
    ):
        out = bucket_set.replay(inputs)
        torch.cuda.synchronize()
        print(f"selected bucket for shape {inputs.shape_key}; output shape = {tuple(out.shape)}")

    missing_inputs = make_inputs(
        tokens=alt_tokens + max(1, args.tokens // 4),
        hidden_size=args.hidden_size,
        attn_size=args.attn_size,
        dtype=dtype,
        device=device,
        seed=args.seed + 102,
    )
    try:
        bucket_set.replay(missing_inputs)
        torch.cuda.synchronize()
        print("Unexpected: bucket set accepted a shape with no bucket.")
    except ValueError as exc:
        print(f"missing bucket lookup: ERROR ({exc})")

    if base_compiled_bucket is None:
        return
    try:
        base_compiled_bucket.replay(alt_inputs)
        torch.cuda.synchronize()
        print("Unexpected: base compiled graph bucket accepted the alternate shape.")
    except ValueError as exc:
        print(f"Compiled CUDA graph bucket with alternate shape: ERROR ({exc})")


def run_profile_variant(
    *,
    variant: str,
    eager_fn: Callable[[], torch.Tensor],
    compiled_fn_0: Callable[[], torch.Tensor],
    eager_bucket: CudaGraphBucket,
    compiled_bucket: CudaGraphBucket | None,
    iters: int,
    cuda_profiler_range: bool,
) -> None:
    if variant == "eager":
        fn = eager_fn
    elif variant == "compile":
        fn = compiled_fn_0
    elif variant == "graph_eager":
        fn = lambda: eager_bucket.replay()
    elif variant == "graph_compile":
        if compiled_bucket is None:
            raise RuntimeError("compiled graph bucket was not captured")
        fn = lambda: compiled_bucket.replay()
    else:
        raise ValueError(f"unknown profile variant {variant}")

    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"{variant}_loop")
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    if cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()
    print(f"profile variant {variant}: completed {iters} iterations")


def print_nsight_commands(script_path: Path, args: argparse.Namespace) -> None:
    base = (
        f'python "{script_path}" '
        f"--tokens {args.tokens} --hidden-size {args.hidden_size} "
        f"--attn-size {args.attn_size} --dtype {args.dtype} "
        "--warmup 3 --iters 10"
    )
    print()
    print("Nsight Systems examples")
    print("-----------------------")
    print("Use Nsight Systems to see cudaGraphLaunch on the CPU side and the captured")
    print("kernels on the GPU timeline. Graph replay is one CPU graph launch, not one")
    print("GPU kernel.")
    print()
    for variant in ("eager", "compile", "graph_eager", "graph_compile"):
        print(
            "  "
            "nsys profile --force-overwrite=true --trace=cuda,nvtx "
            "--capture-range=cudaProfilerApi "
            f"--output /tmp/tutorial13_{variant} "
            f"{base} --profile-variant {variant} --cuda-profiler-range"
        )
        print(f"  nsys stats --report cuda_api_sum,cuda_gpu_kern_sum /tmp/tutorial13_{variant}.nsys-rep")
    print()
    print("What to look for:")
    print("  - eager/compile loops show many CPU CUDA API launches per iteration.")
    print("  - graph_* loops should show cudaGraphLaunch for each replay iteration.")
    print("  - GPU kernel summaries still list the captured kernels; graph replay does")
    print("    not merge them into one GPU kernel.")


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
    base_inputs = make_inputs(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        attn_size=args.attn_size,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    validation_inputs = make_inputs(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        attn_size=args.attn_size,
        dtype=dtype,
        device=device,
        seed=args.seed + 1,
    )

    compiled_fn = torch.compile(
        projection_add_rms_norm,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )

    eager_fn_0 = lambda: projection_add_rms_norm(*base_inputs.as_args())
    compiled_fn_0 = lambda: compiled_fn(*base_inputs.as_args())

    print("Tutorial 13: CUDA graph replay and graph buckets")
    print(
        f"tokens={args.tokens}, hidden={args.hidden_size}, "
        f"attn={args.attn_size}, dtype={args.dtype}"
    )
    print()
    print("Compiling torch.compile callable at the base shape...")
    compiled_fn_0()
    torch.cuda.synchronize()

    print("Capturing eager CUDA graph bucket...")
    eager_bucket = CudaGraphBucket(
        name="graph eager",
        fn=projection_add_rms_norm,
        example_inputs=base_inputs,
        warmup=args.warmup,
    )

    compiled_bucket: CudaGraphBucket | None = None
    if not args.skip_compiled_graph:
        print("Capturing compiled CUDA graph bucket...")
        try:
            compiled_bucket = CudaGraphBucket(
                name="graph compile",
                fn=compiled_fn,
                example_inputs=base_inputs,
                warmup=args.warmup,
            )
        except Exception as exc:
            print(f"Compiled graph capture failed; continuing without it: {exc}")

    if args.profile_variant != "none":
        run_profile_variant(
            variant=args.profile_variant,
            eager_fn=eager_fn_0,
            compiled_fn_0=compiled_fn_0,
            eager_bucket=eager_bucket,
            compiled_bucket=compiled_bucket,
            iters=args.iters,
            cuda_profiler_range=args.cuda_profiler_range,
        )
        return

    validation_results = [
        eager_bucket.validate(
            validation_inputs=validation_inputs,
            gold_fn=projection_add_rms_norm,
            atol=2e-2,
            rtol=2e-2,
        )
    ]
    if compiled_bucket is not None:
        validation_results.append(
            compiled_bucket.validate(
                validation_inputs=validation_inputs,
                gold_fn=projection_add_rms_norm,
                atol=2e-2,
                rtol=2e-2,
            )
        )
    print_validation(validation_results)

    bench_results = [
        named_result("eager", time_cuda(eager_fn_0, warmup=args.warmup, iters=args.iters)),
        named_result("compile", time_cuda(compiled_fn_0, warmup=args.warmup, iters=args.iters)),
        named_result("graph eager replay", time_cuda(lambda: eager_bucket.replay(), warmup=args.warmup, iters=args.iters)),
        named_result(
            "graph eager copy+replay",
            time_cuda(lambda: eager_bucket.replay(base_inputs), warmup=args.warmup, iters=args.iters),
        ),
    ]
    if compiled_bucket is not None:
        bench_results.extend(
            [
                named_result(
                    "graph compile replay",
                    time_cuda(lambda: compiled_bucket.replay(), warmup=args.warmup, iters=args.iters),
                ),
                named_result(
                    "graph compile copy+replay",
                    time_cuda(lambda: compiled_bucket.replay(base_inputs), warmup=args.warmup, iters=args.iters),
                ),
            ]
        )
    print_benchmarks(bench_results)

    run_shape_and_bucket_demo(
        compiled_fn=compiled_fn,
        base_eager_bucket=eager_bucket,
        base_compiled_bucket=compiled_bucket,
        dtype=dtype,
        device=device,
        args=args,
    )

    print()
    print("Key takeaways")
    print("-------------")
    print("  - CUDA graph replay removes repeated CPU launch overhead for a fixed")
    print("    sequence of GPU work.")
    print("  - A graph bucket owns fixed-address static input/output tensors.")
    print("  - Shape changes do not recompile a captured CUDA graph. Pick another")
    print("    bucket or capture a new bucket for that shape.")
    print("  - Validation compares replay output to eager gold because graph capture")
    print("    bugs often look like stale inputs, stale outputs, or wrong bucket picks.")
    print("  - Nsight should show cudaGraphLaunch on the CPU side, while the GPU still")
    print("    runs the captured kernels.")
    print()
    print("Run with --print-nsight-commands for profiling commands.")


if __name__ == "__main__":
    main()
