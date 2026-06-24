#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _drop_package_dir_from_sys_path() -> None:
    # When this file is launched directly, Python puts this package directory on
    # sys.path. That makes local `requests.py` shadow the third-party `requests`
    # package used by Transformers/Hugging Face. We import through the version
    # root instead, so the package dir itself should not be a top-level path.
    package_dir = Path(__file__).resolve().parent

    def is_package_dir(entry: str) -> bool:
        path = Path.cwd() if entry == "" else Path(entry)
        return path.resolve() == package_dir

    sys.path[:] = [entry for entry in sys.path if not is_package_dir(entry)]


_drop_package_dir_from_sys_path()

import torch

_VERSION_ROOT = Path(__file__).resolve().parent.parent
_TEACHING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VERSION_ROOT))
sys.path.insert(1, str(_TEACHING_ROOT))

from simple_vllm_engine.config import EngineConfig
from simple_vllm_engine.engine import EngineResult, timed_run
from common.hf_loader import build_engine_from_pretrained
from simple_vllm_engine.requests import RequestSpec
from common.tokenizer import HFTokenizer


DEFAULT_WORKLOAD_PATH = (
    Path(__file__).resolve().parents[2] / "workloads" / "shared_chat_workload.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ankur/dev/models/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=48)
    parser.add_argument("--max-decode-batch-size", type=int, default=8)
    parser.add_argument(
        "--workload-path",
        type=Path,
        default=DEFAULT_WORKLOAD_PATH,
        help="JSON workload file with request messages and arrival steps.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in a few sentences why continuous batching helps LLM serving throughput.",
    )
    parser.add_argument(
        "--single-prompt",
        action="store_true",
        help="Run a single prompt twice with staggered arrivals instead of the default multi-request workload.",
    )
    parser.add_argument(
        "--skip-serial",
        action="store_true",
        help="Only run the batched teaching engine. Useful for v3-v4 comparisons.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Collect synchronized timing breakdowns. This adds overhead but helps explain where time goes.",
    )
    return parser.parse_args()


def load_workload(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{path} must contain a non-empty 'requests' list")
    return requests


def build_specs(
    tokenizer: HFTokenizer,
    max_new_tokens: int,
    workload_path: Path,
    single_prompt: str | None = None,
) -> list[RequestSpec]:
    if single_prompt is not None:
        messages = [{"role": "user", "content": single_prompt}]
        prompt_ids = tokenizer.encode_chat(messages)
        return [
            RequestSpec(request_id="req-0", prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, arrival_step=0),
            RequestSpec(request_id="req-1", prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, arrival_step=1),
        ]

    specs: list[RequestSpec] = []
    for idx, item in enumerate(load_workload(workload_path)):
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"request {idx} in {workload_path} is missing 'messages'")
        specs.append(
            RequestSpec(
                request_id=str(item.get("request_id", f"req-{idx}")),
                prompt_ids=tokenizer.encode_chat(messages),
                arrival_step=int(item.get("arrival_step", 0)),
                max_new_tokens=max_new_tokens,
            )
        )
    return specs


def summarize(name: str, wall_time_s: float, results: list[EngineResult]) -> str:
    generated_lengths = [len(result.generated_ids) for result in results]
    prefix_hits = [result.prefix_cache_hits for result in results]
    total_generated = sum(generated_lengths)
    toks_per_s = total_generated / wall_time_s if wall_time_s > 0 else 0.0
    return (
        f"{name}: wall={wall_time_s:.3f}s, "
        f"generated={total_generated} toks, "
        f"throughput={toks_per_s:.2f} toks/s, "
        f"mean_gen_len={statistics.mean(generated_lengths):.1f}, "
        f"prefix_hit_tokens={sum(prefix_hits)}"
    )


def make_engine_config(args: argparse.Namespace) -> EngineConfig:
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    return EngineConfig(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        max_decode_batch_size=args.max_decode_batch_size,
        device=args.device,
        dtype=dtype,
        enable_timing=args.timing,
    )


def run_serial_workload(
    model_path: str,
    engine_config: EngineConfig,
    specs: list[RequestSpec],
) -> tuple[float, list[EngineResult], HFTokenizer, list[tuple[str, float, int]]]:
    engine, tokenizer = build_engine_from_pretrained(model_path, engine_config)
    results: list[EngineResult] = []
    if engine.engine_config.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if engine.engine_config.device.startswith("cuda") else None
    end = torch.cuda.Event(enable_timing=True) if engine.engine_config.device.startswith("cuda") else None
    wall_t0 = torch.cuda.Event(enable_timing=True) if engine.engine_config.device.startswith("cuda") else None
    wall_t1 = torch.cuda.Event(enable_timing=True) if engine.engine_config.device.startswith("cuda") else None

    if start is None:
        import time

        t0 = time.perf_counter()
        with torch.no_grad():
            for spec in sorted(specs, key=lambda item: item.arrival_step):
                results.extend(engine.run([spec]))
        if engine.engine_config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall = t1 - t0
    else:
        start.record()
        with torch.no_grad():
            for spec in sorted(specs, key=lambda item: item.arrival_step):
                results.extend(engine.run([spec]))
        end.record()
        torch.cuda.synchronize()
        wall = start.elapsed_time(end) / 1000.0

    timing = [(name, stat.total_s, stat.count) for name, stat in engine.profiler.summary()]
    del engine
    if engine_config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return wall, results, tokenizer, timing


def run_batched_workload(
    model_path: str,
    engine_config: EngineConfig,
    specs: list[RequestSpec],
) -> tuple[float, list[EngineResult], HFTokenizer, list[tuple[str, float, int]]]:
    engine, tokenizer = build_engine_from_pretrained(model_path, engine_config)
    run = timed_run(engine, specs)
    results = run.results
    wall = run.wall_time_s
    timing = [(name, stat.total_s, stat.count) for name, stat in engine.profiler.summary()]
    del engine
    if engine_config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return wall, results, tokenizer, timing


def index_results(results: list[EngineResult]) -> dict[str, EngineResult]:
    return {result.request_id: result for result in results}


def print_timing_summary(name: str, timings: list[tuple[str, float, int]]) -> None:
    if not timings:
        return
    print(f"{name} timing breakdown:")
    for section, total_s, count in timings:
        avg_ms = (total_s / count) * 1000.0 if count > 0 else 0.0
        print(f"  {section}: total={total_s:.3f}s count={count} avg={avg_ms:.2f}ms")
    print()


def main() -> None:
    args = parse_args()
    engine_config = make_engine_config(args)
    tokenizer = HFTokenizer.from_pretrained(args.model_path)
    specs = build_specs(
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        workload_path=args.workload_path,
        single_prompt=args.prompt if args.single_prompt else None,
    )

    print("Workload summary:")
    for spec in specs:
        print(
            f"{spec.request_id}: arrival={spec.arrival_step}, "
            f"prompt_tokens={len(spec.prompt_ids)}, max_new_tokens={spec.max_new_tokens}"
        )
    print()

    print("Running v3 batched engine...")
    batched_wall, batched_results, tokenizer, batched_timings = run_batched_workload(
        args.model_path, engine_config, specs
    )

    serial_wall = 0.0
    serial_results: list[EngineResult] = []
    serial_timings: list[tuple[str, float, int]] = []
    if not args.skip_serial:
        print("Running serial baseline...")
        serial_wall, serial_results, tokenizer, serial_timings = run_serial_workload(
            args.model_path, engine_config, specs
        )

    print()
    print("Kernel summary:")
    if args.device.startswith("cuda"):
        print(
            "Linear layers dispatch GEMM kernels via PyTorch/cuBLAS, while "
            "scaled_dot_product_attention dispatches CUDA attention kernels when supported."
        )
    else:
        print("CPU fallback: same software flow, without CUDA kernel dispatch.")
    print()

    if serial_results:
        print(summarize("serial", serial_wall, serial_results))
    print(summarize("simple_vllm_v3", batched_wall, batched_results))
    if serial_results and batched_wall > 0:
        print(f"speedup={serial_wall / batched_wall:.2f}x")
    print()

    if args.timing:
        if serial_timings:
            print_timing_summary("serial", serial_timings)
        print_timing_summary("simple_vllm_v3", batched_timings)

    batched_by_id = index_results(batched_results)
    for spec in specs:
        batched_result = batched_by_id[spec.request_id]
        if serial_results:
            serial_by_id = index_results(serial_results)
            serial_result = serial_by_id[spec.request_id]
            ids_match = serial_result.generated_ids == batched_result.generated_ids
            serial_finish = serial_result.finish_reason
            serial_prefix_hits = serial_result.prefix_cache_hits
        else:
            ids_match = "skipped"
            serial_finish = "skipped"
            serial_prefix_hits = 0
        print(
            f"{spec.request_id}: ids_match={ids_match}, "
            f"serial_finish={serial_finish}, "
            f"batched_finish={batched_result.finish_reason}, "
            f"serial_prefix_hit_tokens={serial_prefix_hits}, "
            f"batched_prefix_hit_tokens={batched_result.prefix_cache_hits}, "
            f"batched_scheduler_steps={batched_result.scheduler_steps}"
        )
        print("  output:", tokenizer.decode(batched_result.generated_ids))
        print()


if __name__ == "__main__":
    main()
