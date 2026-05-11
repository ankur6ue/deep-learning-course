#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_vllm_engine.config import EngineConfig
from simple_vllm_engine.engine import EngineResult, timed_run
from simple_vllm_engine.hf_loader import build_engine_from_pretrained
from simple_vllm_engine.requests import RequestSpec
from simple_vllm_engine.tokenizer import HFTokenizer


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
        "--timing",
        action="store_true",
        help="Collect synchronized timing breakdowns. This adds overhead but helps explain where time goes.",
    )
    return parser.parse_args()


def make_shared_chat_workload() -> list[tuple[list[dict[str, str]], int]]:
    shared_system = (
        "You are a careful infrastructure analyst. Read the request closely, "
        "reason from the supplied context, and answer in a concise technical "
        "style. Focus on throughput, latency, cache behavior, batching, GPU "
        "utilization, and practical tradeoffs in LLM serving systems."
    )
    shared_context = (
        "Context: A team is serving a decoder-only language model behind an API. "
        "Requests have varying prompt lengths, some prompts share a long common "
        "prefix, and traffic arrives continuously rather than in one static batch. "
        "The system uses a paged KV cache, chunked prefill, and decode-first "
        "continuous batching. Operators are comparing serial execution against a "
        "vLLM-style engine and want to understand throughput, inter-token latency, "
        "time to first token, prefix-cache reuse, and memory tradeoffs. "
        "Assume the system runs on one GPU and that some requests are long enough "
        "to require multiple prefill chunks."
    )
    requests = [
        (
            [
                {"role": "system", "content": shared_system},
                {
                    "role": "user",
                    "content": (
                        f"{shared_context} Explain why continuous batching helps throughput "
                        "when requests arrive at different times, and mention one tradeoff."
                    ),
                },
            ],
            0,
        ),
        (
            [
                {"role": "system", "content": shared_system},
                {
                    "role": "user",
                    "content": (
                        f"{shared_context} Explain how chunked prefill interacts with decode "
                        "latency, and when a smaller max batched token budget is helpful."
                    ),
                },
            ],
            0,
        ),
        (
            [
                {"role": "system", "content": shared_system},
                {
                    "role": "user",
                    "content": (
                        f"{shared_context} Describe how prefix caching helps when several "
                        "requests share the same long instruction prefix but ask different "
                        "follow-up questions."
                    ),
                },
            ],
            1,
        ),
        (
            [
                {"role": "system", "content": shared_system},
                {
                    "role": "user",
                    "content": (
                        f"{shared_context} Summarize how a paged KV cache reduces memory "
                        "movement and why a block table is needed when physical pages are "
                        "not contiguous."
                    ),
                },
            ],
            2,
        ),
        (
            [
                {"role": "system", "content": shared_system},
                {
                    "role": "user",
                    "content": (
                        f"{shared_context} Give a short explanation of why decode is more "
                        "memory-bound than prefill and how that affects batching decisions."
                    ),
                },
            ],
            2,
        ),
        (
            [
                {"role": "system", "content": shared_system},
                {
                    "role": "user",
                    "content": (
                        f"{shared_context} The team wants to compare a serial baseline with "
                        "a vLLM-style engine. Explain what improvement prefix caching alone "
                        "can provide, and what additional benefit continuous batching adds."
                    ),
                },
            ],
            4,
        ),
    ]
    return requests


def build_specs(
    tokenizer: HFTokenizer,
    max_new_tokens: int,
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
    for idx, (messages, arrival_step) in enumerate(make_shared_chat_workload()):
        specs.append(
            RequestSpec(
                request_id=f"req-{idx}",
                prompt_ids=tokenizer.encode_chat(messages),
                max_new_tokens=max_new_tokens,
                arrival_step=arrival_step,
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

    print("Running serial baseline...")
    serial_wall, serial_results, tokenizer, serial_timings = run_serial_workload(args.model_path, engine_config, specs)

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

    print(summarize("serial", serial_wall, serial_results))
    print(summarize("simple_vllm_v3", batched_wall, batched_results))
    if batched_wall > 0:
        print(f"speedup={serial_wall / batched_wall:.2f}x")
    print()

    if args.timing:
        print_timing_summary("serial", serial_timings)
        print_timing_summary("simple_vllm_v3", batched_timings)

    serial_by_id = index_results(serial_results)
    batched_by_id = index_results(batched_results)
    for spec in specs:
        serial_result = serial_by_id[spec.request_id]
        batched_result = batched_by_id[spec.request_id]
        ids_match = serial_result.generated_ids == batched_result.generated_ids
        print(
            f"{spec.request_id}: ids_match={ids_match}, "
            f"serial_finish={serial_result.finish_reason}, "
            f"batched_finish={batched_result.finish_reason}, "
            f"serial_prefix_hit_tokens={serial_result.prefix_cache_hits}, "
            f"batched_prefix_hit_tokens={batched_result.prefix_cache_hits}, "
            f"batched_scheduler_steps={batched_result.scheduler_steps}"
        )
        print("  output:", tokenizer.decode(batched_result.generated_ids))
        print()


if __name__ == "__main__":
    main()
