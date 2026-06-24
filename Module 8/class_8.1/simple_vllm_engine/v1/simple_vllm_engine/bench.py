#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

_VERSION_ROOT = Path(__file__).resolve().parent.parent
_TEACHING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VERSION_ROOT))
sys.path.insert(1, str(_TEACHING_ROOT))

from simple_vllm_engine.config import EngineConfig, ModelConfig
from simple_vllm_engine.engine import SerialEngine, SimpleVLLMEngine, timed_run
from simple_vllm_engine.requests import RequestSpec
from common.tokenizer import SimpleTokenizer


def make_demo_prompts() -> list[str]:
    """Return a small workload with shared prefixes for cache reuse demos."""
    shared = "summarize the quarterly revenue and margin trends for"
    return [
        f"{shared} business unit alpha over the last four quarters",
        f"{shared} business unit beta over the last four quarters",
        f"{shared} business unit gamma over the last four quarters",
        "explain how inventory grew faster than revenue and why that matters",
        "write a short note about customer churn and retention trends this month",
        f"{shared} business unit alpha over the last four quarters and compare with beta",
    ]


def build_specs(tokenizer: SimpleTokenizer) -> list[RequestSpec]:
    """Turn demo prompts into request specs with staggered arrival times.

    Args:
        tokenizer: Tokenizer used to encode each prompt into integer ids. The
            same tokenizer is shared across serial and batched runs so the
            workload is identical.
    """
    prompts = make_demo_prompts()
    arrival_steps = [0, 0, 2, 1, 1, 4]
    specs: list[RequestSpec] = []
    for idx, prompt in enumerate(prompts):
        specs.append(
            RequestSpec(
                request_id=f"req-{idx}",
                prompt_ids=tokenizer.encode(prompt, add_bos=True, add_eos=False),
                max_new_tokens=24,
                arrival_step=arrival_steps[idx],
            )
        )
    return specs


def summarize(name: str, wall_time_s: float, generated_lengths: list[int], prefix_hits: list[int]) -> str:
    """Format one benchmark summary line.

    Args:
        name: Label for the engine being summarized.
        wall_time_s: Total end-to-end runtime.
        generated_lengths: Number of generated tokens per request.
        prefix_hits: Prefix-cache hit tokens per request.
    """
    total_generated = sum(generated_lengths)
    toks_per_s = total_generated / wall_time_s if wall_time_s > 0 else 0.0
    return (
        f"{name}: wall={wall_time_s:.3f}s, "
        f"generated={total_generated} toks, "
        f"throughput={toks_per_s:.2f} toks/s, "
        f"mean_gen_len={statistics.mean(generated_lengths):.1f}, "
        f"prefix_hit_tokens={sum(prefix_hits)}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the demo benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--max-batch-tokens", type=int, default=96)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    """Run the serial baseline and the teaching engine on the same workload."""
    args = parse_args()
    prompts = make_demo_prompts()
    tokenizer = SimpleTokenizer.from_texts(prompts)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 3,
        num_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
    )
    engine_config = EngineConfig(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        device=args.device,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    specs = build_specs(tokenizer)
    torch.manual_seed(0)
    serial = SerialEngine(model_config, engine_config)
    torch.manual_seed(0)
    batched = SimpleVLLMEngine(model_config, engine_config)

    print("Kernel summary:")
    print(batched.kernel_summary())
    print()

    serial_run = timed_run(serial, specs)
    batched_run = timed_run(batched, specs)

    serial_lengths = [len(result.generated_ids) for result in serial_run.results]
    batched_lengths = [len(result.generated_ids) for result in batched_run.results]
    serial_hits = [result.prefix_cache_hits for result in serial_run.results]
    batched_hits = [result.prefix_cache_hits for result in batched_run.results]

    print(summarize("serial", serial_run.wall_time_s, serial_lengths, serial_hits))
    print(summarize("simple_vllm", batched_run.wall_time_s, batched_lengths, batched_hits))
    if batched_run.wall_time_s > 0:
        print(f"speedup={serial_run.wall_time_s / batched_run.wall_time_s:.2f}x")
    print()

    for result in batched_run.results:
        print(
            f"{result.request_id}: finish={result.finish_reason}, "
            f"generated={len(result.generated_ids)}, "
            f"prefix_hit_tokens={result.prefix_cache_hits}, "
            f"scheduler_steps={result.scheduler_steps}"
        )
        print("  output:", tokenizer.decode(result.generated_ids))


if __name__ == "__main__":
    main()
