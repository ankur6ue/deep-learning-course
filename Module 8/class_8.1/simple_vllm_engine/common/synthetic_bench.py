from __future__ import annotations

import argparse
import dataclasses
from importlib import import_module
import statistics
from typing import Any

import torch

from common.tokenizer import SimpleTokenizer


def _version_symbol(module_name: str, symbol_name: str) -> Any:
    module = import_module(f"simple_vllm_engine.{module_name}")
    return getattr(module, symbol_name)


def make_demo_prompts() -> list[str]:
    """Small synthetic workload with shared prefixes for debugger runs."""
    shared = "summarize the quarterly revenue and margin trends for"
    return [
        f"{shared} business unit alpha over the last four quarters",
        f"{shared} business unit beta over the last four quarters",
        f"{shared} business unit gamma over the last four quarters",
        "explain how inventory grew faster than revenue and why that matters",
        "write a short note about customer churn and retention trends this month",
        f"{shared} business unit alpha over the last four quarters and compare with beta",
    ]


def parse_args(version_label: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run a small synthetic debugger benchmark for {version_label}."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--max-batch-tokens", type=int, default=96)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=48)
    parser.add_argument("--max-decode-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help=(
            "Override the version's default attention backend. Useful values are "
            "dense_reference, paged_sdpa, flashinfer_paged, and flashinfer_paged_tc "
            "depending on the version."
        ),
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable torch.compile for debugger stepping in versions that expose it.",
    )
    parser.add_argument(
        "--torch-compile-scope",
        choices=["mlp", "input_qkv", "tail", "all"],
        default=None,
        help="Override the version's compile scope when torch.compile is enabled.",
    )
    return parser.parse_args()


def build_specs(tokenizer: SimpleTokenizer, max_new_tokens: int) -> list[Any]:
    RequestSpec = _version_symbol("requests", "RequestSpec")
    arrival_steps = [0, 0, 2, 1, 1, 4]
    specs: list[Any] = []
    for idx, prompt in enumerate(make_demo_prompts()):
        specs.append(
            RequestSpec(
                request_id=f"req-{idx}",
                prompt_ids=tokenizer.encode(prompt, add_bos=True, add_eos=False),
                max_new_tokens=max_new_tokens,
                arrival_step=arrival_steps[idx],
            )
        )
    return specs


def _filtered_dataclass_init(cls: type, values: dict[str, Any]) -> Any:
    fields = {field.name for field in dataclasses.fields(cls)}
    return cls(**{name: value for name, value in values.items() if name in fields})


def summarize(name: str, wall_time_s: float, results: list[Any]) -> str:
    generated_lengths = [len(result.generated_ids) for result in results]
    total_generated = sum(generated_lengths)
    toks_per_s = total_generated / wall_time_s if wall_time_s > 0 else 0.0
    mean_len = statistics.mean(generated_lengths) if generated_lengths else 0.0
    return (
        f"{name}: wall={wall_time_s:.3f}s, "
        f"generated={total_generated} toks, "
        f"throughput={toks_per_s:.2f} toks/s, "
        f"mean_gen_len={mean_len:.1f}, "
        f"prefix_hit_tokens={sum(result.prefix_cache_hits for result in results)}"
    )


def main(version_label: str) -> None:
    """Run one version's synthetic benchmark through its public engine API.

    `bench.py` is intended for debugger entry, so this uses a tiny random model
    and a built-in prompt set. Use `run_real.py` when you want HF checkpoint
    loading and real tokenizer/model correctness comparisons.
    """
    args = parse_args(version_label)

    ModelConfig = _version_symbol("config", "ModelConfig")
    EngineConfig = _version_symbol("config", "EngineConfig")
    SimpleVLLMEngine = _version_symbol("engine", "SimpleVLLMEngine")
    SerialEngine = _version_symbol("engine", "SerialEngine")
    timed_run = _version_symbol("engine", "timed_run")

    tokenizer = SimpleTokenizer.from_texts(make_demo_prompts())
    model_config = _filtered_dataclass_init(
        ModelConfig,
        {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.hidden_size * 3,
            "num_layers": args.layers,
            "num_attention_heads": args.heads,
            "num_key_value_heads": args.kv_heads,
        },
    )

    engine_values: dict[str, Any] = {
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "max_batch_tokens": args.max_batch_tokens,
        "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
        "max_decode_batch_size": args.max_decode_batch_size,
        "device": args.device,
        "dtype": torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.attention_backend is not None:
        engine_values["attention_backend"] = args.attention_backend
    if args.disable_torch_compile:
        engine_values["enable_torch_compile_model_body"] = False
    if args.torch_compile_scope is not None:
        engine_values["torch_compile_scope"] = args.torch_compile_scope

    engine_config = _filtered_dataclass_init(EngineConfig, engine_values)
    specs = build_specs(tokenizer, args.max_new_tokens)

    torch.manual_seed(0)
    batched = SimpleVLLMEngine(model_config, engine_config)
    torch.manual_seed(0)
    serial = SerialEngine(model_config, engine_config)

    if hasattr(batched, "kernel_summary"):
        print("Kernel summary:")
        print(batched.kernel_summary())
        print()

    batched_run = timed_run(batched, specs)
    serial_run = timed_run(serial, specs)

    print(summarize("serial", serial_run.wall_time_s, serial_run.results))
    print(summarize(f"simple_vllm_{version_label}", batched_run.wall_time_s, batched_run.results))
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
