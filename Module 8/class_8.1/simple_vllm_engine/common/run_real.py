from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from importlib import import_module
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from common.hf_loader import build_engine_from_pretrained
from common.tokenizer import HFTokenizer


DEFAULT_WORKLOAD_PATH = Path(__file__).resolve().parents[1] / "workloads" / "shared_chat_workload.json"


@dataclass(frozen=True)
class VersionSettings:
    """Small per-version surface for the shared real-model runner.

    Each version wrapper should define only the knobs that represent that
    lesson. The shared runner handles CLI parsing, workload loading, timing, and
    JSON output so those support details do not distract from engine code.
    """

    version_name: str
    version_topic: str
    attention_backend: str
    enable_torch_compile: bool = False
    torch_compile_scope: str = "mlp"
    enable_gpu_decode_state: bool = False
    enable_triton_decode_metadata_kernel: bool = False
    enable_eager_decode_workspace: bool = False
    enable_async_output_processing: bool = False
    enable_decode_cuda_graphs: bool = False
    attention_backend_choices: tuple[str, ...] | None = None
    expose_torch_compile_scope_arg: bool = False
    expose_unsafe_decode_graph_arg: bool = False


def _version_symbol(module_name: str, symbol_name: str) -> Any:
    module = import_module(f"simple_vllm_engine.{module_name}")
    return getattr(module, symbol_name)


def parse_args(settings: VersionSettings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {settings.version_name}: {settings.version_topic}."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ankur/dev/models/Llama-3.1-8B-Instruct",
        help="Local Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=48)
    parser.add_argument("--max-decode-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Optional context-length cap used by RoPE setup. When omitted, "
            "the checkpoint's full configured context length is used."
        ),
    )
    parser.add_argument(
        "--workload-path",
        type=Path,
        default=DEFAULT_WORKLOAD_PATH,
        help="JSON file containing request messages and arrival steps.",
    )
    parser.add_argument(
        "--ignore-arrival-steps",
        action="store_true",
        help="Set every request arrival_step to 0 for offline-style comparisons.",
    )
    parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help=(
            "Enable the toy prefix-cache bookkeeping. It is off by default here "
            "because prefix caching is not one of the v1-v8 teaching steps."
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Generate exactly max_new_tokens unless the request budget ends first.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Collect synchronized timing breakdowns. This adds overhead.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable timing and generated token ids.",
    )

    if settings.attention_backend_choices is not None:
        parser.add_argument(
            "--attention-backend",
            choices=list(settings.attention_backend_choices),
            default=settings.attention_backend,
            help=(
                "Teaching comparison for this version. For example, v3 can run "
                "a dense reference attention path or the direct paged-SDPA path."
            ),
        )

    if settings.expose_torch_compile_scope_arg:
        parser.add_argument(
            "--torch-compile-scope",
            choices=["mlp", "input_qkv", "tail", "all"],
            default=settings.torch_compile_scope,
            help=(
                "Teaching knob for torch.compile boundaries. Pure tensor pieces "
                "compile safely; scheduler/cache mutation should stay outside."
            ),
        )

    if settings.expose_unsafe_decode_graph_arg:
        parser.add_argument(
            "--unsafe-decode-cuda-graphs",
            action="store_true",
            help=(
                "Benchmark-only graph replay knob. Current graph replay depends "
                "on static attention-wrapper state, so it is opt-in."
            ),
        )
    else:
        parser.set_defaults(unsafe_decode_cuda_graphs=False)

    return parser.parse_args()


def load_workload(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{path} must contain a non-empty 'requests' list")
    return requests


def build_specs(
    tokenizer: HFTokenizer,
    *,
    workload_path: Path,
    max_new_tokens: int,
    ignore_arrival_steps: bool,
) -> list[Any]:
    """Tokenize the JSON workload into the active version's request specs.

    Minimal workload example:

        {
          "requests": [
            {
              "request_id": "req-0",
              "arrival_step": 0,
              "messages": [{"role": "user", "content": "Explain paged KV."}]
            }
          ]
        }

    Keeping prompts in JSON makes the runner about serving mechanics; readers do
    not have to scan Python code to find hidden benchmark prompts.
    """
    RequestSpec = _version_symbol("requests", "RequestSpec")
    specs: list[Any] = []
    for idx, item in enumerate(load_workload(workload_path)):
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"request {idx} in {workload_path} is missing 'messages'")
        specs.append(
            RequestSpec(
                request_id=str(item.get("request_id", f"req-{idx}")),
                prompt_ids=tokenizer.encode_chat(messages),
                arrival_step=0 if ignore_arrival_steps else int(item.get("arrival_step", 0)),
                max_new_tokens=max_new_tokens,
            )
        )
    return specs


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


def make_engine_config(args: argparse.Namespace, settings: VersionSettings) -> Any:
    EngineConfig = _version_symbol("config", "EngineConfig")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    attention_backend = getattr(args, "attention_backend", settings.attention_backend)
    compile_scope = getattr(args, "torch_compile_scope", settings.torch_compile_scope)

    config_kwargs = dict(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        max_decode_batch_size=args.max_decode_batch_size,
        max_model_len=args.max_model_len,
        enable_prefix_cache=args.enable_prefix_cache,
        attention_backend=attention_backend,
        device=args.device,
        dtype=dtype,
        enable_timing=args.timing,
        enable_decode_cuda_graphs=settings.enable_decode_cuda_graphs,
        unsafe_decode_cuda_graphs=args.unsafe_decode_cuda_graphs,
        enable_torch_compile_model_body=settings.enable_torch_compile,
        torch_compile_scope=compile_scope,
        torch_compile_fullgraph=False,
        torch_compile_dynamic=True,
        prewarm_torch_compile=True,
        enable_gpu_decode_state=settings.enable_gpu_decode_state,
        enable_triton_decode_metadata_kernel=settings.enable_triton_decode_metadata_kernel,
        enable_eager_decode_workspace=settings.enable_eager_decode_workspace,
        enable_async_output_processing=settings.enable_async_output_processing,
        ignore_eos=args.ignore_eos,
    )
    public_fields = {field.name for field in fields(EngineConfig)}
    return EngineConfig(
        **{name: value for name, value in config_kwargs.items() if name in public_fields}
    )


def print_workload_summary(specs: list[Any]) -> None:
    print("Workload summary:")
    for spec in specs:
        print(
            f"  {spec.request_id}: arrival={spec.arrival_step}, "
            f"prompt_tokens={len(spec.prompt_ids)}, max_new_tokens={spec.max_new_tokens}"
        )
    print()


def print_timing_summary(timings: list[tuple[str, float, int]]) -> None:
    if not timings:
        return
    print("Timing breakdown:")
    for section, total_s, count in timings:
        avg_ms = (total_s / count) * 1000.0 if count > 0 else 0.0
        print(f"  {section}: total={total_s:.3f}s count={count} avg={avg_ms:.2f}ms")
    print()


def _empty_graph_warmup() -> Any:
    return SimpleNamespace(captured=[], skipped=[])


def run_workload(
    model_path: str,
    engine_config: Any,
    specs: list[Any],
) -> tuple[float, list[Any], HFTokenizer, list[tuple[str, float, int]], Any, str]:
    timed_run = _version_symbol("engine", "timed_run")

    if (
        getattr(engine_config, "enable_torch_compile_model_body", False)
        and getattr(engine_config, "prewarm_torch_compile", False)
    ):
        # torch.compile compiles on first use. This scratch run keeps the printed
        # benchmark closer to steady-state serving time instead of mixing in the
        # one-time compiler cost.
        warmup_engine, _ = build_engine_from_pretrained(model_path, engine_config)
        if hasattr(warmup_engine, "warmup_decode_cuda_graphs"):
            warmup_engine.warmup_decode_cuda_graphs(specs)
        timed_run(warmup_engine, specs)
        del warmup_engine
        if engine_config.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    engine, tokenizer = build_engine_from_pretrained(model_path, engine_config)
    kernel_summary = engine.kernel_summary() if hasattr(engine, "kernel_summary") else ""
    if hasattr(engine, "warmup_decode_cuda_graphs"):
        decode_graph_warmup = engine.warmup_decode_cuda_graphs(specs)
    else:
        decode_graph_warmup = _empty_graph_warmup()
    run = timed_run(engine, specs)
    timings = [(name, stat.total_s, stat.count) for name, stat in engine.profiler.summary()]
    del engine
    if engine_config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return run.wall_time_s, run.results, tokenizer, timings, decode_graph_warmup, kernel_summary


def main(settings: VersionSettings) -> None:
    args = parse_args(settings)
    engine_config = make_engine_config(args, settings)
    tokenizer = HFTokenizer.from_pretrained(args.model_path)
    specs = build_specs(
        tokenizer,
        workload_path=args.workload_path,
        max_new_tokens=args.max_new_tokens,
        ignore_arrival_steps=args.ignore_arrival_steps,
    )

    print(f"Running {settings.version_name}: {settings.version_topic}")
    print(f"attention_backend={engine_config.attention_backend}")
    print(
        "fixed_optimizations: "
        f"compile={getattr(engine_config, 'enable_torch_compile_model_body', False)} "
        f"compile_scope={getattr(engine_config, 'torch_compile_scope', 'none')} "
        f"gpu_decode_state={getattr(engine_config, 'enable_gpu_decode_state', False)} "
        f"async_output={getattr(engine_config, 'enable_async_output_processing', False)} "
        f"decode_graphs={getattr(engine_config, 'enable_decode_cuda_graphs', False)}"
    )

    if args.timing:
        print("timing mode enables profiler sections, so decode graph replay will not be used.")

    print()
    print_workload_summary(specs)
    wall_s, results, tokenizer, timings, decode_graph_warmup, kernel_summary = run_workload(
        args.model_path,
        engine_config,
        specs,
    )

    if kernel_summary:
        print("Kernel summary:")
        print(kernel_summary)
        print()

    print(summarize(settings.version_name, wall_s, results))
    if getattr(engine_config, "enable_decode_cuda_graphs", False):
        print(
            "decode_graph_warmup="
            f"captured:{decode_graph_warmup.captured}, skipped:{decode_graph_warmup.skipped}"
        )
    print()

    if args.timing:
        print_timing_summary(timings)

    for result in results:
        print(
            f"{result.request_id}: finish={result.finish_reason}, "
            f"generated_tokens={len(result.generated_ids)}, "
            f"prefix_hit_tokens={result.prefix_cache_hits}, "
            f"scheduler_steps={result.scheduler_steps}"
        )
        print("  output:", tokenizer.decode(result.generated_ids))
        print()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        generated_lengths = [len(result.generated_ids) for result in results]
        total_generated = sum(generated_lengths)
        output_data = {
            "runner": settings.version_name,
            "version_topic": settings.version_topic,
            "model_path": args.model_path,
            "workload_path": str(args.workload_path),
            "attention_backend": engine_config.attention_backend,
            "wall_s": wall_s,
            "generated_tokens": total_generated,
            "throughput_toks_per_s": (total_generated / wall_s) if wall_s > 0 else 0.0,
            "mean_gen_len": statistics.mean(generated_lengths) if generated_lengths else 0.0,
            "decode_graph_warmup": {
                "captured": decode_graph_warmup.captured,
                "skipped": decode_graph_warmup.skipped,
            },
            "settings": {
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "block_size": args.block_size,
                "num_blocks": args.num_blocks,
                "max_batch_tokens": args.max_batch_tokens,
                "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
                "max_decode_batch_size": args.max_decode_batch_size,
                "ignore_arrival_steps": args.ignore_arrival_steps,
                "ignore_eos": args.ignore_eos,
                "enable_prefix_cache": args.enable_prefix_cache,
                "enable_torch_compile_model_body": getattr(engine_config, "enable_torch_compile_model_body", False),
                "torch_compile_scope": getattr(engine_config, "torch_compile_scope", "none"),
                "enable_gpu_decode_state": getattr(engine_config, "enable_gpu_decode_state", False),
                "enable_triton_decode_metadata_kernel": getattr(engine_config, "enable_triton_decode_metadata_kernel", False),
                "enable_async_output_processing": getattr(engine_config, "enable_async_output_processing", False),
                "enable_decode_cuda_graphs": getattr(engine_config, "enable_decode_cuda_graphs", False),
                "unsafe_decode_cuda_graphs": getattr(engine_config, "unsafe_decode_cuda_graphs", False),
            },
            "timings": [
                {
                    "section": section,
                    "total_s": total_s,
                    "count": count,
                    "avg_ms": (total_s / count) * 1000.0 if count > 0 else 0.0,
                }
                for section, total_s, count in timings
            ],
            "results": [
                {
                    "request_id": result.request_id,
                    "prompt_tokens": result.prompt_tokens,
                    "generated_ids": result.generated_ids,
                    "finish_reason": result.finish_reason,
                    "prefix_cache_hits": result.prefix_cache_hits,
                    "scheduler_steps": result.scheduler_steps,
                    "text": tokenizer.decode(result.generated_ids),
                }
                for result in results
            ],
        }
        args.output_json.write_text(json.dumps(output_data, indent=2))
