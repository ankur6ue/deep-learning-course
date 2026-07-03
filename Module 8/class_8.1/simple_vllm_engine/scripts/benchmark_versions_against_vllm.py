#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/home/ankur/dev/models/Llama-3.1-8B-Instruct"
DEFAULT_WORKLOAD_PATH = ROOT / "workloads" / "shared_chat_workload.json"
DEFAULT_RESULTS_DIR = ROOT / "benchmark_results"
REAL_MODEL_VERSIONS = [f"v{i}" for i in range(2, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark v2-v10 against the previous version and production vLLM."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--workload-path", type=Path, default=DEFAULT_WORKLOAD_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--versions", nargs="*", default=REAL_MODEL_VERSIONS)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--max-batch-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=64)
    parser.add_argument("--max-decode-batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Optional toy-engine attention backend override, e.g. flash_attn_paged or triton_paged.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--honor-arrival-steps",
        action="store_true",
        help=(
            "Use workload arrival_step values for toy-engine runs. The default "
            "sets every toy request arrival to 0 because the vLLM baseline "
            "submits the workload as one offline batch."
        ),
    )
    parser.add_argument("--child-version", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Disable vLLM graph capture. Default keeps production vLLM graph behavior.",
    )
    parser.add_argument(
        "--vllm-tokenizer-mode",
        type=str,
        default="auto",
        help="Forwarded to vLLM. Use 'hf' for local text-only Mistral compatibility checkpoints.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Forwarded to vLLM. Raising this can be necessary for larger local checkpoints.",
    )
    parser.add_argument(
        "--vllm-num-runs",
        type=int,
        default=1,
        help="Number of measured vLLM runs. The summary uses the fastest run.",
    )
    parser.add_argument(
        "--unsafe-decode-cuda-graphs",
        action="store_true",
        help=(
            "Enable toy-engine decode CUDA graph replay and skip replay repair/"
            "validation. Benchmark-only; can change generated tokens."
        ),
    )
    parser.add_argument(
        "--enable-decode-cuda-graphs",
        action="store_true",
        help=(
            "Enable toy-engine decode CUDA graph replay with capture-time "
            "validation when supported by the selected attention backend."
        ),
    )
    parser.add_argument(
        "--torch-compile-scope",
        type=str,
        default=None,
        choices=["mlp", "input_qkv", "tail", "all"],
        help="Optional toy-engine torch.compile scope override for versions that expose it.",
    )
    parser.add_argument(
        "--disable-toy-torch-compile",
        action="store_true",
        help="Disable the toy engine's model-body torch.compile path when the checkpoint is too memory-constrained.",
    )
    return parser.parse_args()


def load_workload(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{path} must contain a non-empty 'requests' list")
    return requests


def _filtered_dataclass_init(cls: type, values: dict[str, Any]) -> Any:
    public_fields = {field.name for field in dataclasses.fields(cls)}
    return cls(**{name: value for name, value in values.items() if name in public_fields})


def run_child(args: argparse.Namespace) -> None:
    version = args.child_version
    if version is None:
        raise ValueError("child run requires --child-version")
    version_dir = ROOT / version
    sys.path.insert(0, str(version_dir))
    sys.path.insert(1, str(ROOT))

    from common.hf_loader import build_engine_from_pretrained
    from common.tokenizer import HFTokenizer
    from simple_vllm_engine.config import EngineConfig
    from simple_vllm_engine.engine import timed_run
    from simple_vllm_engine.requests import RequestSpec

    tokenizer = HFTokenizer.from_pretrained(args.model_path)
    specs = []
    for idx, item in enumerate(load_workload(args.workload_path)):
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"request {idx} is missing messages")
        specs.append(
            RequestSpec(
                request_id=str(item.get("request_id", f"req-{idx}")),
                prompt_ids=tokenizer.encode_chat(messages),
                arrival_step=int(item.get("arrival_step", 0)) if args.honor_arrival_steps else 0,
                max_new_tokens=args.max_new_tokens,
            )
        )

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    config_values = {
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "max_batch_tokens": args.max_batch_tokens,
        "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
        "max_decode_batch_size": args.max_decode_batch_size,
        "max_model_len": args.max_model_len,
        "enable_prefix_cache": False,
        "device": args.device,
        "dtype": dtype,
        "enable_timing": False,
        "enable_decode_cuda_graphs": (
            args.enable_decode_cuda_graphs or args.unsafe_decode_cuda_graphs
        ),
        "unsafe_decode_cuda_graphs": args.unsafe_decode_cuda_graphs,
        "ignore_eos": True,
        # Older versions do not expose ignore_eos, so force EOS out of range.
        "eos_token_id": -999,
        "pad_token_id": 0,
    }
    if args.torch_compile_scope is not None:
        config_values["torch_compile_scope"] = args.torch_compile_scope
    if args.disable_toy_torch_compile:
        config_values["enable_torch_compile_model_body"] = False
    if args.attention_backend is not None:
        config_values["attention_backend"] = args.attention_backend
    config = _filtered_dataclass_init(EngineConfig, config_values)

    def run_once() -> tuple[float, list[Any], Any]:
        engine, loaded_tokenizer = build_engine_from_pretrained(args.model_path, config)
        if hasattr(engine, "warmup_decode_cuda_graphs"):
            engine.warmup_decode_cuda_graphs(specs)
        run = timed_run(engine, specs)
        del engine
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return run.wall_time_s, run.results, loaded_tokenizer

    if (
        getattr(config, "enable_torch_compile_model_body", False)
        and getattr(config, "prewarm_torch_compile", False)
    ):
        # Keep one-time compiler cost out of the measured steady-state run.
        run_once()

    wall_s, results, loaded_tokenizer = run_once()
    generated_lengths = [len(result.generated_ids) for result in results]
    total_generated = sum(generated_lengths)
    output = {
        "runner": version,
        "model_path": args.model_path,
        "workload_path": str(args.workload_path),
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "max_batch_tokens": args.max_batch_tokens,
            "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
            "max_decode_batch_size": args.max_decode_batch_size,
            "max_model_len": args.max_model_len,
            "honor_arrival_steps": args.honor_arrival_steps,
            "device": args.device,
            "dtype": str(dtype).replace("torch.", ""),
            "attention_backend": getattr(config, "attention_backend", None),
            "enable_torch_compile_model_body": getattr(config, "enable_torch_compile_model_body", False),
            "torch_compile_scope": getattr(config, "torch_compile_scope", None),
            "enable_gpu_decode_state": getattr(config, "enable_gpu_decode_state", False),
            "enable_async_output_processing": getattr(config, "enable_async_output_processing", False),
            "enable_decode_cuda_graphs": getattr(config, "enable_decode_cuda_graphs", False),
            "unsafe_decode_cuda_graphs": getattr(config, "unsafe_decode_cuda_graphs", False),
        },
        "wall_s": wall_s,
        "generated_tokens": total_generated,
        "throughput_toks_per_s": total_generated / wall_s if wall_s > 0 else 0.0,
        "results": [
            {
                "request_id": result.request_id,
                "prompt_tokens": result.prompt_tokens,
                "generated_ids": result.generated_ids,
                "finish_reason": result.finish_reason,
                "scheduler_steps": result.scheduler_steps,
                "text": loaded_tokenizer.decode(result.generated_ids),
            }
            for result in results
        ],
    }
    if args.output_json is None:
        print(json.dumps(output, indent=2))
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2))


def result_by_id(data: dict[str, Any]) -> dict[str, list[int]]:
    return {item["request_id"]: list(item["generated_ids"]) for item in data["results"]}


def ids_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return result_by_id(a) == result_by_id(b)


def summarize(results_dir: Path, versions: list[str], include_vllm: bool) -> str:
    loaded = {}
    for version in versions:
        path = results_dir / f"{version}.json"
        if path.exists():
            loaded[version] = json.loads(path.read_text())
    vllm = None
    if include_vllm and (results_dir / "vllm.json").exists():
        vllm = json.loads((results_dir / "vllm.json").read_text())
        # Existing vLLM runner stores run summaries under `runs`.
        if "wall_s" not in vllm:
            best_run = min(vllm["runs"], key=lambda item: item["wall_s"])
            vllm["wall_s"] = best_run["wall_s"]
            vllm["generated_tokens"] = best_run["generated_tokens"]
            vllm["throughput_toks_per_s"] = best_run["throughput_toks_per_s"]

    rows = []
    for version in versions:
        data = loaded.get(version)
        if data is None:
            continue
        idx = int(version[1:])
        prev_key = f"v{idx - 1}"
        prev = loaded.get(prev_key)
        prev_match = "n/a" if prev is None else ("yes" if ids_match(data, prev) else "no")
        vllm_match = "n/a" if vllm is None else ("yes" if ids_match(data, vllm) else "no")
        improvement = "n/a"
        if prev is not None:
            improvement_value = (prev["wall_s"] - data["wall_s"]) / prev["wall_s"] * 100.0
            improvement = f"{improvement_value:+.1f}%"
        gap = "n/a"
        if vllm is not None:
            gap_value = (data["wall_s"] / vllm["wall_s"] - 1.0) * 100.0
            gap = f"{gap_value:+.1f}%"
        rows.append(
            {
                "version": version,
                "wall_s": data["wall_s"],
                "throughput": data["throughput_toks_per_s"],
                "prev_match": prev_match,
                "vllm_match": vllm_match,
                "improvement": improvement,
                "gap": gap,
            }
        )

    lines = [
        "| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Wall-time improvement vs previous | Remaining wall-time gap vs vLLM |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['version']} | {row['wall_s']:.3f} | {row['throughput']:.2f} | "
            f"{row['prev_match']} | {row['vllm_match']} | {row['improvement']} | {row['gap']} |"
        )
    if vllm is not None:
        lines.extend(
            [
                "",
                f"vLLM baseline wall time: {vllm['wall_s']:.3f}s, throughput: {vllm['throughput_toks_per_s']:.2f} tok/s.",
            ]
        )
    return "\n".join(lines)


def run_parent(args: argparse.Namespace) -> None:
    args.results_dir = args.results_dir.resolve()
    args.workload_path = args.workload_path.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for version in args.versions:
        out = args.results_dir / f"{version}.json"
        print(f"=== running {version} ===", flush=True)
        cmd = [
            sys.executable,
            __file__,
            "--child-version",
            version,
            "--model-path",
            args.model_path,
            "--workload-path",
            str(args.workload_path),
            "--results-dir",
            str(args.results_dir),
            "--output-json",
            str(out),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--block-size",
            str(args.block_size),
            "--num-blocks",
            str(args.num_blocks),
            "--max-batch-tokens",
            str(args.max_batch_tokens),
            "--max-prefill-chunk-tokens",
            str(args.max_prefill_chunk_tokens),
            "--max-decode-batch-size",
            str(args.max_decode_batch_size),
            "--max-model-len",
            str(args.max_model_len),
            "--device",
            args.device,
        ]
        if args.attention_backend is not None:
            cmd.extend(["--attention-backend", args.attention_backend])
        if args.honor_arrival_steps:
            cmd.append("--honor-arrival-steps")
        if args.enable_decode_cuda_graphs:
            cmd.append("--enable-decode-cuda-graphs")
        if args.torch_compile_scope is not None:
            cmd.extend(["--torch-compile-scope", args.torch_compile_scope])
        if args.disable_toy_torch_compile:
            cmd.append("--disable-toy-torch-compile")
        subprocess.run(cmd, cwd=ROOT, check=True)

    if not args.skip_vllm:
        print("=== running vLLM baseline ===", flush=True)
        vllm_out = args.results_dir / "vllm.json"
        cmd = [
            sys.executable,
            str(ROOT / "vllm_baseline" / "run_vllm_real.py"),
            "--model-path",
            args.model_path,
            "--workload-path",
            str(args.workload_path),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--tokenizer-mode",
            args.vllm_tokenizer_mode,
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-batched-tokens",
            str(args.max_batch_tokens),
            "--max-num-seqs",
            str(args.max_decode_batch_size),
            "--block-size",
            str(args.block_size),
            "--gpu-memory-utilization",
            str(args.vllm_gpu_memory_utilization),
            "--num-runs",
            str(args.vllm_num_runs),
            "--disable-prefix-caching",
            "--output-json",
            str(vllm_out),
        ]
        if args.vllm_enforce_eager:
            cmd.append("--enforce-eager")
        subprocess.run(cmd, cwd=ROOT, check=True)

    table = summarize(args.results_dir, args.versions, include_vllm=not args.skip_vllm)
    summary_path = args.results_dir / "summary.md"
    summary_path.write_text(table + "\n")
    print("\n" + table)
    print(f"\nsummary written to {summary_path}")


def main() -> None:
    args = parse_args()
    if args.child_version is not None:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
