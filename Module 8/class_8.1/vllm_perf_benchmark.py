#!/usr/bin/env python3
"""
Simple vLLM optimization harness for Llama 8B Instruct (or similar HF models).

- Reads .txt files from an input directory.
- Runs summarization with different vLLM configs.
- Writes summaries + per-run timing/stats.

Run e.g.:

  python vllm_summarization_harness.py \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --input-dir ./inputs \
      --output-dir ./outputs

Then inspect the subdirs under ./outputs.
"""

import argparse
import json
import os
import time
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from vllm import LLM, SamplingParams

@dataclass
class RunConfig:
    """Configuration for a single vLLM run."""
    name: str

    # Parallelism
    tensor_parallel_size: int = 1

    # Precision / KV cache / quantization
    dtype: str = "float16"           # "float16", "bfloat16", "float32"
    kv_cache_dtype: str = "auto"     # "auto", "fp8", etc. (depends on vLLM version)
    quantization: Optional[str] = None   # e.g. "awq", "gptq", "marlin", "int4", None

    # Engine behavior
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="/home/ankur/dev/models/Llama-3.1-8B-Instruct-quant-nvfp4-NVFP4-calib256-msl2048",
        help="HF model ID, e.g. meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        help="Directory containing .txt files to summarize.",
        default="./benchmark_data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_data",
        required=False,
        help="Directory to write summaries and stats.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for summaries.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling.",
    )
    return parser.parse_args()


def build_default_run_grid() -> List[RunConfig]:
    """Configs from simple → more aggressive optimizations."""
    grid1: List[RunConfig] = []
    grid: List[RunConfig] = []
    # 1. Baseline, full precision-ish (fp16, no quantization)
    grid1.append(
        RunConfig(
            name="baseline_fp16_tp1",
            tensor_parallel_size=2,
            # dtype="bfloat16",
            kv_cache_dtype="fp8",
            quantization=None,
            enforce_eager=False,
            gpu_memory_utilization=0.7,
            max_model_len=2096,
        )
    )

    # 2. Tensor parallel on 2 GPUs
    grid.append(
        RunConfig(
            name="fp16_tp2",
            tensor_parallel_size=2,
            dtype="float16",
            kv_cache_dtype="auto",
            quantization=None,
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
    )

    # 3. bfloat16 (if GPUs support it)
    grid.append(
        RunConfig(
            name="bf16_tp2",
            tensor_parallel_size=2,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            quantization=None,
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
    )

    # 4. KV cache in lower precision (if your vLLM build supports it)
    grid.append(
        RunConfig(
            name="bf16_tp2_kv_fp8",
            tensor_parallel_size=2,
            dtype="bfloat16",
            kv_cache_dtype="fp8",   # or "fp8_e5m2" depending on vLLM version
            quantization=None,
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
    )

    # 5. Quantized weights – AWQ / GPTQ / INT4 etc. (you need matching weights)
    grid.append(
        RunConfig(
            name="int4_tp2",
            tensor_parallel_size=2,
            dtype="float16",        # internal compute dtype
            kv_cache_dtype="auto",
            quantization="int4",    # change to "awq", "gptq", "marlin", ...
            enforce_eager=False,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
    )

    # For now, just return the first list (baseline) as in your original code.
    return grid1


def to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype_str}")


def get_system_vram() -> Dict[str, Any]:
    """
    Query nvidia-smi for per-GPU VRAM usage.

    Returns:
        {"gpus": [{"gpu_index": int, "memory_used_mb": int, "memory_total_mb": int}, ...]}
        or {"error": "..."} on failure.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except Exception as e:
        return {"error": f"nvidia-smi failed: {e}"}

    gpus = []
    for idx, line in enumerate(result.stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            used_str, total_str = [x.strip() for x in line.split(",")]
            used = int(used_str)
            total = int(total_str)
            gpus.append(
                {
                    "gpu_index": idx,
                    "memory_used_mb": used,
                    "memory_total_mb": total,
                }
            )
        except Exception:
            # skip malformed lines
            continue

    return {"gpus": gpus}


def get_torch_vram_per_gpu() -> Dict[str, Any]:
    """
    Collect per-GPU allocated/reserved memory from torch.cuda.*.

    Returns:
        {"gpus": [{"gpu_index": int, "allocated_mb": int, "reserved_mb": int}, ...]}
        or {"error": "..."} if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available in torch"}

    gpus = []
    torch.cuda.synchronize()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            alloc = int(torch.cuda.memory_allocated() // (1024 ** 2))
            reserved = int(torch.cuda.memory_reserved() // (1024 ** 2))
        gpus.append(
            {
                "gpu_index": i,
                "allocated_mb": alloc,
                "reserved_mb": reserved,
            }
        )
    return {"gpus": gpus}


def snapshot_vram(label: str, llm: Optional[LLM] = None) -> Dict[str, Any]:
    """
    Take a VRAM snapshot combining:
      - nvidia-smi
      - torch CUDA stats
      - vLLM llm.get_memory_stats() (if available)

    This is stored into run_stats.json for offline comparison.
    """
    snap: Dict[str, Any] = {
        "label": label,
        "time_sec": time.time(),
        "system_vram": get_system_vram(),
        "torch_vram": get_torch_vram_per_gpu(),
    }

    if llm is not None and hasattr(llm, "get_memory_stats"):
        try:
            snap["vllm_memory_stats"] = llm.get_memory_stats()
        except Exception as e:
            snap["vllm_memory_stats_error"] = str(e)

    # Also print a short human-readable summary
    print(f"\n[VRAM] Snapshot: {label}")
    if isinstance(snap["system_vram"], dict) and "gpus" in snap["system_vram"]:
        for gpu in snap["system_vram"]["gpus"]:
            print(
                f"  [nvidia-smi] GPU{gpu['gpu_index']}: "
                f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB used"
            )
    if isinstance(snap["torch_vram"], dict) and "gpus" in snap["torch_vram"]:
        for gpu in snap["torch_vram"]["gpus"]:
            print(
                f"  [torch] GPU{gpu['gpu_index']}: "
                f"alloc={gpu['allocated_mb']} MB, "
                f"reserved={gpu['reserved_mb']} MB"
            )
    if "vllm_memory_stats" in snap:
        print(f"  [vLLM] {snap['vllm_memory_stats']}")

    return snap


def make_llm(model_id: str, cfg: RunConfig) -> LLM:
    """Instantiate vLLM LLM() with supported arguments only."""
    torch_dtype = to_torch_dtype(cfg.dtype)

    kwargs: Dict[str, Any] = dict(
        model=model_id,
        tensor_parallel_size=cfg.tensor_parallel_size,
        dtype=torch_dtype,
        kv_cache_dtype=cfg.kv_cache_dtype,
        enforce_eager=cfg.enforce_eager,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
    )

    if cfg.quantization is not None:
        kwargs["quantization"] = cfg.quantization

    print(f"\n[INFO] Creating LLM with config: {cfg.name}")
    print(f"       kwargs={kwargs}")
    return LLM(**kwargs)


def read_input_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.txt") if p.is_file()])


def build_prompt(text: str) -> str:
    # Very simple prompt; tweak as you like.
    return (
        "You are a helpful assistant. Summarize the following text in 3–5 bullet points, "
        "focusing on the main ideas and avoiding minor details.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Summary:\n"
    )


def run_one_config(
    model_id: str,
    cfg: RunConfig,
    input_files: List[Path],
    output_base: Path,
    sampling_params: SamplingParams,
):
    out_dir = output_base / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stats container
    stats: Dict[str, Any] = {
        "config": asdict(cfg),
        "files": [],
        "total_wall_time_sec": None,
        "avg_wall_time_per_file_sec": None,
        "vram_snapshots": [],
    }

    # Snapshot before model init
    stats["vram_snapshots"].append(snapshot_vram("before_model_init"))

    # Create LLM
    llm = make_llm(model_id, cfg)

    # Snapshot after model init
    stats["vram_snapshots"].append(snapshot_vram("after_model_init", llm))

    t0_total = time.time()
    for path in input_files:
        with path.open("r", encoding="utf-8") as f:
            text = f.read()

        prompt = build_prompt(text)

        t0 = time.time()
        outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
        dt = time.time() - t0

        generated_text = outputs[0].outputs[0].text

        # Save summary
        out_txt = out_dir / f"{path.stem}_summary.txt"
        with out_txt.open("w", encoding="utf-8") as f:
            f.write(generated_text)

        stats["files"].append(
            {
                "input_file": str(path),
                "output_file": str(out_txt),
                "wall_time_sec": dt,
                # Token-level stats could be added here later.
            }
        )

        print(f"[{cfg.name}] {path.name}: {dt:.3f}s")

    t_total = time.time() - t0_total
    stats["total_wall_time_sec"] = t_total
    stats["avg_wall_time_per_file_sec"] = (
        t_total / len(input_files) if input_files else None
    )

    # Snapshot after all generations
    stats["vram_snapshots"].append(snapshot_vram("after_all_generations", llm))

    # Save stats as JSON
    stats_path = out_dir / "run_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"[{cfg.name}] Done. total={t_total:.3f}s, "
        f"avg/file={stats['avg_wall_time_per_file_sec']:.3f}s"
    )


def main():
    args = parse_args()

    model_id = args.model
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = read_input_files(input_dir)
    if not input_files:
        raise SystemExit(f"No .txt files found in {input_dir}")

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    run_grid = build_default_run_grid()

    for cfg in run_grid:
        run_one_config(
            model_id=model_id,
            cfg=cfg,
            input_files=input_files,
            output_base=output_dir,
            sampling_params=sampling_params,
        )


if __name__ == "__main__":
    main()
