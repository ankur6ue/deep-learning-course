from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ray


def load_prompt_sets(path: str) -> List[Dict[str, Any]]:
    sets: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sets.append(json.loads(line))
    return sets


def materialize_prompts(prompt_set: Dict[str, Any]) -> List[str]:
    prefix = prompt_set["common_prefix"]
    return [prefix + s for s in prompt_set["suffixes"]]


def summarize(latencies: List[float]) -> Dict[str, float]:
    return {
        "n": float(len(latencies)),
        "min_s": min(latencies),
        "median_s": statistics.median(latencies),
        "mean_s": statistics.mean(latencies),
        "max_s": max(latencies),
    }


@ray.remote(num_gpus=1, num_cpus=2)
class VLLMEngineActor:
    def __init__(
        self,
        model: str,
        enable_prefix_caching: bool,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        prefix_caching_hash_algo: Optional[str] = None,
    ):
        self.model = model
        self.enable_prefix_caching = enable_prefix_caching
        self.cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")

        from vllm import LLM

        llm_kwargs: Dict[str, Any] = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=enable_prefix_caching,
        )
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if prefix_caching_hash_algo is not None:
            llm_kwargs["prefix_caching_hash_algo"] = prefix_caching_hash_algo

        self.llm = LLM(**llm_kwargs)

    def info(self) -> Dict[str, Any]:
        info = {
            "model": self.model,
            "enable_prefix_caching": self.enable_prefix_caching,
            "CUDA_VISIBLE_DEVICES": self.cuda_visible,
            "pid": os.getpid(),
        }
        try:
            import torch
            d = torch.cuda.current_device()
            info["torch_device_name"] = torch.cuda.get_device_name(d)
            info["torch_total_mem_gb"] = float(torch.cuda.get_device_properties(d).total_memory / (1024**3))
        except Exception as e:
            info["torch_error"] = str(e)
        return info

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> float:
        """Returns latency_seconds for this batch."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        t0 = time.perf_counter()
        _ = self.llm.generate(prompts, params)
        t1 = time.perf_counter()
        return t1 - t0

    def warmup(self, prompts: List[str], max_tokens: int = 32) -> None:
        # Warm-up to avoid measuring init / compilation effects.
        _ = self.generate_batch(prompts, max_tokens=max_tokens, temperature=0.0, top_p=1.0)


@dataclass
class BenchConfig:
    prompt_sets_path: str = "data/prompt_sets.jsonl"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # generation settings (keep fixed across experiments)
    max_tokens: int = 4
    temperature: float = 0.2
    top_p: float = 0.95

    # engine memory knobs
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048

    # experiment structure
    warmup_prompts: int = 8
    warmup_iters: int = 2
    prime_iters: int = 1          # run once to populate prefix cache for that set
    measure_iters: int = 4        # measure steady-state after priming


def make_warmup_prompts(n: int) -> List[str]:
    # Simple prompts that don't share a big prefix; just to warm the engine.
    return [f"Say 'warmup' and count to {i}.\nAnswer:" for i in range(1, n + 1)]


def run_condition(enable_prefix_caching: bool, prompt_sets: List[Dict[str, Any]], cfg: BenchConfig) -> List[Dict[str, Any]]:
    actor = VLLMEngineActor.remote(
        model=cfg.model,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=1,
    )

    print("\n=== Engine Info ===")
    print(ray.get(actor.info.remote()))

    # Global warm-up (not part of measurements)
    warm_prompts = make_warmup_prompts(cfg.warmup_prompts)
    for _ in range(cfg.warmup_iters):
        ray.get(actor.generate_batch.remote(warm_prompts, max_tokens=32, temperature=0.0, top_p=1.0))

    results: List[Dict[str, Any]] = []
    for ps in prompt_sets:
        name = ps["name"]
        prompts = materialize_prompts(ps)

        # Prime: run once to populate prefix cache for this set (if enabled)
        for _ in range(cfg.prime_iters):
            ray.get(actor.generate_batch.remote(prompts, cfg.max_tokens, cfg.temperature, cfg.top_p))

        # Measure steady-state (repeat the exact same batch)
        lats: List[float] = []
        for _ in range(cfg.measure_iters):
            lat = ray.get(actor.generate_batch.remote(prompts, cfg.max_tokens, cfg.temperature, cfg.top_p))
            lats.append(lat)

        results.append(
            {
                "set": name,
                "enable_prefix_caching": enable_prefix_caching,
                "batch_size": len(prompts),
                "latencies_s": lats,
                "summary": summarize(lats),
            }
        )

    ray.kill(actor)
    return results


def main():
    cfg = BenchConfig()
    import vllm
    print(vllm.__version__)

    prompt_sets = load_prompt_sets(cfg.prompt_sets_path)

    print("Loaded prompt sets:", [ps["name"] for ps in prompt_sets])

    ray.init(num_gpus=2)

    print("\n########### PREFIX CACHING OFF ###########")
    off_results = run_condition(False, prompt_sets, cfg)

    print("\n########### PREFIX CACHING ON  ###########")
    on_results = run_condition(True, prompt_sets, cfg)

    # Compare medians
    by_key: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    for r in off_results + on_results:
        by_key[(r["set"], r["enable_prefix_caching"])] = r

    print("\n\n================== STEADY-STATE RESULTS (median over repeats) ==================")
    for ps in prompt_sets:
        name = ps["name"]
        off = by_key[(name, False)]["summary"]["median_s"]
        on = by_key[(name, True)]["summary"]["median_s"]
        print(f"\nSet: {name}")
        print(f"  OFF median: {off:.4f}s")
        print(f"   ON median: {on:.4f}s")
        print(f"  Improvement (OFF - ON): {off - on:+.4f}s  ({(off/on - 1.0)*100:+.1f}% faster when ON)")

    ray.shutdown()


if __name__ == "__main__":
    main()
