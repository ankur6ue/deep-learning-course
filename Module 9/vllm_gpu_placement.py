"""
Two vLLM Ray actors: one per GPU, benchmark throughput/latency side-by-side.

Goal:
- Spin up 2 vLLM engines (one pinned to each GPU via a placement group bundle).
- Run the same prompts on both engines.
- Measure per-engine latency and tokens/sec to see if RTX 5090 is faster than RTX 5080.

Run:
  python ray_vllm_two_gpu_bench.py

Notes:
- Bundle index 0/1 usually maps to physical GPU0/GPU1 on a single machine.
  If you want to be 100% sure which is 5090 vs 5080, we also query torch device name.
- This benchmark is simple and should be good enough for comparing relative performance.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Dict, Any

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_gpus=1, num_cpus=1)
class VLLMActor:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = 2048,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")

        from vllm import LLM

        llm_kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True

        self.llm = LLM(**llm_kwargs)

    def device_info(self) -> Dict[str, Any]:
        """Return what GPU this actor sees + a human-readable device name."""
        info = {
            "CUDA_VISIBLE_DEVICES": self.cuda_visible,
            "pid": os.getpid(),
            "model": self.model_name,
        }
        try:
            import torch

            d = torch.cuda.current_device()
            info.update(
                {
                    "torch_current_device": int(d),
                    "torch_device_name": torch.cuda.get_device_name(d),
                    "mem_total_gb": float(torch.cuda.get_device_properties(d).total_memory / (1024**3)),
                }
            )
        except Exception as e:
            info["torch_error"] = str(e)
        return info

    def warmup(self, prompt: str = "Hello", max_tokens: int = 16) -> str:
        """One warmup request to trigger any lazy init/compilation."""
        from vllm import SamplingParams

        out = self.llm.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.0))
        return out[0].outputs[0].text if out and out[0].outputs else ""

    def predict(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [(o.outputs[0].text if o.outputs else "") for o in outputs]

    def bench(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.2,
        runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Benchmark end-to-end generate() time inside the actor.
        Returns per-run latencies and a rough tokens/sec estimate.

        tokens/sec here is approximate: we use output text length as a proxy when tokenizer is not exposed.
        If you want exact token counts, we can also load the tokenizer in the actor and count tokens precisely.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latencies = []
        out_char_counts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            outputs = self.llm.generate(prompts, sampling_params)
            t1 = time.perf_counter()
            lat = t1 - t0
            latencies.append(lat)

            # proxy for work done: sum of output characters
            chars = 0
            for o in outputs:
                if o.outputs:
                    chars += len(o.outputs[0].text)
            out_char_counts.append(chars)

        # Rough throughput proxy: chars/sec (useful for relative comparison)
        char_per_sec = []
        for lat, chars in zip(latencies, out_char_counts):
            char_per_sec.append(chars / lat if lat > 0 else 0.0)

        return {
            "latencies_sec": latencies,
            "out_chars": out_char_counts,
            "chars_per_sec": char_per_sec,
            "summary": {
                "runs": runs,
                "median_latency_sec": sorted(latencies)[runs // 2],
                "avg_latency_sec": sum(latencies) / runs,
                "median_chars_per_sec": sorted(char_per_sec)[runs // 2],
                "avg_chars_per_sec": sum(char_per_sec) / runs,
                "batch_size": len(prompts),
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }


def main():
    ray.init(num_gpus=2)

    # Placement group: one bundle per GPU
    pg = placement_group([{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}], strategy="PACK", name="pg_two_gpus")
    ray.get(pg.ready())

    sched0 = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    sched1 = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=1)

    model = "Qwen/Qwen2.5-1.5B-Instruct"

    # Create one engine per GPU
    a0 = VLLMActor.options(scheduling_strategy=sched0).remote(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    a1 = VLLMActor.options(scheduling_strategy=sched1).remote(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )

    info0, info1 = ray.get([a0.device_info.remote(), a1.device_info.remote()])
    print("\n=== Actor 0 device ===")
    print(info0)
    print("\n=== Actor 1 device ===")
    print(info1)

    # Warmup both engines (important for fair timing)
    ray.get([a0.warmup.remote(), a1.warmup.remote()])

    # Prompts: pick something GSM8K-ish + a couple varied prompts
    prompts = [
        "Solve: If John has 3 apples and buys 4 more, how many apples does he have?\nAnswer:",
        "Solve: A store sold 12 packs of pencils. Each pack has 8 pencils. How many pencils were sold?\nAnswer:",
        "Explain in 2 sentences what reinforcement learning is.\nAnswer:",
        "Write a short, one-paragraph summary of New York City.\nAnswer:",
    ]

    runs = 3
    max_tokens = 128
    temperature = 0.2

    # Run benchmark on both GPUs
    b0_ref = a0.bench.remote(prompts, max_tokens=max_tokens, temperature=temperature, runs=runs)
    b1_ref = a1.bench.remote(prompts, max_tokens=max_tokens, temperature=temperature, runs=runs)
    b0, b1 = ray.get([b0_ref, b1_ref])

    print("\n=== Benchmark Actor 0 ===")
    print(b0["summary"])
    print("latencies:", [round(x, 4) for x in b0["latencies_sec"]])
    print("\n=== Benchmark Actor 1 ===")
    print(b1["summary"])
    print("latencies:", [round(x, 4) for x in b1["latencies_sec"]])

    # Decide which is likely the 5090 vs 5080 based on device name
    def dev_name(info: Dict[str, Any]) -> str:
        return str(info.get("torch_device_name", info.get("CUDA_VISIBLE_DEVICES", "")))

    name0, name1 = dev_name(info0), dev_name(info1)

    # Simple comparison
    med0 = b0["summary"]["median_latency_sec"]
    med1 = b1["summary"]["median_latency_sec"]
    faster = "Actor 0" if med0 < med1 else "Actor 1"
    print("\n=== Comparison ===")
    print(f"Actor 0 device: {name0}")
    print(f"Actor 1 device: {name1}")
    print(f"Median latency (lower is better): actor0={med0:.4f}s, actor1={med1:.4f}s -> {faster} is faster")

    ray.shutdown()


if __name__ == "__main__":
    main()
