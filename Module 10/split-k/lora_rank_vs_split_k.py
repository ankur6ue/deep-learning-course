"""Measure LoRA rank versus split-K with real vLLM inference.

vLLM uses split-K in the LoRA shrink projection:

    hidden [M, H] @ lora_A.T [H, rank] -> low_rank [M, rank]

During batched decode, vLLM flattens one token per request into M=B rows. A
small batch and rank produce few output tiles, so splitting the H reduction can
expose more parallel work. This experiment changes the split factor used by
vLLM's own Punica/Triton shrink kernel for both prefill and decode; it does not
provide a replacement matrix-multiplication kernel.

Each rank needs a fresh engine because vLLM allocates and computes with
``max_lora_rank``. The top-level process therefore starts one worker process per
rank and CUDA-stream mode. Every worker loads Mistral, generates with a
synthetic LoRA applied to all attention projections, and sweeps split-K while
the model remains resident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SUPPORTED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def stream_modes(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip() for part in value.split(",")))
    allowed = {"single", "dual"}
    if not values or any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError("expected single, dual, or single,dual")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument(
        "--tokenizer-mode",
        default="auto",
        choices=("auto", "hf", "slow", "mistral"),
    )
    parser.add_argument(
        "--ranks",
        type=comma_separated_ints,
        default=comma_separated_ints("8,32,128,512"),
        help="LoRA ranks. vLLM starts a fresh engine for each rank.",
    )
    parser.add_argument(
        "--split-k",
        type=comma_separated_ints,
        default=comma_separated_ints("1,2,4,8,16,32,64"),
        help="Split factors passed to vLLM's LoRA shrink kernel.",
    )
    parser.add_argument(
        "--stream-modes",
        type=stream_modes,
        default=stream_modes("single"),
        help="Use vLLM's normal stream or its experimental dual-stream LoRA path.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=2,
        help=(
            "Generated tokens. One measures prefill only; two adds one decode "
            "forward pass."
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow vLLM to keep multiple model steps in flight. Disabled by "
            "default so first-to-second-token timing includes the decode pass."
        ),
    )
    parser.add_argument(
        "--numeric-tokens",
        type=int,
        default=2,
        help="Greedy steps for raw-logit comparison; use 0 to disable it.",
    )
    parser.add_argument(
        "--numeric-repeats",
        type=int,
        default=2,
        help="Runs per split factor for measuring run-to-run atomic drift.",
    )
    parser.add_argument(
        "--numeric-stream-modes",
        type=stream_modes,
        default=stream_modes("single"),
        help="Stream modes that run full-vocabulary logit comparisons.",
    )
    parser.add_argument("--adapter-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--gpu", default="0", help="CUDA device exposed to each worker."
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--adapter-cache",
        type=Path,
        default=Path.home() / ".cache/deep-learning-course/split-k",
    )
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config is not None:
        return args
    unsupported = sorted(set(args.ranks) - set(SUPPORTED_LORA_RANKS))
    if unsupported:
        parser.error(
            f"vLLM supports max_lora_rank values {SUPPORTED_LORA_RANKS}; "
            f"unsupported: {unsupported}"
        )
    if 1 not in args.split_k:
        args.split_k = (1, *args.split_k)
    if args.batch_size < 1 or args.prompt_tokens < 2 or args.output_tokens < 1:
        parser.error("batch-size >= 1, prompt-tokens >= 2, and output-tokens >= 1")
    if args.warmups < 1 or args.trials < 1:
        parser.error("warmups and trials must be at least 1")
    if args.numeric_tokens < 0 or args.numeric_repeats < 1:
        parser.error("numeric-tokens must be nonnegative and numeric-repeats positive")
    if args.adapter_scale <= 0:
        parser.error("adapter-scale must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    return args


def model_cache_key(model: str, revision: str | None) -> str:
    name = model.rstrip("/").rsplit("/", 1)[-1]
    digest = hashlib.sha256(f"{model}@{revision}".encode()).hexdigest()[:10]
    return f"{name}-{digest}"


def adapter_directory(config: dict[str, Any]) -> Path:
    scale = str(config["adapter_scale"]).replace(".", "p")
    return (
        Path(config["adapter_cache"])
        / model_cache_key(config["model"], config.get("revision"))
        / f"rank-{config['rank']}-seed-{config['seed']}-scale-{scale}"
    )


def make_synthetic_adapter(config: dict[str, Any]) -> Path:
    """Create a deterministic PEFT-format Mistral adapter if it is not cached."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    output_dir = adapter_directory(config)
    metadata_path = output_dir / "benchmark_metadata.json"
    expected_metadata = {
        "model": config["model"],
        "revision": config.get("revision"),
        "rank": config["rank"],
        "seed": config["seed"],
        "adapter_scale": config["adapter_scale"],
        "target_modules": list(TARGET_MODULES),
    }
    if metadata_path.exists():
        actual_metadata = json.loads(metadata_path.read_text())
        if actual_metadata != expected_metadata:
            raise RuntimeError(f"cached adapter metadata does not match: {output_dir}")
        return output_dir
    if output_dir.exists():
        raise RuntimeError(f"incomplete adapter cache directory: {output_dir}")

    model_config = AutoConfig.from_pretrained(
        config["model"],
        revision=config.get("revision"),
        cache_dir=config.get("download_dir"),
    )
    if model_config.model_type != "mistral":
        raise ValueError(
            "synthetic adapter generation currently expects a Mistral model; "
            f"got model_type={model_config.model_type!r}"
        )

    hidden_size = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    rank = config["rank"]

    # Scaling A by 1/sqrt(input_size) and B by scale/sqrt(rank) keeps the
    # expected LoRA update magnitude roughly constant as rank changes.
    shapes = {
        "q_proj": (hidden_size, q_size),
        "k_proj": (hidden_size, kv_size),
        "v_proj": (hidden_size, kv_size),
        "o_proj": (q_size, hidden_size),
    }
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    weights: dict[str, torch.Tensor] = {}
    for layer in range(model_config.num_hidden_layers):
        for module_name, (input_size, output_size) in shapes.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module_name}"
            lora_a = torch.randn(rank, input_size, generator=generator)
            lora_a.mul_(1.0 / math.sqrt(input_size))
            lora_b = torch.randn(output_size, rank, generator=generator)
            lora_b.mul_(config["adapter_scale"] / math.sqrt(rank))
            weights[f"{prefix}.lora_A.weight"] = lora_a.to(torch.bfloat16)
            weights[f"{prefix}.lora_B.weight"] = lora_b.to(torch.bfloat16)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f"{output_dir.name}-", dir=output_dir.parent)
    )
    save_file(weights, temporary_dir / "adapter_model.safetensors")
    adapter_config = {
        "base_model_name_or_path": config["model"],
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(TARGET_MODULES),
        "task_type": "CAUSAL_LM",
    }
    (temporary_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2) + "\n"
    )
    (temporary_dir / "benchmark_metadata.json").write_text(
        json.dumps(expected_metadata, indent=2) + "\n"
    )
    temporary_dir.replace(output_dir)
    return output_dir


class SplitKController:
    """Override the split factor for every measured LoRA shrink invocation."""

    def __init__(self) -> None:
        from vllm.lora.ops.triton_ops import lora_shrink_op

        self._module = lora_shrink_op
        self._original = lora_shrink_op.get_lora_op_configs
        self.split_k = 1
        self.calls = 0
        self.shapes: set[tuple[int | None, ...]] = set()

        def forced_config(op_type: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            kernel_config = dict(self._original(op_type, *args, **kwargs))
            if op_type == "shrink":
                positional_names = (
                    "max_loras",
                    "batch",
                    "hidden_size",
                    "rank",
                    "num_slices",
                )
                arguments = dict(zip(positional_names, args))
                arguments.update(kwargs)
                # This hook executes for each eager shrink invocation, so one
                # run uses the same setting for prompt and generated tokens.
                kernel_config["split_k"] = self.split_k
                self.calls += 1
                shape = tuple(arguments.get(name) for name in positional_names[1:])
                self.shapes.add(shape)
            return kernel_config

        lora_shrink_op.get_lora_op_configs = forced_config

    def begin(self, split_k: int) -> None:
        self.split_k = split_k
        self.calls = 0
        self.shapes.clear()

    def verify(self, expected_rank: int) -> dict[str, Any]:
        if self.calls == 0:
            raise RuntimeError(
                "vLLM did not call the patched LoRA shrink configuration hook; "
                "its internal API may have changed"
            )
        ranks = {shape[2] for shape in self.shapes}
        if ranks != {expected_rank}:
            raise RuntimeError(
                f"vLLM computed with rank(s) {sorted(ranks)}, expected {expected_rank}"
            )
        return {
            "shrink_config_calls": self.calls,
            "shrink_shapes": [list(shape) for shape in sorted(self.shapes)],
        }


def fixed_token_prompts(
    *, batch_size: int, prompt_tokens: int, vocab_size: int, bos_token_id: int | None
) -> list[dict[str, list[int]]]:
    """Build exact-length prompts without tying the benchmark to prompt wording."""
    bos = 1 if bos_token_id is None else bos_token_id
    first_regular_token = 3
    regular_tokens = vocab_size - first_regular_token
    prompts = []
    for request_index in range(batch_size):
        ids = [bos]
        ids.extend(
            first_regular_token
            + ((position * 7919 + request_index * 104729) % regular_tokens)
            for position in range(prompt_tokens - 1)
        )
        prompts.append({"prompt_token_ids": ids})
    return prompts


def run_generation(
    llm: Any,
    prompts: Sequence[dict[str, list[int]]],
    sampling_params: Any,
    lora_request: Any,
    controller: SplitKController,
    split_k: int,
    rank: int,
) -> tuple[list[Any], float, dict[str, Any]]:
    import torch

    controller.begin(split_k)
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(
        list(prompts),
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    hook_data = controller.verify(rank)
    return outputs, elapsed, hook_data


def performance_sample(
    outputs: Sequence[Any], elapsed: float
) -> dict[str, float | None]:
    output_lengths = [len(item.outputs[0].token_ids) for item in outputs]
    total_tokens = sum(output_lengths)
    metrics = [item.metrics for item in outputs]
    if any(item is None for item in metrics):
        raise RuntimeError("vLLM did not return per-request timing metrics")

    decode_tokens = sum(max(length - 1, 0) for length in output_lengths)
    request_itls = [
        (item.last_token_ts - item.first_token_ts) * 1000.0 / (length - 1)
        for item, length in zip(metrics, output_lengths)
        if length > 1
    ]
    decode_seconds = (
        max(item.last_token_ts for item in metrics)
        - min(item.first_token_ts for item in metrics)
        if decode_tokens
        else 0.0
    )
    return {
        "wall_ms": elapsed * 1000.0,
        "wall_tokens_per_second": total_tokens / elapsed,
        # vLLM defines prefill time as scheduled -> first generated token.
        "prefill_ms": statistics.median(
            (item.first_token_ts - item.scheduled_ts) * 1000.0 for item in metrics
        ),
        "time_to_first_token_ms": statistics.median(
            item.first_token_latency * 1000.0 for item in metrics
        ),
        "decode_ms_per_token": (
            statistics.median(request_itls) if request_itls else None
        ),
        "decode_tokens_per_second": (
            decode_tokens / decode_seconds if decode_seconds > 0 else None
        ),
    }


def median_sample(
    samples: Sequence[dict[str, float | None]],
) -> dict[str, float | None]:
    medians: dict[str, float | None] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples if sample[key] is not None]
        medians[key] = statistics.median(values) if values else None
    return medians


def benchmark_performance(
    config: dict[str, Any],
    llm: Any,
    prompts: list[dict[str, list[int]]],
    lora_request: Any,
    controller: SplitKController,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    # With the default max_tokens=2, prefill produces token 1 and the only
    # decode forward produces token 2.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=config["output_tokens"],
        ignore_eos=True,
        detokenize=False,
        watermarking=False,
    )
    split_factors = config["split_k"]

    # The Triton kernel specializes on split-K. Warm each value before timing
    # so compilation and first-time adapter loading are outside the samples.
    for split_k in split_factors:
        for _ in range(config["warmups"]):
            run_generation(
                llm,
                prompts,
                params,
                lora_request,
                controller,
                split_k,
                config["rank"],
            )

    samples: dict[int, list[dict[str, float | None]]] = {
        split_k: [] for split_k in split_factors
    }
    hook_data: dict[int, dict[str, Any]] = {}
    randomizer = random.Random(config["seed"])
    for _ in range(config["trials"]):
        trial_order = list(split_factors)
        randomizer.shuffle(trial_order)
        for split_k in trial_order:
            outputs, elapsed, observed_hook_data = run_generation(
                llm,
                prompts,
                params,
                lora_request,
                controller,
                split_k,
                config["rank"],
            )
            samples[split_k].append(performance_sample(outputs, elapsed))
            hook_data[split_k] = observed_hook_data

    rows = []
    for split_k in split_factors:
        row: dict[str, Any] = {
            "rank": config["rank"],
            "stream_mode": config["stream_mode"],
            "split_k": split_k,
            **median_sample(samples[split_k]),
            **hook_data[split_k],
            "trial_samples": samples[split_k],
        }
        rows.append(row)

    baseline = next(row for row in rows if row["split_k"] == 1)
    for row in rows:
        row["wall_speedup_vs_k1"] = baseline["wall_ms"] / row["wall_ms"]
        row["prefill_speedup_vs_k1"] = (
            baseline["prefill_ms"] / row["prefill_ms"]
        )
        row["decode_speedup_vs_k1"] = (
            baseline["decode_ms_per_token"] / row["decode_ms_per_token"]
            if baseline["decode_ms_per_token"] is not None
            and row["decode_ms_per_token"] is not None
            else None
        )
    return rows


def flat_logit_rows(completion: Any) -> list[tuple[Any, Any]]:
    import numpy as np

    def sorted_unique(token_ids: Any, logits: Any) -> tuple[Any, Any]:
        order = np.argsort(token_ids)
        token_ids = token_ids[order]
        logits = logits[order]
        # vLLM may append the sampled token even when logprobs=-1 already
        # returned that token as part of the full vocabulary.
        keep = np.ones(len(token_ids), dtype=bool)
        keep[1:] = token_ids[1:] != token_ids[:-1]
        return token_ids[keep], logits[keep]

    logprobs = completion.logprobs
    if logprobs is None:
        raise RuntimeError("raw logits were requested but vLLM returned none")
    rows = []
    if all(hasattr(logprobs, name) for name in ("start_indices", "end_indices")):
        for start, end in zip(logprobs.start_indices, logprobs.end_indices):
            token_ids = np.asarray(logprobs.token_ids[start:end], dtype=np.int32)
            logits = np.asarray(logprobs.logprobs[start:end], dtype=np.float32)
            rows.append(sorted_unique(token_ids, logits))
        return rows

    for position in logprobs:
        token_ids = np.asarray(sorted(position), dtype=np.int32)
        logits = np.asarray(
            [position[token_id].logprob for token_id in token_ids], dtype=np.float32
        )
        rows.append((token_ids, logits))
    return rows


def capture_logits(
    config: dict[str, Any],
    llm: Any,
    prompt: dict[str, list[int]],
    lora_request: Any,
    controller: SplitKController,
    split_k: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=config["numeric_tokens"],
        ignore_eos=True,
        detokenize=False,
        logprobs=-1,
        flat_logprobs=True,
        watermarking=False,
    )
    outputs, _, _ = run_generation(
        llm,
        [prompt],
        params,
        lora_request,
        controller,
        split_k,
        config["rank"],
    )
    completion = outputs[0].outputs[0]
    return {
        "token_ids": list(completion.token_ids),
        "logits": flat_logit_rows(completion),
    }


def first_token_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_captures(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    first_difference = first_token_difference(
        reference["token_ids"], candidate["token_ids"]
    )
    total_steps = min(len(reference["logits"]), len(candidate["logits"]))
    # The logits at the first differing token still used identical input
    # prefixes. Only later rows include different generated token histories.
    comparable_steps = (
        total_steps
        if first_difference is None
        else min(total_steps, first_difference + 1)
    )
    max_differences = []
    mean_differences = []
    for step in range(comparable_steps):
        reference_ids, reference_logits = reference["logits"][step]
        candidate_ids, candidate_logits = candidate["logits"][step]
        if not np.array_equal(reference_ids, candidate_ids):
            raise RuntimeError("vLLM returned different vocabulary IDs across runs")
        difference = np.abs(reference_logits - candidate_logits)
        max_differences.append(float(difference.max()))
        mean_differences.append(float(difference.mean()))

    return {
        "comparable_steps": comparable_steps,
        "first_token_difference": first_difference,
        "first_step_max_abs_logit_diff": max_differences[0],
        "last_step_max_abs_logit_diff": max_differences[-1],
        "max_abs_logit_diff": max(max_differences),
        "mean_abs_logit_diff": statistics.mean(mean_differences),
    }


def benchmark_numerics(
    config: dict[str, Any],
    llm: Any,
    prompt: dict[str, list[int]],
    lora_request: Any,
    controller: SplitKController,
) -> list[dict[str, Any]]:
    if config["numeric_tokens"] == 0:
        return []

    captures: dict[int, list[dict[str, Any]]] = {}
    for split_k in config["split_k"]:
        captures[split_k] = [
            capture_logits(config, llm, prompt, lora_request, controller, split_k)
            for _ in range(config["numeric_repeats"])
        ]

    reference = captures[1][0]
    rows = []
    for split_k in config["split_k"]:
        comparison = compare_captures(reference, captures[split_k][0])
        repeat_drift = 0.0
        for repeated in captures[split_k][1:]:
            repeat_comparison = compare_captures(captures[split_k][0], repeated)
            repeat_drift = max(repeat_drift, repeat_comparison["max_abs_logit_diff"])
        rows.append(
            {
                "rank": config["rank"],
                "stream_mode": config["stream_mode"],
                "split_k": split_k,
                **comparison,
                "max_repeat_abs_logit_diff": repeat_drift,
                "generated_token_ids": captures[split_k][0]["token_ids"],
            }
        )
    return rows


def run_worker(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    print(
        f"\nrank={config['rank']} stream={config['stream_mode']}: "
        "preparing adapter and loading vLLM",
        flush=True,
    )
    adapter_path = make_synthetic_adapter(config)

    # Install the hook before constructing the engine. Multiprocessing is
    # disabled by the coordinator so model execution stays in this process.
    controller = SplitKController()

    import torch
    import transformers
    import vllm
    from transformers import AutoConfig
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    model_config = AutoConfig.from_pretrained(
        config["model"],
        revision=config.get("revision"),
        cache_dir=config.get("download_dir"),
    )
    max_tokens = max(config["output_tokens"], config["numeric_tokens"])
    engine_args: dict[str, Any] = {
        "model": config["model"],
        "revision": config.get("revision"),
        "tokenizer_mode": config["tokenizer_mode"],
        "dtype": config["dtype"],
        "seed": config["seed"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "max_model_len": config["prompt_tokens"] + max_tokens + 8,
        "max_num_seqs": config["batch_size"],
        "enable_prefix_caching": False,
        # With async scheduling, vLLM may process the prefill and decode outputs
        # together. Their request timestamps then understate a one-step decode.
        "async_scheduling": config["async_scheduling"],
        "disable_log_stats": False,
        "enforce_eager": True,
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": config["rank"],
        "lora_target_modules": list(TARGET_MODULES),
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
    }
    if config.get("download_dir"):
        engine_args["download_dir"] = config["download_dir"]
    llm = LLM(**engine_args)

    prompts = fixed_token_prompts(
        batch_size=config["batch_size"],
        prompt_tokens=config["prompt_tokens"],
        vocab_size=model_config.vocab_size,
        bos_token_id=model_config.bos_token_id,
    )
    lora_request = LoRARequest(f"synthetic-rank-{config['rank']}", 1, str(adapter_path))
    performance = benchmark_performance(config, llm, prompts, lora_request, controller)
    numerics = benchmark_numerics(config, llm, prompts[0], lora_request, controller)
    result = {
        "configuration": config,
        "environment": {
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "adapter_path": str(adapter_path),
        },
        "performance": performance,
        "numerics": numerics,
    }
    Path(config["result_file"]).write_text(json.dumps(result, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def format_optional_index(value: int | None) -> str:
    return "-" if value is None else str(value)


def format_optional_float(
    value: float | None, width: int, precision: int = 3, suffix: str = ""
) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:.{precision}f}{suffix}".rjust(width)


def print_results(results: Sequence[dict[str, Any]]) -> None:
    performance = [row for result in results for row in result["performance"]]
    print("\nPerformance (medians across trials)")
    print(
        "mode    rank  split_k  wall ms  prefill ms  prefill speedup  "
        "decode ms/tok  decode speedup"
    )
    for row in performance:
        print(
            f"{row['stream_mode']:<7} {row['rank']:>4} {row['split_k']:>8} "
            f"{row['wall_ms']:>8.2f} {row['prefill_ms']:>11.2f} "
            f"{format_optional_float(row['prefill_speedup_vs_k1'], 15, suffix='x')} "
            f"{format_optional_float(row['decode_ms_per_token'], 14)} "
            f"{format_optional_float(row['decode_speedup_vs_k1'], 14, suffix='x')}"
        )

    numerics = [row for result in results for row in result["numerics"]]
    if not numerics:
        return
    print("\nNumerics (raw logits; split_k=1 is the reference)")
    print("mode    rank  split_k  steps  first token diff  max abs diff  repeat max")
    for row in numerics:
        print(
            f"{row['stream_mode']:<7} {row['rank']:>4} {row['split_k']:>8} "
            f"{row['comparable_steps']:>6} "
            f"{format_optional_index(row['first_token_difference']):>16} "
            f"{row['max_abs_logit_diff']:>13.6g} "
            f"{row['max_repeat_abs_logit_diff']:>11.6g}"
        )


def coordinator_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_mode": args.tokenizer_mode,
        "split_k": list(args.split_k),
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "warmups": args.warmups,
        "trials": args.trials,
        "async_scheduling": args.async_scheduling,
        "numeric_tokens": args.numeric_tokens,
        "numeric_repeats": args.numeric_repeats,
        "adapter_scale": args.adapter_scale,
        "seed": args.seed,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "adapter_cache": str(args.adapter_cache.expanduser().resolve()),
        "download_dir": (
            str(args.download_dir.expanduser().resolve())
            if args.download_dir is not None
            else None
        ),
    }


def write_aggregate_results(
    output: Path, model: str, results: Sequence[dict[str, Any]], expected_workers: int
) -> None:
    aggregate = {
        "model": model,
        "complete": len(results) == expected_workers,
        "workers": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n")


def run_coordinator(args: argparse.Namespace) -> None:
    base_config = coordinator_config(args)
    results = []
    script = Path(__file__).resolve()
    output = args.output.expanduser().resolve() if args.output is not None else None
    expected_workers = len(args.stream_modes) * len(args.ranks)
    with tempfile.TemporaryDirectory(prefix="vllm-split-k-") as temporary:
        temporary_dir = Path(temporary)
        for stream_mode in args.stream_modes:
            for rank in args.ranks:
                stem = f"{stream_mode}-rank-{rank}"
                config_path = temporary_dir / f"{stem}-config.json"
                result_path = temporary_dir / f"{stem}-result.json"
                worker_config = {
                    **base_config,
                    "rank": rank,
                    "stream_mode": stream_mode,
                    "numeric_tokens": (
                        args.numeric_tokens
                        if stream_mode in args.numeric_stream_modes
                        else 0
                    ),
                    "result_file": str(result_path),
                }
                config_path.write_text(json.dumps(worker_config, indent=2) + "\n")
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": args.gpu,
                        "PATH": (
                            f"{Path(sys.executable).parent}:"
                            f"{environment.get('PATH', '')}"
                        ),
                        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                        "VLLM_USE_FLASHINFER_SAMPLER": "0",
                        "VLLM_LORA_ENABLE_DUAL_STREAM": (
                            "1" if stream_mode == "dual" else "0"
                        ),
                    }
                )
                subprocess.run(
                    [sys.executable, str(script), "--worker-config", str(config_path)],
                    check=True,
                    env=environment,
                )
                results.append(json.loads(result_path.read_text()))
                if output is not None:
                    write_aggregate_results(
                        output, args.model, results, expected_workers
                    )

    print_results(results)
    if output is not None:
        print(f"\nWrote {output}")


def main() -> None:
    args = parse_args()
    if args.worker_config is not None:
        run_worker(args.worker_config)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
