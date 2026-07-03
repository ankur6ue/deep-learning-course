#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


DEFAULT_MODEL_PATH = "/home/ankur/dev/models/Llama-3.1-8B-Instruct"
DEFAULT_WORKLOAD_PATH = Path(__file__).resolve().parents[1] / "workloads" / "shared_chat_workload.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared simple_vllm workload on production vLLM.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--workload-path", type=Path, default=DEFAULT_WORKLOAD_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="Forwarded to vLLM. Use 'hf' for local text-only Mistral compatibility checkpoints.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override vLLM scheduler max_num_batched_tokens.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override vLLM scheduler max_num_seqs.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override vLLM KV cache block size.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graph capture. Useful when isolating the attention/kernel stack.",
    )
    parser.add_argument(
        "--include-load-time",
        action="store_true",
        help="Include LLM construction/model load time in the reported wall time.",
    )
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--reset-prefix-cache-between-runs",
        action="store_true",
        help="Clear vLLM's prefix cache before each measured run.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM automatic prefix caching for model-runner-only comparisons.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write machine-readable timing and generated token ids for comparison runs.",
    )
    return parser.parse_args()


def load_workload(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{path} must contain a non-empty 'requests' list")
    return requests


def encode_prompts(tokenizer, workload: list[dict[str, object]]) -> list[list[int]]:
    prompt_ids: list[list[int]] = []
    for idx, item in enumerate(workload):
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"request {idx} is missing 'messages'")
        if not getattr(tokenizer, "chat_template", None):
            text_parts = []
            for message in messages:
                role = str(message.get("role", "user")).strip() or "user"
                content = str(message.get("content", ""))
                text_parts.append(f"{role}: {content}")
            text_parts.append("assistant:")
            ids = list(tokenizer.encode("\n\n".join(text_parts), add_special_tokens=False))
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            if bos_token_id is not None and (not ids or ids[0] != bos_token_id):
                ids = [int(bos_token_id)] + ids
        else:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            if hasattr(encoded, "input_ids"):
                ids = encoded.input_ids
            else:
                ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        prompt_ids.append(list(ids))
    return prompt_ids


def main() -> None:
    args = parse_args()
    load_t0 = time.perf_counter()
    try:
        # Some local Mistral text-only compatibility checkpoints use
        # Transformers' MistralCommonBackend, which behaves like a slow tokenizer
        # but does not expose every tokenizer method that vLLM checks.
        import transformers.tokenization_mistral_common as mistral_tokenization

        backend_classes = [
            getattr(mistral_tokenization, name)
            for name in dir(mistral_tokenization)
            if name.endswith("Backend")
        ]
        for backend_cls in backend_classes:
            if not hasattr(backend_cls, "is_fast"):
                backend_cls.is_fast = property(lambda self: False)  # type: ignore[attr-defined]
            if not hasattr(backend_cls, "get_added_vocab"):
                backend_cls.get_added_vocab = lambda self: {}  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    workload = load_workload(args.workload_path)
    prompt_ids = encode_prompts(tokenizer, workload)

    print("Workload summary:")
    for idx, (item, ids) in enumerate(zip(workload, prompt_ids, strict=True)):
        request_id = str(item.get("request_id", f"req-{idx}"))
        arrival_step = int(item.get("arrival_step", 0))
        print(
            f"{request_id}: arrival={arrival_step}, "
            f"prompt_tokens={len(ids)}, max_new_tokens={args.max_new_tokens}"
        )
    print()

    llm_kwargs = {
        "model": args.model_path,
        "dtype": args.dtype,
        "tokenizer_mode": args.tokenizer_mode,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": not args.disable_prefix_caching,
        "seed": args.seed,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.block_size is not None:
        llm_kwargs["block_size"] = args.block_size

    llm = LLM(**llm_kwargs)
    load_t1 = time.perf_counter()

    vllm_config = getattr(llm, "llm_engine", None)
    vllm_config = getattr(vllm_config, "vllm_config", None)
    if vllm_config is not None:
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        print("vLLM effective config:")
        print(f"  dtype={model_config.dtype}")
        print(f"  max_model_len={model_config.max_model_len}")
        print(f"  kv_cache_dtype={cache_config.cache_dtype}")
        print(f"  block_size={cache_config.block_size}")
        print(f"  enable_prefix_caching={cache_config.enable_prefix_caching}")
        print(f"  max_num_batched_tokens={scheduler_config.max_num_batched_tokens}")
        print(f"  max_num_seqs={scheduler_config.max_num_seqs}")
        print(f"  enable_chunked_prefill={scheduler_config.enable_chunked_prefill}")
        print()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]

    load_wall = load_t1 - load_t0
    label = "vllm_offline"
    if args.enforce_eager:
        label += "[eager]"

    if args.num_runs < 1:
        raise ValueError("num-runs must be at least 1")

    outputs = None
    outputs_by_run = []
    run_summaries = []
    for run_idx in range(args.num_runs):
        if args.reset_prefix_cache_between_runs:
            llm.reset_prefix_cache()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        run_t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        outputs_by_run.append(outputs)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        run_t1 = time.perf_counter()

        generate_wall = run_t1 - run_t0
        reported_wall = run_t1 - load_t0 if args.include_load_time else generate_wall
        generated_lengths = [len(output.outputs[0].token_ids) for output in outputs]
        total_generated = sum(generated_lengths)
        throughput = total_generated / reported_wall if reported_wall > 0 else 0.0
        print(
            f"{label} run={run_idx + 1}: wall={reported_wall:.3f}s, "
            f"generate_wall={generate_wall:.3f}s, load_wall={load_wall:.3f}s, "
            f"generated={total_generated} toks, throughput={throughput:.2f} toks/s, "
            f"mean_gen_len={statistics.mean(generated_lengths):.1f}"
        )
        run_summaries.append(
            {
                "run": run_idx + 1,
                "wall_s": reported_wall,
                "generate_wall_s": generate_wall,
                "load_wall_s": load_wall,
                "generated_tokens": total_generated,
                "throughput_toks_per_s": throughput,
                "mean_gen_len": statistics.mean(generated_lengths),
            }
        )
    print()

    if outputs is None:
        raise RuntimeError("no vLLM outputs were produced")
    for idx, output in enumerate(outputs):
        item = workload[idx]
        request_id = str(item.get("request_id", f"req-{idx}"))
        generated = output.outputs[0]
        text = generated.text.replace("\n", " ")[:160]
        print(
            f"{request_id}: generated={len(generated.token_ids)}, "
            f"finish={generated.finish_reason}, output: {text}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        def serialize_results(run_outputs):
            results = []
            for idx, output in enumerate(run_outputs):
                item = workload[idx]
                request_id = str(item.get("request_id", f"req-{idx}"))
                generated = output.outputs[0]
                results.append(
                    {
                        "request_id": request_id,
                        "prompt_tokens": len(prompt_ids[idx]),
                        "generated_ids": list(generated.token_ids),
                        "finish_reason": generated.finish_reason,
                        "text": generated.text,
                    }
                )
            return results

        results = serialize_results(outputs)
        results_by_run = [
            {"run": run_idx + 1, "results": serialize_results(run_outputs)}
            for run_idx, run_outputs in enumerate(outputs_by_run)
        ]
        args.output_json.write_text(
            json.dumps(
                {
                    "runner": "vllm",
                    "model_path": args.model_path,
                    "workload_path": str(args.workload_path),
                    "label": label,
                    "settings": {
                        "max_new_tokens": args.max_new_tokens,
                        "dtype": args.dtype,
                        "max_model_len": args.max_model_len,
                        "max_num_batched_tokens": args.max_num_batched_tokens,
                        "max_num_seqs": args.max_num_seqs,
                        "block_size": args.block_size,
                        "gpu_memory_utilization": args.gpu_memory_utilization,
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "enforce_eager": args.enforce_eager,
                        "enable_prefix_caching": not args.disable_prefix_caching,
                        "seed": args.seed,
                    },
                    "runs": run_summaries,
                    "results_by_run": results_by_run,
                    "results": results,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
