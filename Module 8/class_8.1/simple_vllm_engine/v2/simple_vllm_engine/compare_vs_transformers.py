#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _drop_package_dir_from_sys_path() -> None:
    # When this file is launched directly, Python puts this package directory on
    # sys.path. That makes local `requests.py` shadow the third-party `requests`
    # package used by Transformers/Hugging Face. We import through the version
    # root instead, so the package dir itself should not be a top-level path.
    package_dir = Path(__file__).resolve().parent

    def is_package_dir(entry: str) -> bool:
        path = Path.cwd() if entry == "" else Path(entry)
        return path.resolve() == package_dir

    sys.path[:] = [entry for entry in sys.path if not is_package_dir(entry)]


_drop_package_dir_from_sys_path()

import torch

_VERSION_ROOT = Path(__file__).resolve().parent.parent
_TEACHING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VERSION_ROOT))
sys.path.insert(1, str(_TEACHING_ROOT))

from simple_vllm_engine.config import EngineConfig
from simple_vllm_engine.engine import sample_greedy
from common.hf_loader import build_engine_from_pretrained
from simple_vllm_engine.requests import RequestSpec, RequestState
from common.tokenizer import HFTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ankur/dev/models/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=128)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=64)
    parser.add_argument("--max-decode-batch-size", type=int, default=8)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in a few sentences why continuous batching helps LLM serving throughput.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Encode the prompt directly instead of using the tokenizer chat template.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def build_prompt_ids(tokenizer: HFTokenizer, prompt: str, use_chat_template: bool) -> list[int]:
    if use_chat_template:
        return tokenizer.encode_chat([{"role": "user", "content": prompt}])
    return tokenizer.encode(prompt, add_bos=True, add_eos=False)


def summarize_topk(logits: torch.Tensor, k: int) -> dict[str, list[float] | list[int] | float]:
    topk = torch.topk(logits[0].float(), k=k)
    values = topk.values.tolist()
    indices = topk.indices.tolist()
    margin = values[0] - values[1] if len(values) > 1 else float("inf")
    return {"ids": indices, "scores": values, "margin": margin}


def generate_with_transformers_stepwise(
    model_path: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[list[int], list[dict[str, list[float] | list[int] | float]]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated: list[int] = []
    step_summaries: list[dict[str, list[float] | list[int] | float]] = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        step_summaries.append(summarize_topk(logits, k=8))
        next_token = int(sample_greedy(logits)[0].item())
        generated.append(next_token)
        past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens - 1):
            step_input_ids = torch.tensor([[next_token]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=step_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            step_summaries.append(summarize_topk(logits, k=8))
            next_token = int(sample_greedy(logits)[0].item())
            generated.append(next_token)
            past_key_values = outputs.past_key_values

    del model
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return generated, step_summaries


def generate_with_engine_stepwise(
    model_path: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    max_batch_tokens: int,
    max_prefill_chunk_tokens: int,
    max_decode_batch_size: int,
) -> tuple[list[int], HFTokenizer, list[dict[str, list[float] | list[int] | float]]]:
    engine_config = EngineConfig(
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_tokens=max_batch_tokens,
        max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        max_decode_batch_size=max_decode_batch_size,
        device=device,
        dtype=dtype,
    )
    engine, tokenizer = build_engine_from_pretrained(model_path, engine_config)
    req = RequestState.from_spec(
        RequestSpec(request_id="req-0", prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, arrival_step=0)
    )
    req.block_ids = engine.kv_cache.ensure_capacity(req.block_ids, req.prompt_len)
    step_summaries: list[dict[str, list[float] | list[int] | float]] = []

    with torch.no_grad():
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        positions = torch.arange(req.prompt_len, device=device, dtype=torch.long).unsqueeze(0)
        logits = engine.model.prefill_chunk(
            requests=[req],
            input_ids=input_ids,
            positions=positions,
            lengths=[req.prompt_len],
            kv_cache=engine.kv_cache,
        )
        step_summaries.append(summarize_topk(logits, k=8))
        req.prompt_tokens_computed = req.prompt_len

        next_token = int(sample_greedy(logits)[0].item())
        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            req.block_ids = engine.kv_cache.ensure_capacity(req.block_ids, req.cached_seq_len + 1)
            step_input_ids = torch.tensor([[next_token]], device=device, dtype=torch.long)
            step_positions = torch.tensor([[req.cached_seq_len]], device=device, dtype=torch.long)
            logits = engine.model.decode_tokens(
                requests=[req],
                input_ids=step_input_ids,
                positions=step_positions,
                kv_cache=engine.kv_cache,
            )
            step_summaries.append(summarize_topk(logits, k=8))
            req.generated_tokens_in_cache += 1
            next_token = int(sample_greedy(logits)[0].item())
            generated.append(next_token)

    return generated, tokenizer, step_summaries


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for idx, (lhs, rhs) in enumerate(zip(a, b)):
        if lhs != rhs:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def format_topk(summary: dict[str, list[float] | list[int] | float], tokenizer: HFTokenizer) -> str:
    ids = summary["ids"]
    scores = summary["scores"]
    margin = summary["margin"]
    pieces = []
    for token_id, score in zip(ids, scores):
        pieces.append(f"{token_id}:{tokenizer.decode([int(token_id)]).replace(chr(10), ' ')} ({score:.4f})")
    return f"margin={margin:.6f} topk=[{'; '.join(pieces)}]"


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    tokenizer = HFTokenizer.from_pretrained(args.model_path)
    prompt_ids = build_prompt_ids(tokenizer, args.prompt, use_chat_template=not args.no_chat_template)

    print("Running transformers reference...")
    ref_ids, ref_steps = generate_with_transformers_stepwise(
        model_path=args.model_path,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=dtype,
    )

    print("Running v3 engine...")
    engine_ids, engine_tokenizer, engine_steps = generate_with_engine_stepwise(
        model_path=args.model_path,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        max_decode_batch_size=args.max_decode_batch_size,
    )

    mismatch = first_mismatch(ref_ids, engine_ids)
    print()
    print(f"reference ids == engine ids: {mismatch is None}")
    print(f"reference text: {engine_tokenizer.decode(ref_ids)}")
    print(f"engine text:    {engine_tokenizer.decode(engine_ids)}")

    if mismatch is not None:
        print(f"first mismatch at generated token index {mismatch}")
        if mismatch < len(ref_ids):
            print(f"reference token id: {ref_ids[mismatch]}")
        if mismatch < len(engine_ids):
            print(f"engine token id:    {engine_ids[mismatch]}")
        if mismatch < len(ref_steps):
            print("reference logits:", format_topk(ref_steps[mismatch], engine_tokenizer))
        if mismatch < len(engine_steps):
            print("engine logits:   ", format_topk(engine_steps[mismatch], engine_tokenizer))


if __name__ == "__main__":
    main()
