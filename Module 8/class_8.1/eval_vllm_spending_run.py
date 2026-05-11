#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests


@dataclass
class PromptSpec:
    idx: int
    kind: str
    months: list[str]
    items: list[str]
    values: dict[str, dict[str, int]]
    system_prompt: str
    user_prompt: str


@dataclass
class SampleResult:
    idx: int
    system_prompt: str
    user_prompt: str
    model_label: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_text: str
    reasoning_text: str
    finish_reason: str | None
    raw_message: dict[str, Any] | None
    raw_choice: dict[str, Any] | None
    raw_response: dict[str, Any]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def summarize_perf(results: list[SampleResult]) -> dict[str, Any]:
    latencies = [r.latency_s for r in results]
    completion_tokens = [r.completion_tokens for r in results]
    total_completion_tokens = sum(completion_tokens)
    total_wall = sum(latencies)

    per_req_tps = [
        (r.completion_tokens / r.latency_s) if r.latency_s > 0 else 0.0
        for r in results
    ]

    empty_output_count = sum(1 for r in results if not r.output_text.strip())
    reasoning_present_count = sum(1 for r in results if r.reasoning_text.strip())

    return {
        "num_prompts": len(results),
        "avg_latency_s": statistics.mean(latencies) if latencies else float("nan"),
        "median_latency_s": statistics.median(latencies) if latencies else float("nan"),
        "p95_latency_s": percentile(latencies, 0.95),
        "min_latency_s": min(latencies) if latencies else float("nan"),
        "max_latency_s": max(latencies) if latencies else float("nan"),
        "avg_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0.0,
        "sum_completion_tokens": total_completion_tokens,
        "mean_req_completion_toks_per_s": statistics.mean(per_req_tps) if per_req_tps else 0.0,
        "aggregate_completion_toks_per_s": (
            total_completion_tokens / total_wall if total_wall > 0 else 0.0
        ),
        "empty_output_count": empty_output_count,
        "reasoning_present_count": reasoning_present_count,
    }


def load_prompt_specs(prompt_file: str, instruction_file: str) -> list[PromptSpec]:
    system_prompt = Path(instruction_file).read_text(encoding="utf-8").strip()
    specs: list[PromptSpec] = []

    with Path(prompt_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            specs.append(
                PromptSpec(
                    idx=int(obj["idx"]),
                    kind=obj.get("kind", ""),
                    months=obj.get("months", []),
                    items=obj.get("items", []),
                    values=obj.get("values", {}),
                    system_prompt=system_prompt,
                    user_prompt=obj["user_prompt"],
                )
            )

    return sorted(specs, key=lambda x: x.idx)


def extract_text_from_chat_completion(
    data: dict[str, Any],
) -> tuple[str, str, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    choices = data.get("choices", [])
    if not choices:
        return "", "", None, None, None

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    msg = choice.get("message") or {}

    output_text = ""
    reasoning_text = ""

    content = msg.get("content")
    if isinstance(content, str):
        output_text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and item.get("text"):
                parts.append(item["text"])
        output_text = "\n".join(parts).strip()

    if not output_text and msg.get("output_text"):
        output_text = msg.get("output_text") or ""

    if msg.get("reasoning_content"):
        reasoning_text = msg.get("reasoning_content") or ""
    elif msg.get("reasoning"):
        if isinstance(msg["reasoning"], str):
            reasoning_text = msg["reasoning"]

    if not output_text and choice.get("text") is not None:
        output_text = choice.get("text") or ""

    return output_text or "", reasoning_text or "", finish_reason, msg, choice


def extract_text_from_responses(
    data: dict[str, Any],
) -> tuple[str, str, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    output_text_parts: list[str] = []
    reasoning_parts: list[str] = []

    output = data.get("output", [])
    finish_reason = None

    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        status = item.get("status")
        if finish_reason is None and status:
            finish_reason = status

        if item_type == "message":
            for content_item in item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"} and content_item.get("text"):
                    output_text_parts.append(content_item["text"])

        elif item_type in {"output_text", "text"} and item.get("text"):
            output_text_parts.append(item["text"])

        # reasoning extraction pattern you provided
        if item.get("type") == "reasoning":
            for content_item in item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"reasoning_text", "reasoning"} and content_item.get("text"):
                    reasoning_parts.append(content_item["text"])

    if not output_text_parts:
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            output_text_parts.append(output_text)

    return (
        "\n".join(output_text_parts).strip(),
        "\n".join(reasoning_parts).strip(),
        finish_reason,
        None,
        None,
    )


def post_chat_completion(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def post_responses(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def extract_usage(data: dict[str, Any]) -> tuple[int, int, int]:
    usage = data.get("usage", {})

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens", 0)
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens", 0)
    if total_tokens is None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    return int(prompt_tokens), int(completion_tokens), int(total_tokens)


def run_one_request(
    spec: PromptSpec,
    base_url: str,
    model: str,
    model_label: str,
    api_mode: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> SampleResult:
    t0 = time.perf_counter()

    if api_mode == "responses":
        data = post_responses(
            base_url=base_url,
            model=model,
            system_prompt=spec.system_prompt,
            user_prompt=spec.user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        output_text, reasoning_text, finish_reason, raw_message, raw_choice = extract_text_from_responses(data)
    else:
        data = post_chat_completion(
            base_url=base_url,
            model=model,
            system_prompt=spec.system_prompt,
            user_prompt=spec.user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        output_text, reasoning_text, finish_reason, raw_message, raw_choice = extract_text_from_chat_completion(data)

    t1 = time.perf_counter()
    prompt_tokens, completion_tokens, total_tokens = extract_usage(data)

    return SampleResult(
        idx=spec.idx,
        system_prompt=spec.system_prompt,
        user_prompt=spec.user_prompt,
        model_label=model_label,
        latency_s=t1 - t0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_text=output_text,
        reasoning_text=reasoning_text,
        finish_reason=finish_reason,
        raw_message=raw_message,
        raw_choice=raw_choice,
        raw_response=data,
    )


def run_batched_eval(
    specs: list[PromptSpec],
    base_url: str,
    model: str,
    model_label: str,
    api_mode: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> list[SampleResult]:
    results: list[SampleResult] = []
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        with cf.ThreadPoolExecutor(max_workers=batch_size) as ex:
            futs = [
                ex.submit(
                    run_one_request,
                    spec,
                    base_url,
                    model,
                    model_label,
                    api_mode,
                    max_tokens,
                    temperature,
                    timeout_s,
                )
                for spec in batch
            ]
            for fut in cf.as_completed(futs):
                results.append(fut.result())
    return sorted(results, key=lambda x: x.idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--instruction-file", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--api-mode", type=str, choices=["chat", "responses"], default="chat")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    specs = load_prompt_specs(
        prompt_file=args.prompt_file,
        instruction_file=args.instruction_file,
    )

    print(f"[INFO] Loaded {len(specs)} prompts")
    print(f"[INFO] Running against {args.base_url}")
    print(f"[INFO] API mode: {args.api_mode}")
    print(f"[INFO] Batch size: {args.batch_size}")

    results = run_batched_eval(
        specs=specs,
        base_url=args.base_url,
        model=args.model,
        model_label=args.model_label,
        api_mode=args.api_mode,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )

    summary = summarize_perf(results)
    payload = {
        "run_metadata": {
            "base_url": args.base_url,
            "model": args.model,
            "model_label": args.model_label,
            "api_mode": args.api_mode,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "instruction_file": args.instruction_file,
            "prompt_file": args.prompt_file,
            "num_prompts": len(specs),
        },
        "prompt_specs": [asdict(x) for x in specs],
        "summary": summary,
        "results": [asdict(x) for x in results],
    }

    Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote run results to {args.output_json}")


if __name__ == "__main__":
    main()