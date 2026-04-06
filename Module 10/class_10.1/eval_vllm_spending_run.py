#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests


CATEGORY_NAMES = [
    "Groceries",
    "Travel",
    "Dining",
    "Gas",
    "Entertainment",
    "Shopping",
]

MERCHANT_NAMES = [
    "Amazon",
    "Uber",
    "WholeFoods",
    "Netflix",
    "Starbucks",
    "Target",
    "Delta",
    "Shell",
    "Apple",
    "Airbnb",
]

MONTH_SETS_3 = [
    ["Jan", "Feb", "Mar"],
    ["Apr", "May", "Jun"],
    ["Jul", "Aug", "Sep"],
    ["Oct", "Nov", "Dec"],
]

MONTH_SETS_4 = [
    ["Jan", "Feb", "Mar", "Apr"],
    ["May", "Jun", "Jul", "Aug"],
    ["Sep", "Oct", "Nov", "Dec"],
]

MONTH_SETS_5 = [
    ["Jan", "Feb", "Mar", "Apr", "May"],
    ["Jun", "Jul", "Aug", "Sep", "Oct"],
    ["Nov", "Dec", "Jan", "Feb", "Mar"],
]


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
    }


def build_values(
    rng: random.Random,
    items: list[str],
    months: list[str],
    kind: str,
) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = {}

    for item in items:
        if kind == "category":
            base = rng.randint(120, 600)
            noise = rng.randint(10, 60)
        else:
            if item == "Netflix":
                base = rng.randint(15, 20)
                noise = rng.randint(0, 2)
            elif item in {"Delta", "Airbnb", "Apple"}:
                base = rng.randint(0, 260)
                noise = rng.randint(30, 120)
            else:
                base = rng.randint(40, 320)
                noise = rng.randint(8, 50)

        trend = rng.uniform(-0.18, 0.22)
        cur = float(base)
        month_map: dict[str, int] = {}

        for month in months:
            seasonal = rng.uniform(-noise, noise)
            cur = max(0.0, cur * (1.0 + trend) + seasonal)

            if kind == "merchant" and item in {"Delta", "Airbnb", "Apple"} and rng.random() < 0.18:
                cur = max(0.0, cur * rng.uniform(0.0, 0.35))
            if kind == "merchant" and item in {"Delta", "Airbnb"} and rng.random() < 0.15:
                cur += rng.uniform(120, 260)

            month_map[month] = int(round(cur))
        values[item] = month_map

    return values


def render_user_prompt(
    kind: str,
    months: list[str],
    items: list[str],
    values: dict[str, dict[str, int]],
) -> str:
    lines: list[str] = []
    label = "category" if kind == "category" else "merchant"
    lines.append(f"Data type: {label}-level spending")
    lines.append("")

    for month in months:
        parts = [f"{item} ${values[item][month]}" for item in items]
        lines.append(f"{month}: " + ", ".join(parts))

    lines.append("")
    lines.append("Please summarize how each item changed month over month, including dollar and percentage changes.")
    return "\n".join(lines)


def generate_prompt_specs(
    instruction_text: str,
    num_prompts: int,
    seed: int,
    include_categories: bool,
    include_merchants: bool,
) -> list[PromptSpec]:
    if not include_categories and not include_merchants:
        raise ValueError("At least one of include_categories/include_merchants must be true.")

    rng = random.Random(seed)
    specs: list[PromptSpec] = []

    available_kinds: list[str] = []
    if include_categories:
        available_kinds.append("category")
    if include_merchants:
        available_kinds.append("merchant")

    for idx in range(num_prompts):
        kind = available_kinds[idx % len(available_kinds)]
        if len(available_kinds) > 1:
            kind = rng.choice(available_kinds)

        month_count = rng.choice([3, 4, 5])
        if month_count == 3:
            months = list(rng.choice(MONTH_SETS_3))
        elif month_count == 4:
            months = list(rng.choice(MONTH_SETS_4))
        else:
            months = list(rng.choice(MONTH_SETS_5))

        items = list(CATEGORY_NAMES if kind == "category" else MERCHANT_NAMES)
        values = build_values(rng, items, months, kind)

        specs.append(
            PromptSpec(
                idx=idx,
                kind=kind,
                months=months,
                items=items,
                values=values,
                system_prompt=instruction_text.strip(),
                user_prompt=render_user_prompt(kind, months, items, values),
            )
        )

    return specs


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
        "Authorization": "Bearer dummy",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def run_one_request(
    spec: PromptSpec,
    base_url: str,
    model: str,
    model_label: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> SampleResult:
    t0 = time.perf_counter()
    data = post_chat_completion(
        base_url=base_url,
        model=model,
        system_prompt=spec.system_prompt,
        user_prompt=spec.user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )
    t1 = time.perf_counter()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

    return SampleResult(
        idx=spec.idx,
        system_prompt=spec.system_prompt,
        user_prompt=spec.user_prompt,
        model_label=model_label,
        latency_s=t1 - t0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_text=text,
    )


def run_batched_eval(
    specs: list[PromptSpec],
    base_url: str,
    model: str,
    model_label: str,
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
    parser.add_argument("--num-prompts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--categories-only", action="store_true")
    parser.add_argument("--merchants-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    if args.categories_only and args.merchants_only:
        raise ValueError("Choose at most one of --categories-only / --merchants-only")

    instruction_text = Path(args.instruction_file).read_text()
    specs = generate_prompt_specs(
        instruction_text=instruction_text,
        num_prompts=args.num_prompts,
        seed=args.seed,
        include_categories=not args.merchants_only,
        include_merchants=not args.categories_only,
    )

    print(f"[INFO] Generated {len(specs)} prompts")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Running against {args.base_url}")
    print(f"[INFO] Batch size: {args.batch_size}")

    results = run_batched_eval(
        specs=specs,
        base_url=args.base_url,
        model=args.model,
        model_label=args.model_label,
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
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "num_prompts": len(specs),
            "seed": args.seed,
            "instruction_file": args.instruction_file,
            "categories_only": args.categories_only,
            "merchants_only": args.merchants_only,
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