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


DEFAULT_PROMPTS = [
    "Explain suffix decoding in 5 bullet points.",
    "Summarize the causes of the French Revolution in 6 bullet points.",
    "Return a JSON object with keys topic, pros, cons comparing Rust and Go for backend systems.",
    "Write a short explanation of why KV cache helps autoregressive decoding.",
    "Give a 7-step debugging plan for CUDA out-of-memory errors in PyTorch.",
    "Explain grouped query attention and why it can reduce memory bandwidth pressure.",
    "Describe how GPTQ quantization works at a high level in 8 bullet points.",
    "Write a concise explanation of product quantization for approximate nearest neighbor search.",
    "Compare batch normalization and layer normalization in a short table.",
    "Explain how a transformer uses queries, keys, and values in simple terms.",
    "What is MinHash and why does it estimate Jaccard similarity?",
    "Give a structured answer with headings on why decode is often memory-bound.",
    "Explain the difference between TF-IDF and BM25.",
    "Give a short tutorial on how speculative decoding works.",
    "Write a compact explanation of RoPE and why it encodes relative position information.",
    "Explain why residual connections help deep neural networks train.",
]


@dataclass
class SampleResult:
    idx: int
    prompt: str
    model_label: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_text: str


@dataclass
class PairwiseQuality:
    idx: int
    prompt: str
    bf16_text: str
    quant_text: str
    exact_match: bool
    norm_edit_similarity: float
    rouge_l_f1: float
    bf16_len: int
    quant_len: int
    len_ratio: float


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(ref: str, hyp: str) -> float:
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    if not ref_toks or not hyp_toks:
        return 0.0
    lcs = lcs_length(ref_toks, hyp_toks)
    prec = lcs / max(len(hyp_toks), 1)
    rec = lcs / max(len(ref_toks), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def norm_edit_similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / denom)


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


def post_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    idx: int,
    prompt: str,
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
        prompt=prompt,
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
        idx=idx,
        prompt=prompt,
        model_label=model_label,
        latency_s=t1 - t0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_text=text,
    )


def run_batched_eval(
    prompts: list[str],
    base_url: str,
    model: str,
    model_label: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> list[SampleResult]:
    results: list[SampleResult] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        with cf.ThreadPoolExecutor(max_workers=batch_size) as ex:
            futs = [
                ex.submit(
                    run_one_request,
                    start + i,
                    prompt,
                    base_url,
                    model,
                    model_label,
                    max_tokens,
                    temperature,
                    timeout_s,
                )
                for i, prompt in enumerate(batch)
            ]
            for fut in cf.as_completed(futs):
                results.append(fut.result())
    return sorted(results, key=lambda x: x.idx)


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
        "mean_req_completion_toks_per_s": (
            statistics.mean(per_req_tps) if per_req_tps else 0.0
        ),
        "aggregate_completion_toks_per_s": (
            total_completion_tokens / total_wall if total_wall > 0 else 0.0
        ),
    }


def compare_quality(
    bf16_results: list[SampleResult],
    quant_results: list[SampleResult],
) -> tuple[list[PairwiseQuality], dict[str, Any]]:
    assert len(bf16_results) == len(quant_results)
    pairs: list[PairwiseQuality] = []

    for a, b in zip(bf16_results, quant_results):
        assert a.idx == b.idx
        sim = norm_edit_similarity(a.output_text, b.output_text)
        rouge = rouge_l_f1(a.output_text, b.output_text)
        len_ratio = (len(b.output_text) / max(len(a.output_text), 1))
        pairs.append(
            PairwiseQuality(
                idx=a.idx,
                prompt=a.prompt,
                bf16_text=a.output_text,
                quant_text=b.output_text,
                exact_match=(a.output_text == b.output_text),
                norm_edit_similarity=sim,
                rouge_l_f1=rouge,
                bf16_len=len(a.output_text),
                quant_len=len(b.output_text),
                len_ratio=len_ratio,
            )
        )

    summary = {
        "num_prompts": len(pairs),
        "exact_match_rate": sum(p.exact_match for p in pairs) / max(len(pairs), 1),
        "avg_norm_edit_similarity": statistics.mean(p.norm_edit_similarity for p in pairs),
        "avg_rouge_l_f1": statistics.mean(p.rouge_l_f1 for p in pairs),
        "avg_len_ratio": statistics.mean(p.len_ratio for p in pairs),
    }
    return pairs, summary


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON prompts file must contain a list of strings.")
        return [str(x) for x in data]
    lines = [line.strip() for line in p.read_text().splitlines()]
    return [line for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-base-url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--bf16-model", type=str, required=True)
    parser.add_argument("--quant-base-url", type=str, default="http://localhost:8002/v1")
    parser.add_argument("--quant-model", type=str, required=True)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output-json", type=str, default="eval_results.json")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    print(f"[INFO] Loaded {len(prompts)} prompts")
    print(f"[INFO] Batch size: {args.batch_size}")

    print("[INFO] Running BF16 baseline...")
    bf16_results = run_batched_eval(
        prompts=prompts,
        base_url=args.bf16_base_url,
        model=args.bf16_model,
        model_label="bf16",
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )

    print("[INFO] Running W4A16 model...")
    quant_results = run_batched_eval(
        prompts=prompts,
        base_url=args.quant_base_url,
        model=args.quant_model,
        model_label="w4a16",
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )

    bf16_perf = summarize_perf(bf16_results)
    quant_perf = summarize_perf(quant_results)
    quality_rows, quality_summary = compare_quality(bf16_results, quant_results)

    speedup = (
        bf16_perf["avg_latency_s"] / quant_perf["avg_latency_s"]
        if quant_perf["avg_latency_s"] > 0
        else float("nan")
    )
    throughput_gain = (
        quant_perf["aggregate_completion_toks_per_s"] / bf16_perf["aggregate_completion_toks_per_s"]
        if bf16_perf["aggregate_completion_toks_per_s"] > 0
        else float("nan")
    )

    final_summary = {
        "settings": {
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "num_prompts": len(prompts),
        },
        "bf16_perf": bf16_perf,
        "w4a16_perf": quant_perf,
        "relative": {
            "avg_latency_speedup_bf16_over_w4a16": speedup,
            "aggregate_completion_tps_gain_w4a16_over_bf16": throughput_gain,
        },
        "quality": quality_summary,
    }

    print("\n=== PERFORMANCE SUMMARY ===")
    print(json.dumps(final_summary, indent=2))

    worst = sorted(quality_rows, key=lambda x: (x.rouge_l_f1, x.norm_edit_similarity))[:5]
    print("\n=== WORST 5 QUALITY CASES ===")
    for row in worst:
        print(f"\n--- idx={row.idx} ---")
        print(f"Prompt: {row.prompt}")
        print(f"Exact match: {row.exact_match}")
        print(f"Norm edit similarity: {row.norm_edit_similarity:.4f}")
        print(f"ROUGE-L F1: {row.rouge_l_f1:.4f}")
        print(f"\n[BF16]\n{row.bf16_text}")
        print(f"\n[W4A16]\n{row.quant_text}")

    payload = {
        "summary": final_summary,
        "bf16_results": [asdict(x) for x in bf16_results],
        "w4a16_results": [asdict(x) for x in quant_results],
        "quality_rows": [asdict(x) for x in quality_rows],
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(f"\n[INFO] Wrote results to {args.output_json}")


if __name__ == "__main__":
    main()