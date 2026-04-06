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
    "Write a compact explanation of product quantization for approximate nearest neighbor search.",
    "Compare batch normalization and layer normalization in a short table.",
    "Explain how a transformer uses queries, keys, and values in simple terms.",
    "What is MinHash and why does it estimate Jaccard similarity?",
    "Give a structured answer with headings on why decode is often memory-bound.",
    "Explain the difference between TF-IDF and BM25.",
    "Give a short tutorial on how speculative decoding works.",
    "Write a concise explanation of RoPE and why it encodes relative position information.",
    "Explain why residual connections help deep neural networks train.",
]

DEFAULT_PROMPTS = [
    "Draft a professional email asking for a one-week extension on a project deadline because integration testing uncovered unexpected issues.",
    "Revise the previous email to sound warmer and more collaborative.",
    "Revise the previous email again to be more concise while preserving the tone.",
    "Revise the previous email for an executive audience.",

    "Write a Python function that loads a JSONL file, filters rows by a field called 'score', and returns the filtered rows.",
    "Revise the previous code to include type hints and docstrings.",
    "Revise the previous code to stream rows instead of loading everything into memory.",
    "Revise the previous code to use pathlib and add basic error handling.",

    "Return a JSON object comparing PyTorch and JAX with keys: overview, strengths, weaknesses, ideal_use_cases.",
    "Revise the previous JSON to focus on large-scale distributed training.",
    "Revise the previous JSON to focus on research prototyping.",
    "Keep the same JSON schema but compare TensorFlow and PyTorch instead.",

    "Write a technical design note for a tool that measures per-layer activation memory during transformer inference.",
    "Revise the previous design note to include a section on risks and limitations.",
    "Revise the previous design note to be shorter and more suitable for an internal RFC.",
    "Revise the previous design note to emphasize implementation milestones.",
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
    a_text: str
    b_text: str
    exact_match: bool
    norm_edit_similarity: float
    rouge_l_f1: float
    a_len: int
    b_len: int
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
    a_results: list[dict[str, Any]],
    b_results: list[dict[str, Any]],
) -> tuple[list[PairwiseQuality], dict[str, Any]]:
    if len(a_results) != len(b_results):
        raise ValueError("Result files have different numbers of prompts.")

    pairs: list[PairwiseQuality] = []

    for a, b in zip(a_results, b_results):
        if a["idx"] != b["idx"]:
            raise ValueError("Prompt ordering mismatch between result files.")

        a_text = a["output_text"]
        b_text = b["output_text"]
        sim = norm_edit_similarity(a_text, b_text)
        rouge = rouge_l_f1(a_text, b_text)
        len_ratio = len(b_text) / max(len(a_text), 1)

        pairs.append(
            PairwiseQuality(
                idx=a["idx"],
                prompt=a["prompt"],
                a_text=a_text,
                b_text=b_text,
                exact_match=(a_text == b_text),
                norm_edit_similarity=sim,
                rouge_l_f1=rouge,
                a_len=len(a_text),
                b_len=len(b_text),
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


def run_mode(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts_file)
    print(f"[INFO] Loaded {len(prompts)} prompts")
    print(f"[INFO] Running against {args.base_url}")
    print(f"[INFO] Batch size: {args.batch_size}")

    results = run_batched_eval(
        prompts=prompts,
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
            "num_prompts": len(prompts),
        },
        "summary": summary,
        "results": [asdict(x) for x in results],
    }

    Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote run results to {args.output_json}")


def compare_mode(args: argparse.Namespace) -> None:
    a = json.loads(Path(args.file_a).read_text())
    b = json.loads(Path(args.file_b).read_text())

    a_summary = a["summary"]
    b_summary = b["summary"]
    a_results = a["results"]
    b_results = b["results"]

    quality_rows, quality_summary = compare_quality(a_results, b_results)

    relative = {
        "avg_latency_speedup_a_over_b": (
            a_summary["avg_latency_s"] / b_summary["avg_latency_s"]
            if b_summary["avg_latency_s"] > 0 else float("nan")
        ),
        "aggregate_completion_tps_gain_b_over_a": (
            b_summary["aggregate_completion_toks_per_s"] / a_summary["aggregate_completion_toks_per_s"]
            if a_summary["aggregate_completion_toks_per_s"] > 0 else float("nan")
        ),
    }

    payload = {
        "file_a": args.file_a,
        "file_b": args.file_b,
        "summary_a": a_summary,
        "summary_b": b_summary,
        "relative": relative,
        "quality_summary": quality_summary,
        "quality_rows": [asdict(x) for x in quality_rows],
    }

    print("\n=== A SUMMARY ===")
    print(json.dumps(a_summary, indent=2))
    print("\n=== B SUMMARY ===")
    print(json.dumps(b_summary, indent=2))
    print("\n=== RELATIVE ===")
    print(json.dumps(relative, indent=2))
    print("\n=== QUALITY SUMMARY ===")
    print(json.dumps(quality_summary, indent=2))

    worst = sorted(quality_rows, key=lambda x: (x.rouge_l_f1, x.norm_edit_similarity))[:5]
    print("\n=== WORST 5 QUALITY CASES ===")
    for row in worst:
        print(f"\n--- idx={row.idx} ---")
        print(f"Prompt: {row.prompt}")
        print(f"Exact match: {row.exact_match}")
        print(f"Norm edit similarity: {row.norm_edit_similarity:.4f}")
        print(f"ROUGE-L F1: {row.rouge_l_f1:.4f}")
        print(f"\n[A]\n{row.a_text}")
        print(f"\n[B]\n{row.b_text}")

    Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(f"\n[INFO] Wrote comparison results to {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--base-url", type=str, required=True)
    run_parser.add_argument("--model", type=str, required=True)
    run_parser.add_argument("--model-label", type=str, required=True)
    run_parser.add_argument("--prompts-file", type=str, default=None)
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--timeout-s", type=float, default=180.0)
    run_parser.add_argument("--output-json", type=str, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--file-a", type=str, required=True)
    compare_parser.add_argument("--file-b", type=str, required=True)
    compare_parser.add_argument("--output-json", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "run":
        run_mode(args)
    else:
        compare_mode(args)


if __name__ == "__main__":
    main()