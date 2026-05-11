#!/usr/bin/env python3
"""
Benchmark vLLM generation latency on gpt-oss-20b across attention backends.

Focus metric:
- total generation time for ~N output tokens (default N=50)

Notes:
- Uses chat mode with gpt-oss Harmony template.
- Measures model init time separately from generation time.
- Continues on backend failures and reports them.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from vllm import LLM, SamplingParams

try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    VALID_BACKEND_NAMES = {x.name for x in AttentionBackendEnum}
except Exception:
    VALID_BACKEND_NAMES = set()


@dataclass
class BackendResult:
    backend: str
    status: str
    note: str
    init_time_s: Optional[float] = None
    run_times_s: Optional[List[float]] = None
    generated_tokens: Optional[List[int]] = None
    median_time_s: Optional[float] = None
    mean_time_s: Optional[float] = None
    p90_time_s: Optional[float] = None
    median_toks_per_s: Optional[float] = None
    sample_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ankur/dev/models/gpt-oss-20b",
        help="Local model path or HF model id.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help=(
            "Optional chat template path. If omitted, uses "
            "<model>/chat_template.jinja when present."
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="auto,FLASH_ATTN,TRITON_ATTN,FLASHINFER",
        help="Comma-separated backend names. Use 'auto' for default selection.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per backend.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Target generation tokens (approx; exact if --ignore-eos).",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=True,
        help="Ignore EOS to force generation up to max_tokens (default: on).",
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        default=False,
        help="Stop at EOS if generated before max_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Harmony reasoning effort (used for GPT-OSS templates).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain the bias-variance tradeoff in machine learning in about 5 sentences. "
            "Be concrete and concise."
        ),
        help="User prompt content.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--auto-model-fallback",
        action="store_true",
        default=False,
        help=(
            "Enable automatic fallback to compatible sibling model dirs "
            "(used for some Ministral checkpoints). Default is off."
        ),
    )
    parser.add_argument(
        "--auto-shrink-max-model-len",
        action="store_true",
        default=True,
        help=(
            "If model init fails with KV-cache memory errors, retry with "
            "the estimated maximum model length from vLLM error message."
        ),
    )
    parser.add_argument(
        "--no-auto-shrink-max-model-len",
        action="store_true",
        default=False,
        help="Disable automatic max_model_len retry logic.",
    )
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def normalize_backend_name(raw: str) -> str:
    name = raw.strip()
    if not name:
        raise ValueError("Empty backend name.")
    if name.lower() == "auto":
        return "auto"
    name = name.replace("-", "_").upper()
    if VALID_BACKEND_NAMES and name not in VALID_BACKEND_NAMES:
        valid = ", ".join(sorted(VALID_BACKEND_NAMES))
        raise ValueError(f"Unknown backend '{raw}'. Valid backends: {valid}, auto")
    return name


def build_sampling_params(args: argparse.Namespace, max_tokens: Optional[int] = None) -> SamplingParams:
    out_tokens = args.max_tokens if max_tokens is None else max_tokens
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_tokens=out_tokens,
        max_tokens=out_tokens,
        ignore_eos=args.ignore_eos,
        skip_special_tokens=True,
    )


def build_messages(args: argparse.Namespace) -> List[Dict[str, str]]:
    return [{"role": "user", "content": args.prompt}]


def detect_model_family(model_ref: str) -> str:
    lower = model_ref.lower()
    if "gpt-oss" in lower:
        return "gpt_oss"
    if "ministral" in lower or "mistral" in lower:
        return "ministral"
    return "generic"


def _load_model_config_json(model_ref: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(model_ref):
        return None
    cfg_path = os.path.join(model_ref, "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def detect_quantization_method(model_ref: str) -> Optional[str]:
    cfg = _load_model_config_json(model_ref)
    if not isinstance(cfg, dict):
        return None

    top_qcfg = cfg.get("quantization_config")
    if isinstance(top_qcfg, dict):
        q = top_qcfg.get("quant_method")
        if isinstance(q, str) and q:
            return q.lower()

    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        text_qcfg = text_cfg.get("quantization_config")
        if isinstance(text_qcfg, dict):
            q = text_qcfg.get("quant_method")
            if isinstance(q, str) and q:
                return q.lower()
    return None


def resolve_model_path(
    model_ref: str, model_family: str, enable_fallback: bool
) -> tuple[str, Optional[str]]:
    if not enable_fallback:
        return model_ref, None
    if model_family != "ministral":
        return model_ref, None
    if not os.path.isdir(model_ref):
        return model_ref, None

    cfg = _load_model_config_json(model_ref)
    qcfg = (cfg or {}).get("quantization_config")
    quant_method = qcfg.get("quant_method") if isinstance(qcfg, dict) else None
    if str(quant_method).lower() != "fp8":
        return model_ref, None

    # Prefer patched BF16 sibling, then BF16 sibling.
    candidates = [f"{model_ref}-BF16-PATCHED-457", f"{model_ref}-BF16"]
    for cand in candidates:
        if os.path.isdir(cand) and _load_model_config_json(cand) is not None:
            note = (
                f"auto-fallback: using compatible model dir '{cand}' instead of "
                f"'{model_ref}' (original has fp8 quantization config)."
            )
            return cand, note
    return model_ref, None


def resolve_chat_template_path(model_ref: str, chat_template_arg: str) -> Optional[str]:
    if chat_template_arg:
        return chat_template_arg

    # If model_ref is a local directory, try <model_ref>/chat_template.jinja.
    if os.path.isdir(model_ref):
        candidate = os.path.join(model_ref, "chat_template.jinja")
        if os.path.exists(candidate):
            return candidate
    return None


def build_chat_template_kwargs(model_family: str, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if model_family == "gpt_oss":
        return {"reasoning_effort": args.reasoning_effort}
    return None


def patch_ministral_hf_config(hf_config: Any) -> Any:
    # vLLM may route through hf_config.text_config for mistral3 models.
    # Some Ministral configs have text_config.architectures=None, which
    # triggers a TypeError during architecture selection. Patch it in-memory.
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None:
        # For some Ministral checkpoints, quantization config is only present
        # at top-level. Copy it down so text tower allocates expected FP8 params.
        top_qcfg = getattr(hf_config, "quantization_config", None)
        text_qcfg = getattr(text_cfg, "quantization_config", None)
        if text_qcfg is None and top_qcfg is not None:
            text_cfg.quantization_config = top_qcfg

        text_arch = getattr(text_cfg, "architectures", None)
        text_model_type = getattr(text_cfg, "model_type", "")
        if text_arch is None and str(text_model_type).lower() in {"ministral3", "mistral3", "mistral"}:
            text_cfg.architectures = ["MistralForCausalLM"]
    return hf_config


def build_hf_overrides(model_family: str) -> Optional[Any]:
    if model_family != "ministral":
        return None
    return patch_ministral_hf_config


def _extract_text_and_tokens(output: Any) -> tuple[str, int]:
    if not output or not output.outputs:
        return "", 0
    first = output.outputs[0]
    text = first.text if hasattr(first, "text") else ""
    token_ids = first.token_ids if hasattr(first, "token_ids") else []
    return text, len(token_ids)


def extract_estimated_max_len_from_error(msg: str) -> Optional[int]:
    m = re.search(r"estimated maximum model length is (\d+)", msg, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def run_backend(
    backend: str,
    args: argparse.Namespace,
    resolved_model: str,
    model_family: str,
    resolved_chat_template: Optional[str],
) -> BackendResult:
    llm = None
    result = BackendResult(backend=backend, status="error", note="uninitialized")
    try:
        effective_max_model_len = int(args.max_model_len)
        quant_method = detect_quantization_method(resolved_model)
        llm_kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "tokenizer": resolved_model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": effective_max_model_len,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
        }
        if quant_method is not None:
            llm_kwargs["quantization"] = quant_method
        hf_overrides = build_hf_overrides(model_family)
        if hf_overrides is not None:
            llm_kwargs["hf_overrides"] = hf_overrides
        if resolved_chat_template:
            llm_kwargs["chat_template"] = resolved_chat_template
        if backend != "auto":
            llm_kwargs["attention_config"] = {"backend": backend}

        init_attempts = 0
        init_start = time.perf_counter()
        while True:
            init_attempts += 1
            try:
                llm_kwargs["max_model_len"] = effective_max_model_len
                llm = LLM(**llm_kwargs)
                break
            except Exception as exc:
                if llm is not None:
                    del llm
                    llm = None
                msg = str(exc)
                if not args.auto_shrink_max_model_len:
                    raise
                est = extract_estimated_max_len_from_error(msg)
                # Keep one request feasible: prompt + generation + small buffer.
                min_needed = max(32, int(args.max_tokens) + 16)
                if est is None or est <= 0:
                    raise
                new_max = max(min_needed, min(est, effective_max_model_len - 1))
                if new_max >= effective_max_model_len:
                    raise
                effective_max_model_len = new_max
                # Retry with reduced max model len.
                continue

        init_end = time.perf_counter()
        result.init_time_s = init_end - init_start

        # Ensure generation budget fits in the available sequence budget.
        effective_max_tokens = min(args.max_tokens, max(1, effective_max_model_len - 16))
        sampling_params = build_sampling_params(args, max_tokens=effective_max_tokens)
        messages = build_messages(args)
        chat_kwargs = {
            "messages": messages,
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
        template_kwargs = build_chat_template_kwargs(model_family, args)
        if template_kwargs:
            chat_kwargs["chat_template_kwargs"] = template_kwargs

        for _ in range(args.warmup):
            _ = llm.chat(**chat_kwargs)

        run_times: List[float] = []
        token_counts: List[int] = []
        sample_text = ""
        for i in range(args.runs):
            t0 = time.perf_counter()
            out = llm.chat(**chat_kwargs)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            run_times.append(elapsed)

            text, tok_count = _extract_text_and_tokens(out[0] if out else None)
            token_counts.append(tok_count)
            if i == 0:
                sample_text = text

        sorted_times = sorted(run_times)
        p90_index = max(0, int(0.9 * len(sorted_times)) - 1)
        median_time = statistics.median(sorted_times)
        mean_time = statistics.mean(sorted_times)
        p90_time = sorted_times[p90_index]
        median_tok_count = statistics.median(token_counts) if token_counts else 0
        median_toks_per_s = (
            (median_tok_count / median_time) if median_time > 0 and median_tok_count > 0 else None
        )

        result.status = "ok"
        result.note = (
            "ok"
            if effective_max_model_len == args.max_model_len
            else f"ok (max_model_len auto-reduced to {effective_max_model_len})"
        )
        result.run_times_s = run_times
        result.generated_tokens = token_counts
        result.median_time_s = median_time
        result.mean_time_s = mean_time
        result.p90_time_s = p90_time
        result.median_toks_per_s = median_toks_per_s
        result.sample_text = sample_text
        return result
    except Exception as exc:
        result.status = "error"
        result.note = str(exc)
        return result
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_env(
    args: argparse.Namespace,
    model_family: str,
    resolved_model: str,
    model_resolve_note: Optional[str],
    resolved_chat_template: Optional[str],
) -> None:
    quant_method = detect_quantization_method(resolved_model)
    print("=== Environment ===")
    print(f"model_requested: {args.model}")
    print(f"model_resolved: {resolved_model}")
    print(f"model_family: {model_family}")
    print(f"quantization: {quant_method or 'none (or auto)'}")
    print(f"chat_template: {resolved_chat_template or 'auto (tokenizer/default)'}")
    if model_resolve_note:
        print(model_resolve_note)
    print(
        "hf_overrides: "
        + (
            "ministral text_config.architectures auto-fix enabled"
            if model_family == "ministral"
            else "none"
        )
    )
    print(f"torch: {torch.__version__}")
    try:
        import vllm

        print(f"vllm: {vllm.__version__}")
    except Exception:
        print("vllm: unknown")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"device: {props.name} (cc {props.major}.{props.minor})")
    print()


def print_result(r: BackendResult) -> None:
    print(f"=== backend={r.backend} status={r.status} ===")
    if r.status != "ok":
        print(f"note: {r.note}")
        print()
        return
    print(f"init_time_s: {r.init_time_s:.3f}")
    print(f"run_times_s: {[round(x, 4) for x in (r.run_times_s or [])]}")
    print(f"generated_tokens: {r.generated_tokens}")
    print(f"median_time_s: {r.median_time_s:.4f}")
    print(f"mean_time_s: {r.mean_time_s:.4f}")
    print(f"p90_time_s: {r.p90_time_s:.4f}")
    if r.median_toks_per_s is not None:
        print(f"median_tokens_per_s: {r.median_toks_per_s:.2f}")
    if r.sample_text:
        preview = r.sample_text.replace("\n", " ").strip()
        print(f"sample_text_preview: {preview[:160]}")
    print()


def print_summary(results: List[BackendResult]) -> None:
    ok_results = [r for r in results if r.status == "ok" and r.median_time_s is not None]
    print("=== Summary (sorted by median generation time) ===")
    if not ok_results:
        print("No successful backend runs.")
        return
    ok_results.sort(key=lambda x: x.median_time_s or 1e9)
    print(
        f"{'backend':<16} {'median_s':>10} {'tokens/s':>10} {'init_s':>10} {'gen_tokens':>12}"
    )
    for r in ok_results:
        med_toks = int(statistics.median(r.generated_tokens)) if r.generated_tokens else 0
        tokps = r.median_toks_per_s if r.median_toks_per_s is not None else 0.0
        print(
            f"{r.backend:<16} {r.median_time_s:>10.4f} {tokps:>10.2f} "
            f"{(r.init_time_s or 0.0):>10.3f} {med_toks:>12d}"
        )


def main() -> None:
    args = parse_args()
    if args.respect_eos:
        args.ignore_eos = False
    if args.no_auto_shrink_max_model_len:
        args.auto_shrink_max_model_len = False
    model_family = detect_model_family(args.model)
    resolved_model, model_resolve_note = resolve_model_path(
        model_ref=args.model,
        model_family=model_family,
        enable_fallback=args.auto_model_fallback,
    )
    # Re-detect family from resolved model ref in case fallback changed naming.
    model_family = detect_model_family(resolved_model)
    resolved_chat_template = resolve_chat_template_path(resolved_model, args.chat_template)
    print_env(args, model_family, resolved_model, model_resolve_note, resolved_chat_template)

    if resolved_chat_template is not None:
        template_path = Path(resolved_chat_template)
        if not template_path.exists():
            raise FileNotFoundError(f"chat template not found: {resolved_chat_template}")

    requested = [x.strip() for x in args.backends.split(",") if x.strip()]
    if not requested:
        raise ValueError("No backends specified.")
    backends = [normalize_backend_name(x) for x in requested]

    results: List[BackendResult] = []
    for backend in backends:
        r = run_backend(
            backend=backend,
            args=args,
            resolved_model=resolved_model,
            model_family=model_family,
            resolved_chat_template=resolved_chat_template,
        )
        print_result(r)
        results.append(r)

    print_summary(results)

    if args.json_out:
        payload = {
            "args": vars(args),
            "results": [r.to_dict() for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
