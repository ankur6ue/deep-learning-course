#!/usr/bin/env python3
"""
Minimal attention benchmark harness.

Goals
-----
- Tutorial-first: small amount of code, easy to read.
- Benchmark-first: capture stable timing and correctness checks.
- Minimal dependencies: torch is required, flashinfer is optional.
- Hardware agnostic: runs on whatever CUDA GPU is present and reports capabilities.

First benchmark cases
---------------------
1. dense causal prefill
2. single-token decode with contiguous KV
3. single-token decode with paged KV
4. GQA decode

First backend set
-----------------
- torch_reference
- flashinfer (if importable and shape supported)

Notes
-----
- This is intentionally forward-only and BF16-only for the first version.
- The paged KV path is backend-specific in practice. For v1, torch and Triton treat
  paged KV as a gather into contiguous K/V before attention. FlashInfer can use its
  native paged path when available.
- The Triton path here is a placeholder wrapper because Triton's official fused
  attention tutorial is not packaged as a stable pip API. The harness is written so
  you can drop in a local Triton kernel later.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import platform
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")



def gpu_info() -> Dict[str, Any]:
    require_cuda()
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "device_index": idx,
        "device_name": props.name,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


# -----------------------------------------------------------------------------
# Benchmark case definitions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionCase:
    name: str
    batch_size: int
    q_len: int
    kv_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool
    paged_kv: bool
    page_size: int = 256
    dtype: torch.dtype = torch.bfloat16

    @property
    def is_decode(self) -> bool:
        return self.q_len == 1

    @property
    def is_gqa(self) -> bool:
        return self.num_q_heads != self.num_kv_heads

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["dtype"] = str(self.dtype).replace("torch.", "")
        return out



def default_cases() -> List[AttentionCase]:
    return [
        AttentionCase(
            name="dense_causal_prefill",
            batch_size=4,
            q_len=2048,
            kv_len=2048,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=128,
            causal=True,
            paged_kv=False,
        ),
        AttentionCase(
            name="single_token_decode_contiguous_kv",
            batch_size=32,
            q_len=1,
            kv_len=8192,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=128,
            causal=True,
            paged_kv=False,
        ),
        AttentionCase(
            name="single_token_decode_paged_kv",
            batch_size=32,
            q_len=1,
            kv_len=8192,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=128,
            causal=True,
            paged_kv=True,
            page_size=256,
        ),
        AttentionCase(
            name="gqa_decode",
            batch_size=32,
            q_len=1,
            kv_len=8192,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=128,
            causal=True,
            paged_kv=False,
        ),
    ]


# -----------------------------------------------------------------------------
# Tensor generation
# -----------------------------------------------------------------------------


@dataclass
class AttentionInputs:
    q: torch.Tensor               # [B, Hq, Tq, D]
    k: torch.Tensor               # [B, Hkv, Tk, D]
    v: torch.Tensor               # [B, Hkv, Tk, D]
    k_pages: Optional[torch.Tensor] = None
    v_pages: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    valid_lens: Optional[torch.Tensor] = None



def make_qkv(case: AttentionCase, device: torch.device) -> AttentionInputs:
    B, Tq, Tk = case.batch_size, case.q_len, case.kv_len
    Hq, Hkv, D = case.num_q_heads, case.num_kv_heads, case.head_dim
    dtype = case.dtype

    q = torch.randn(B, Hq, Tq, D, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, Tk, D, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, Tk, D, device=device, dtype=dtype)

    if not case.paged_kv:
        return AttentionInputs(q=q, k=k, v=v)

    page_size = case.page_size
    assert Tk % page_size == 0, "kv_len must be divisible by page_size in v1"
    pages_per_seq = Tk // page_size
    total_pages = B * pages_per_seq

    # Physical page storage: [num_pages, Hkv, page_size, D]
    k_pages = k.view(B, Hkv, pages_per_seq, page_size, D).permute(0, 2, 1, 3, 4).contiguous()
    v_pages = v.view(B, Hkv, pages_per_seq, page_size, D).permute(0, 2, 1, 3, 4).contiguous()
    k_pages = k_pages.view(total_pages, Hkv, page_size, D)
    v_pages = v_pages.view(total_pages, Hkv, page_size, D)

    page_table = torch.arange(total_pages, device=device, dtype=torch.int32).view(B, pages_per_seq)
    valid_lens = torch.full((B,), Tk, device=device, dtype=torch.int32)

    return AttentionInputs(
        q=q,
        k=k,
        v=v,
        k_pages=k_pages,
        v_pages=v_pages,
        page_table=page_table,
        valid_lens=valid_lens,
    )


# -----------------------------------------------------------------------------
# Reference attention math
# -----------------------------------------------------------------------------



def repeat_kv_for_gqa(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    # x: [B, Hkv, T, D] -> [B, Hq, T, D]
    B, Hkv, T, D = x.shape
    if Hkv == num_q_heads:
        return x
    if num_q_heads % Hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA.")
    repeat = num_q_heads // Hkv
    return x.repeat_interleave(repeat, dim=1)



def causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    # True means keep, False means mask out.
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    # Decode and prefill are both handled. For decode, q_len=1 and this becomes all True.
    return k_idx <= (kv_len - q_len + q_idx)



def paged_to_contiguous(
    pages: torch.Tensor,
    page_table: torch.Tensor,
    valid_lens: torch.Tensor,
) -> torch.Tensor:
    # pages: [num_pages, Hkv, page_size, D]
    B, pages_per_seq = page_table.shape
    _, Hkv, page_size, D = pages.shape
    out_list = []
    for b in range(B):
        idx = page_table[b].long()
        seq = pages.index_select(0, idx).permute(1, 0, 2, 3).contiguous()  # [Hkv, P, page, D]
        seq = seq.view(Hkv, pages_per_seq * page_size, D)
        out_list.append(seq[:, : int(valid_lens[b].item()), :])
    max_len = max(x.shape[1] for x in out_list)
    out = torch.zeros(B, Hkv, max_len, D, device=pages.device, dtype=pages.dtype)
    for b, x in enumerate(out_list):
        out[b, :, : x.shape[1], :] = x
    return out



def torch_reference_attention(case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
    q = inp.q
    if case.paged_kv:
        k = paged_to_contiguous(inp.k_pages, inp.page_table, inp.valid_lens)
        v = paged_to_contiguous(inp.v_pages, inp.page_table, inp.valid_lens)
    else:
        k = inp.k
        v = inp.v

    k = repeat_kv_for_gqa(k, case.num_q_heads)
    v = repeat_kv_for_gqa(v, case.num_q_heads)

    scale = 1.0 / math.sqrt(case.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, Hq, Tq, Tk]

    if case.causal:
        mask = causal_mask(case.q_len, k.shape[-2], q.device)
        scores = scores.masked_fill(~mask.view(1, 1, case.q_len, k.shape[-2]), float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(probs, v)  # [B, Hq, Tq, D]
    return out


# -----------------------------------------------------------------------------
# Backend wrappers
# -----------------------------------------------------------------------------


class AttentionBackend:
    name = "base"

    def available(self) -> Tuple[bool, str]:
        raise NotImplementedError

    def supports(self, case: AttentionCase) -> Tuple[bool, str]:
        return True, "ok"

    def run(self, case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
        raise NotImplementedError


class TorchReferenceBackend(AttentionBackend):
    name = "torch_reference"

    def available(self) -> Tuple[bool, str]:
        return True, "builtin"

    def run(self, case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
        return torch_reference_attention(case, inp)


class FlashInferBackend(AttentionBackend):
    name = "flashinfer"

    def __init__(self) -> None:
        self.mod = None
        self._ragged_prefill_plan_cache: Dict[Tuple[Any, ...], Tuple[Any, torch.Tensor]] = {}
        # Cache the working prefill path per shape so benchmark iterations do not
        # repeatedly pay exception/fallback overhead.
        self._ragged_prefill_backend_choice: Dict[Tuple[Any, ...], str] = {}

    def available(self) -> Tuple[bool, str]:
        try:
            self.mod = importlib.import_module("flashinfer")
            return True, f"flashinfer {getattr(self.mod, '__version__', 'unknown')}"
        except Exception as exc:
            return False, f"flashinfer import failed: {exc}"

    def supports(self, case: AttentionCase) -> Tuple[bool, str]:
        if case.dtype != torch.bfloat16:
            return False, "v1 only benchmarks BF16"
        return True, "ok"

    def run(self, case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
        if self.mod is None:
            raise RuntimeError("flashinfer module is not initialized")

        decode_mod = getattr(self.mod, "decode", None)
        prefill_mod = getattr(self.mod, "prefill", None)
        if decode_mod is None or prefill_mod is None:
            raise RuntimeError("flashinfer.decode or flashinfer.prefill module not found")

        def is_mixed_cudart_error(exc: Exception) -> bool:
            return "Multiple libcudart libraries found" in str(exc)

        def run_paged_decode_fallback() -> torch.Tensor:
            single_decode_fn = getattr(decode_mod, "single_decode_with_kv_cache", None)
            if single_decode_fn is None:
                raise RuntimeError(
                    "flashinfer.decode.single_decode_with_kv_cache not found "
                    "(needed as fallback when cuDNN paged decode is unavailable)"
                )
            if inp.k_pages is None or inp.v_pages is None or inp.page_table is None or inp.valid_lens is None:
                raise RuntimeError("paged KV inputs are incomplete")

            out_list = []
            for b in range(case.batch_size):
                q_b = inp.q[b, :, 0, :]  # [Hq, D]
                page_idx = inp.page_table[b].long()
                kv_len = int(inp.valid_lens[b].item())

                k_b = inp.k_pages.index_select(0, page_idx).permute(1, 0, 2, 3).contiguous()
                v_b = inp.v_pages.index_select(0, page_idx).permute(1, 0, 2, 3).contiguous()
                k_b = k_b.view(case.num_kv_heads, -1, case.head_dim)[:, :kv_len, :].permute(1, 0, 2).contiguous()
                v_b = v_b.view(case.num_kv_heads, -1, case.head_dim)[:, :kv_len, :].permute(1, 0, 2).contiguous()

                out_b = single_decode_fn(
                    q=q_b,
                    k=k_b,
                    v=v_b,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                )
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                out_list.append(out_b)
            return torch.stack(out_list, dim=0).unsqueeze(2)  # [B, Hq, 1, D]

        def run_single_prefill_fallback() -> torch.Tensor:
            single_prefill_fn = getattr(prefill_mod, "single_prefill_with_kv_cache", None)
            if single_prefill_fn is None:
                raise RuntimeError(
                    "flashinfer.prefill.single_prefill_with_kv_cache not found "
                    "(needed as fallback when cuDNN prefill is unavailable)"
                )

            out_list = []
            for b in range(case.batch_size):
                q_b = inp.q[b].permute(1, 0, 2).contiguous()  # [Tq, Hq, D]
                k_b = inp.k[b].permute(1, 0, 2).contiguous()  # [Tk, Hkv, D]
                v_b = inp.v[b].permute(1, 0, 2).contiguous()  # [Tk, Hkv, D]
                out_b = single_prefill_fn(
                    q=q_b,
                    k=k_b,
                    v=v_b,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    causal=case.causal,
                )
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                out_list.append(out_b)
            return torch.stack(out_list, dim=0)

        def run_ragged_batch_prefill_wrapper(backend_name: str) -> torch.Tensor:
            wrapper_cls = getattr(self.mod, "BatchPrefillWithRaggedKVCacheWrapper", None)
            if wrapper_cls is None:
                raise RuntimeError("flashinfer.BatchPrefillWithRaggedKVCacheWrapper not found")

            B = case.batch_size
            Tq = case.q_len
            Tk = case.kv_len
            cache_key = (
                str(inp.q.device),
                B,
                Tq,
                Tk,
                case.num_q_heads,
                case.num_kv_heads,
                case.head_dim,
                case.causal,
                str(case.dtype),
                backend_name,
            )
            if cache_key not in self._ragged_prefill_plan_cache:
                workspace = torch.empty(128 * 1024 * 1024, device=inp.q.device, dtype=torch.uint8)
                wrapper = wrapper_cls(workspace, "NHD", backend=backend_name)
                qo_indptr = torch.arange(0, (B + 1) * Tq, Tq, device=inp.q.device, dtype=torch.int32)
                kv_indptr = torch.arange(0, (B + 1) * Tk, Tk, device=inp.q.device, dtype=torch.int32)
                # FlashInfer plan docs specify seq_lens/seq_lens_q as uint32.
                seq_lens_q = torch.full((B,), Tq, device=inp.q.device, dtype=torch.uint32)
                seq_lens_kv = torch.full((B,), Tk, device=inp.q.device, dtype=torch.uint32)
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=case.num_q_heads,
                    num_kv_heads=case.num_kv_heads,
                    head_dim_qk=case.head_dim,
                    causal=case.causal,
                    q_data_type=case.dtype,
                    kv_data_type=case.dtype,
                    o_data_type=case.dtype,
                    seq_lens=seq_lens_kv,
                    seq_lens_q=seq_lens_q,
                    max_token_per_sequence=Tq,
                    max_sequence_kv=Tk,
                )
                self._ragged_prefill_plan_cache[cache_key] = (wrapper, workspace)

            wrapper, _workspace = self._ragged_prefill_plan_cache[cache_key]
            q = inp.q.permute(0, 2, 1, 3).reshape(B * Tq, case.num_q_heads, case.head_dim).contiguous()
            k = inp.k.permute(0, 2, 1, 3).reshape(B * Tk, case.num_kv_heads, case.head_dim).contiguous()
            v = inp.v.permute(0, 2, 1, 3).reshape(B * Tk, case.num_kv_heads, case.head_dim).contiguous()
            out = wrapper.run(q=q, k=k, v=v, return_lse=False)
            if isinstance(out, tuple):
                out = out[0]
            return out.reshape(B, Tq, case.num_q_heads, case.head_dim).permute(0, 2, 1, 3).contiguous()

        if case.paged_kv:
            if not case.is_decode:
                raise RuntimeError("v1 FlashInfer paged path is only implemented for decode")

            batch_decode_fn = getattr(decode_mod, "cudnn_batch_decode_with_kv_cache", None)
            if batch_decode_fn is None:
                return run_paged_decode_fallback()

            if inp.k_pages is None or inp.v_pages is None or inp.page_table is None or inp.valid_lens is None:
                raise RuntimeError("paged KV inputs are incomplete")

            q = inp.q[:, :, 0, :]  # [B, Hq, D]

            actual_seq_lens_kv = inp.valid_lens.to(dtype=torch.int32, device="cpu")
            block_tables = inp.page_table.to(dtype=torch.int32)

            try:
                out = batch_decode_fn(
                    q=q,
                    k_cache=inp.k_pages,
                    v_cache=inp.v_pages,
                    actual_seq_lens_kv=actual_seq_lens_kv,
                    block_tables=block_tables,
                )
            except Exception as exc:
                if not is_mixed_cudart_error(exc):
                    raise
                return run_paged_decode_fallback()
            return out.unsqueeze(2)  # [B, Hq, 1, D]

        if case.is_decode:
            single_decode_fn = getattr(decode_mod, "single_decode_with_kv_cache", None)
            if single_decode_fn is None:
                raise RuntimeError("flashinfer.decode.single_decode_with_kv_cache not found")

            out_list = []
            for b in range(case.batch_size):
                q_b = inp.q[b, :, 0, :]             # [Hq, D]
                k_b = inp.k[b].permute(1, 0, 2)     # [Tk, Hkv, D]
                v_b = inp.v[b].permute(1, 0, 2)     # [Tk, Hkv, D]

                out_b = single_decode_fn(
                    q=q_b,
                    k=k_b,
                    v=v_b,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                )
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                out_list.append(out_b)

            return torch.stack(out_list, dim=0).unsqueeze(2)  # [B, Hq, 1, D]

        prefill_runtime_key = (
            str(inp.q.device),
            case.batch_size,
            case.q_len,
            case.kv_len,
            case.num_q_heads,
            case.num_kv_heads,
            case.head_dim,
            case.causal,
            str(case.dtype),
        )
        chosen_backend = self._ragged_prefill_backend_choice.get(prefill_runtime_key)
        if chosen_backend == "single_prefill":
            out = run_single_prefill_fallback()
            return out.permute(0, 2, 1, 3).contiguous()
        if chosen_backend in {"cudnn", "auto"}:
            return run_ragged_batch_prefill_wrapper(chosen_backend)

        try:
            out = run_ragged_batch_prefill_wrapper("cudnn")
            self._ragged_prefill_backend_choice[prefill_runtime_key] = "cudnn"
            return out
        except Exception as cudnn_exc:
            try:
                out = run_ragged_batch_prefill_wrapper("auto")
                print(f"can't use batch prefill cudnn ({cudnn_exc}).. using auto")
                self._ragged_prefill_backend_choice[prefill_runtime_key] = "auto"
                return out
            except Exception as auto_exc:
                print(
                    f"can't use batch prefill cudnn ({cudnn_exc}) and auto ({auto_exc}).. "
                    "reverting to single prefill"
                )
                self._ragged_prefill_backend_choice[prefill_runtime_key] = "single_prefill"
                out = run_single_prefill_fallback()
                return out.permute(0, 2, 1, 3).contiguous()


# -----------------------------------------------------------------------------
# Benchmark engine
# -----------------------------------------------------------------------------


@dataclass
class TimingStats:
    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    num_iters: int


@dataclass
class RunRecord:
    case: Dict[str, Any]
    backend: str
    availability: str
    support: str
    status: str
    timing: Optional[TimingStats] = None
    max_abs_err: Optional[float] = None
    mean_abs_err: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if self.timing is not None:
            out["timing"] = dataclasses.asdict(self.timing)
        return out



def cuda_timeit(fn, warmup: int, iters: int) -> TimingStats:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        _ = fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    return TimingStats(
        median_ms=statistics.median(times),
        mean_ms=float(statistics.mean(times)),
        p90_ms=float(sorted(times)[max(0, int(0.9 * len(times)) - 1)]),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
        num_iters=iters,
    )



def compare_outputs(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())



def benchmark_backend_case(
    backend: AttentionBackend,
    case: AttentionCase,
    inp: AttentionInputs,
    warmup: int,
    iters: int,
    check_correctness: bool = True,
) -> RunRecord:
    avail, avail_msg = backend.available()
    if not avail:
        return RunRecord(
            case=case.to_dict(),
            backend=backend.name,
            availability=avail_msg,
            support="n/a",
            status="skipped",
            note="backend unavailable",
        )

    supported, support_msg = backend.supports(case)
    if not supported:
        return RunRecord(
            case=case.to_dict(),
            backend=backend.name,
            availability=avail_msg,
            support=support_msg,
            status="skipped",
            note="backend does not support this case",
        )

    def fn():
        return backend.run(case, inp)

    try:
        max_abs_err = None
        mean_abs_err = None
        out = fn()
        if check_correctness:
            ref_out = torch_reference_attention(case, inp)
            max_abs_err, mean_abs_err = compare_outputs(out, ref_out)
            del ref_out
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del out
        torch.cuda.synchronize()
        timing = cuda_timeit(fn, warmup=warmup, iters=iters)
        return RunRecord(
            case=case.to_dict(),
            backend=backend.name,
            availability=avail_msg,
            support=support_msg,
            status="ok",
            timing=timing,
            max_abs_err=max_abs_err,
            mean_abs_err=mean_abs_err,
        )
    except Exception as exc:
        return RunRecord(
            case=case.to_dict(),
            backend=backend.name,
            availability=avail_msg,
            support=support_msg,
            status="error",
            note=str(exc),
        )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------



def print_header() -> None:
    info = gpu_info()
    print("=== Environment ===")
    for k, v in info.items():
        print(f"{k:>20}: {v}")
    print()



def print_case(case: AttentionCase) -> None:
    print(f"=== Case: {case.name} ===")
    for k, v in case.to_dict().items():
        print(f"{k:>20}: {v}")
    print()



def print_record(rec: RunRecord) -> None:
    status = rec.status.upper()
    print(f"[{status:>7}] backend={rec.backend}")
    print(f"         availability: {rec.availability}")
    print(f"             support: {rec.support}")
    if rec.timing is not None:
        print(f"           median ms: {rec.timing.median_ms:.3f}")
        print(f"             mean ms: {rec.timing.mean_ms:.3f}")
        print(f"              p90 ms: {rec.timing.p90_ms:.3f}")
        print(f"              min ms: {rec.timing.min_ms:.3f}")
        print(f"              max ms: {rec.timing.max_ms:.3f}")
    if rec.max_abs_err is not None:
        print(f"         max abs err: {rec.max_abs_err:.6f}")
        print(f"        mean abs err: {rec.mean_abs_err:.6f}")
    if rec.note:
        print(f"                note: {rec.note}")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="all",
                        choices=["all", "dense_causal_prefill", "single_token_decode_contiguous_kv",
                                 "single_token_decode_paged_kv", "gqa_decode"])
    parser.add_argument("--json_out", type=str, default="")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    require_cuda()
    set_seed(args.seed)
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print_header()

    cases = default_cases()
    if args.case != "all":
        cases = [c for c in cases if c.name == args.case]

    backends: List[AttentionBackend] = [
        TorchReferenceBackend(),
        FlashInferBackend(),
    ]

    all_records: List[RunRecord] = []

    for case in cases:
        print_case(case)
        inp = make_qkv(case, device=device)
        for backend in backends:
            rec = benchmark_backend_case(
                backend=backend,
                case=case,
                inp=inp,
                warmup=args.warmup,
                iters=args.iters,
                check_correctness=(backend.name != "torch_reference"),
            )
            print_record(rec)
            all_records.append(rec)
        del inp
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if args.json_out:
        payload = {
            "environment": gpu_info(),
            "records": [r.to_dict() for r in all_records],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote results to {args.json_out}")


if __name__ == "__main__":
    main()
