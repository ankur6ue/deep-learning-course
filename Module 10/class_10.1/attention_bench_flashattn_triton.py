#!/usr/bin/env python3
"""
Minimal attention benchmark harness.

Goals
-----
- Tutorial-first: small amount of code, easy to read.
- Benchmark-first: capture stable timing and correctness checks.
- Minimal dependencies: torch is required, triton / flash-attn are optional.
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
- sdpa (PyTorch scaled_dot_product_attention, mem-efficient path)
- triton_reference (vLLM Triton unified attention kernel path)
- flash_attn (if importable and shape supported)

Notes
-----
- This is intentionally forward-only and BF16-only for the first version.
- The paged KV path is backend-specific in practice. `flash_attn` and
  `triton_reference` use paged-cache style kernels; `sdpa` currently skips paged
  cases in this harness.
- The `triton_reference` path in this file is wired to vLLM's Triton unified
  attention kernel implementation, while `sdpa` is explicitly PyTorch SDPA.
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
            name="chunked_decode_contiguous_kv",
            batch_size=32,
            q_len=64,
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
    vllm_triton_q: Optional[torch.Tensor] = None
    vllm_triton_k_cache: Optional[torch.Tensor] = None
    vllm_triton_v_cache: Optional[torch.Tensor] = None
    vllm_triton_block_table: Optional[torch.Tensor] = None
    vllm_triton_seq_lens: Optional[torch.Tensor] = None
    vllm_triton_cu_seqlens_q: Optional[torch.Tensor] = None
    vllm_triton_k_descale: Optional[torch.Tensor] = None
    vllm_triton_v_descale: Optional[torch.Tensor] = None



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



def prepare_vllm_triton_inputs(case: AttentionCase, inp: AttentionInputs) -> None:
    if inp.vllm_triton_q is not None:
        return

    B, Tq, Tk = case.batch_size, case.q_len, case.kv_len
    Hq, Hkv, D = case.num_q_heads, case.num_kv_heads, case.head_dim

    if Hq % Hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads for Triton GQA.")
    if case.page_size % 16 != 0:
        raise ValueError("vLLM Triton requires page_size to be a multiple of 16.")

    q_flat = inp.q.permute(0, 2, 1, 3).contiguous().view(B * Tq, Hq, D)
    cu_seqlens_q = torch.arange(
        0, (B + 1) * Tq, Tq, device=inp.q.device, dtype=torch.int32
    )

    if case.paged_kv:
        if inp.k_pages is None or inp.v_pages is None or inp.page_table is None or inp.valid_lens is None:
            raise RuntimeError("paged KV inputs are incomplete")
        # [num_pages, Hkv, page_size, D] -> [num_pages, page_size, Hkv, D]
        k_cache = inp.k_pages.permute(0, 2, 1, 3).contiguous()
        v_cache = inp.v_pages.permute(0, 2, 1, 3).contiguous()
        block_table = inp.page_table.to(dtype=torch.int32).contiguous()
        seq_lens = inp.valid_lens.to(dtype=torch.int32).contiguous()
    else:
        block_size = case.page_size
        if Tk % block_size != 0:
            raise ValueError("For non-paged Triton path, kv_len must be divisible by page_size.")
        blocks_per_seq = Tk // block_size
        total_blocks = B * blocks_per_seq

        # [B, Hkv, Tk, D] -> [num_blocks, block_size, Hkv, D]
        k_bt = inp.k.permute(0, 2, 1, 3).contiguous()
        v_bt = inp.v.permute(0, 2, 1, 3).contiguous()
        k_cache = k_bt.view(B, blocks_per_seq, block_size, Hkv, D).view(
            total_blocks, block_size, Hkv, D
        ).contiguous()
        v_cache = v_bt.view(B, blocks_per_seq, block_size, Hkv, D).view(
            total_blocks, block_size, Hkv, D
        ).contiguous()
        block_table = torch.arange(
            total_blocks, device=inp.q.device, dtype=torch.int32
        ).view(B, blocks_per_seq)
        seq_lens = torch.full((B,), Tk, device=inp.q.device, dtype=torch.int32)

    descale_shape = (B, Hkv)
    inp.vllm_triton_q = q_flat
    inp.vllm_triton_k_cache = k_cache
    inp.vllm_triton_v_cache = v_cache
    inp.vllm_triton_block_table = block_table
    inp.vllm_triton_seq_lens = seq_lens
    inp.vllm_triton_cu_seqlens_q = cu_seqlens_q
    inp.vllm_triton_k_descale = torch.ones(
        descale_shape, device=inp.q.device, dtype=torch.float32
    )
    inp.vllm_triton_v_descale = torch.ones(
        descale_shape, device=inp.q.device, dtype=torch.float32
    )


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


class TritonReferenceBackend(AttentionBackend):
    name = "triton_reference"

    def __init__(self) -> None:
        self.triton = None
        self.unified_attention = None

    def available(self) -> Tuple[bool, str]:
        module_candidates = [
            "vllm.v1.attention.ops.triton_unified_attention",
            "vllm.attention.ops.triton_unified_attention",
        ]
        last_exc = None
        for mod_name in module_candidates:
            try:
                self.triton = importlib.import_module("triton")
                mod = importlib.import_module(mod_name)
                self.unified_attention = getattr(mod, "unified_attention")
                vllm = importlib.import_module("vllm")
                return True, (
                    f"vllm {getattr(vllm, '__version__', 'unknown')} + "
                    f"triton {getattr(self.triton, '__version__', 'unknown')}"
                )
            except Exception as exc:
                last_exc = exc
        return False, f"vLLM Triton import failed: {last_exc}"

    def supports(self, case: AttentionCase) -> Tuple[bool, str]:
        if case.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False, "vLLM Triton supports fp16/bf16/fp32"
        if not case.causal:
            return False, "vLLM unified_attention path here is causal-only"
        if case.head_dim < 32:
            return False, "vLLM Triton backend requires head_dim >= 32"
        if case.num_q_heads % case.num_kv_heads != 0:
            return False, "num_q_heads must be divisible by num_kv_heads"
        if case.page_size % 16 != 0:
            return False, "page_size must be a multiple of 16"
        if (not case.paged_kv) and (case.kv_len % case.page_size != 0):
            return False, "for non-paged mode, kv_len must be divisible by page_size"
        return True, "ok"

    def run(self, case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
        if self.unified_attention is None:
            raise RuntimeError("vLLM Triton unified_attention was not initialized")

        prepare_vllm_triton_inputs(case, inp)
        assert inp.vllm_triton_q is not None
        assert inp.vllm_triton_k_cache is not None
        assert inp.vllm_triton_v_cache is not None
        assert inp.vllm_triton_block_table is not None
        assert inp.vllm_triton_seq_lens is not None
        assert inp.vllm_triton_cu_seqlens_q is not None
        assert inp.vllm_triton_k_descale is not None
        assert inp.vllm_triton_v_descale is not None

        q_flat = inp.vllm_triton_q
        out_flat = torch.empty_like(q_flat)
        max_seqlen_k = int(inp.vllm_triton_seq_lens.max().item())
        softmax_scale = 1.0 / math.sqrt(case.head_dim)

        self.unified_attention(
            q=q_flat,
            k=inp.vllm_triton_k_cache,
            v=inp.vllm_triton_v_cache,
            out=out_flat,
            cu_seqlens_q=inp.vllm_triton_cu_seqlens_q,
            max_seqlen_q=case.q_len,
            seqused_k=inp.vllm_triton_seq_lens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=inp.vllm_triton_block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=inp.vllm_triton_k_descale,
            v_descale=inp.vllm_triton_v_descale,
        )

        out = out_flat.view(
            case.batch_size, case.q_len, case.num_q_heads, case.head_dim
        ).permute(0, 2, 1, 3).contiguous()
        return out


class SDPABackend(AttentionBackend):
    name = "sdpa"

    def available(self) -> Tuple[bool, str]:
        return True, "builtin"

    def supports(self, case: AttentionCase) -> Tuple[bool, str]:
        if case.paged_kv:
            return False, "sdpa backend does not implement native paged KV"
        return True, "ok"

    def run(self, case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
        q = inp.q
        k = inp.k
        v = inp.v

        if case.is_gqa:
            k = repeat_kv_for_gqa(k, case.num_q_heads)
            v = repeat_kv_for_gqa(v, case.num_q_heads)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        use_offset_causal_mask = case.causal and (q.shape[-2] != k.shape[-2])
        attn_mask = causal_mask(case.q_len, k.shape[-2], q.device) if use_offset_causal_mask else None
        use_causal_flag = case.causal and (not use_offset_causal_mask)

        # Force PyTorch's memory-efficient SDPA implementation.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore

            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=use_causal_flag,
                )
        except Exception:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True,
            ):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=use_causal_flag,
                )

        return out


class FlashAttnBackend(AttentionBackend):
    name = "flash_attn"

    def __init__(self) -> None:
        self.mod = None
        self.flash_attn_func = None
        self.flash_attn_with_kvcache = None

    def available(self) -> Tuple[bool, str]:
        candidates = [
            ("flash_attn_interface", ["flash_attn_func", "flash_attn_with_kvcache"]),
            ("flash_attn", ["flash_attn_func", "flash_attn_with_kvcache"]),
            ("flash_attn.flash_attn_interface", ["flash_attn_func", "flash_attn_with_kvcache"]),
        ]
        last_exc = None
        for module_name, attrs in candidates:
            try:
                mod = importlib.import_module(module_name)
                self.mod = mod
                self.flash_attn_func = getattr(mod, "flash_attn_func", None)
                self.flash_attn_with_kvcache = getattr(mod, "flash_attn_with_kvcache", None)
                if self.flash_attn_func is not None:
                    return True, f"{module_name}"
            except Exception as exc:
                last_exc = exc
        return False, f"flash-attn import failed: {last_exc}"

    def supports(self, case: AttentionCase) -> Tuple[bool, str]:
        if case.dtype != torch.bfloat16:
            return False, "v1 only benchmarks BF16"
        if case.is_gqa and case.num_q_heads % case.num_kv_heads != 0:
            return False, "num_q_heads must be divisible by num_kv_heads"

        if case.is_decode:
            if self.flash_attn_with_kvcache is None:
                return False, "flash_attn_with_kvcache not found for decode path"
            if case.paged_kv and (case.page_size % 256 != 0):
                return False, "flash-attn paged decode requires page_size divisible by 256"
            return True, "ok"

        if case.paged_kv:
            return False, "v1 flash-attn wrapper does not implement paged KV prefill"
        if self.flash_attn_func is None:
            return False, "flash_attn_func not found for prefill path"
        return True, "ok"

    def run(self, case: AttentionCase, inp: AttentionInputs) -> torch.Tensor:
        if case.is_decode:
            # q: [B, Hq, 1, D] -> [B, 1, Hq, D]
            q = inp.q.permute(0, 2, 1, 3).contiguous()
            if case.paged_kv:
                if inp.k_pages is None or inp.v_pages is None or inp.page_table is None or inp.valid_lens is None:
                    raise RuntimeError("paged KV inputs are incomplete")
                # FlashAttention paged cache layout: [num_blocks, page_size, Hkv, D]
                k_cache = inp.k_pages.permute(0, 2, 1, 3).contiguous()
                v_cache = inp.v_pages.permute(0, 2, 1, 3).contiguous()
                out = self.flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=inp.valid_lens.to(dtype=torch.int32),
                    block_table=inp.page_table.to(dtype=torch.int32),
                    causal=case.causal,
                )
            else:
                k_cache = inp.k.permute(0, 2, 1, 3).contiguous()
                v_cache = inp.v.permute(0, 2, 1, 3).contiguous()
                out = self.flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    causal=case.causal,
                )
            return out.permute(0, 2, 1, 3).contiguous()

        q = inp.q.permute(0, 2, 1, 3).contiguous()  # [B, Tq, Hq, D]
        k = inp.k.permute(0, 2, 1, 3).contiguous()  # [B, Tk, Hkv, D]
        v = inp.v.permute(0, 2, 1, 3).contiguous()

        out = self.flash_attn_func(
            q=q,
            k=k,
            v=v,
            dropout_p=0.0,
            causal=case.causal,
        )
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
                                 "chunked_decode_contiguous_kv",
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
        SDPABackend(),
        FlashAttnBackend(),
        TritonReferenceBackend(),
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
