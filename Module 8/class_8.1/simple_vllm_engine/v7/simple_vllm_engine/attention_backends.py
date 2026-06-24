from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Protocol

import torch

from common.profiling import SimpleProfiler
from .kernels import TritonDecodeMetadata

_FLASH_ATTN_VARLEN = None
_FLASH_ATTN_VERSION_FN = None
_FLASH_ATTN_LOAD_ATTEMPTED = False
_FLASH_ATTN_LOAD_ERROR = None


def _load_flash_attn():
    """Load the paged-varlen FlashAttention binding used by vLLM.

    vLLM's normal CUDA backend calls this same API with paged K/V cache tensors:

        flash_attn_varlen_func(q, k_cache, v_cache, block_table=...)

    We import it lazily so importing the teaching package still works before a
    benchmark actually asks for the optimized attention path.
    """
    global _FLASH_ATTN_VARLEN
    global _FLASH_ATTN_VERSION_FN
    global _FLASH_ATTN_LOAD_ATTEMPTED
    global _FLASH_ATTN_LOAD_ERROR
    if _FLASH_ATTN_LOAD_ATTEMPTED:
        return _FLASH_ATTN_VARLEN, _FLASH_ATTN_VERSION_FN
    _FLASH_ATTN_LOAD_ATTEMPTED = True
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        _FLASH_ATTN_VARLEN = flash_attn_varlen_func
        _FLASH_ATTN_VERSION_FN = get_flash_attn_version
        _FLASH_ATTN_LOAD_ERROR = None
    except Exception as exc:  # pragma: no cover - depends on local CUDA packages.
        _FLASH_ATTN_VARLEN = None
        _FLASH_ATTN_VERSION_FN = None
        _FLASH_ATTN_LOAD_ERROR = repr(exc)
    return _FLASH_ATTN_VARLEN, _FLASH_ATTN_VERSION_FN


def _flash_attn_load_error() -> str | None:
    return _FLASH_ATTN_LOAD_ERROR


class AttentionBackend(Protocol):
    """Attention backend contract used from v4 onward.

    The model supplies Q plus the current K/V chunk. The backend is responsible
    for writing that current K/V into the paged cache before reading attention
    from the cache. Returning `wrote_kv=True` tells the model loop not to write
    K/V a second time.
    """

    def forward(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        kv_cache,
        block_id_lists: list[list[int]],
        block_tables: torch.Tensor,
        query_lens: torch.Tensor,
        past_lens: torch.Tensor,
        key_lens: torch.Tensor,
        key_lens_i32: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        block_tables_i32: torch.Tensor | None = None,
        total_queries: int | None = None,
        max_key_len: int | None = None,
        triton_decode_metadata: TritonDecodeMetadata | None = None,
        decode_slot_mapping: torch.Tensor | None = None,
        prefill_slot_mapping: torch.Tensor | None = None,
        prefill_slot_mapping_all_valid: bool = False,
    ) -> tuple[torch.Tensor, bool]: ...


@dataclass
class FlashAttentionPagedBackend:
    """Optimized paged attention using the FlashAttention varlen API.

    This is the first version of the teaching engine where attention follows the
    same high-level shape as vLLM's CUDA path:

    1. Write current K/V rows into the paged cache.
    2. Flatten the valid query rows into `[sum(query_lens), Hq, D]`.
    3. Call FlashAttention with:
       - `cu_seqlens_q`: where each request's queries begin in the flat tensor
       - `seqused_k`: total visible K/V length per request
       - `block_table`: logical page -> physical page mapping

    Example:

        past_len=10, query_len=4, key_len=14

    With `causal=True`, FlashAttention aligns the 4 query rows to the end of
    the 14-token KV sequence. Query row 0 can see keys 0..10, row 1 can see
    0..11, and so on. That is exactly the chunked-prefill rule we implemented
    manually in the reference versions.
    """

    num_attention_heads: int
    profiler: SimpleProfiler | None = None
    _fa_version_by_head_dim: dict[int, int] = field(default_factory=dict, init=False)

    def _fa_version(self, head_dim: int) -> int:
        version = self._fa_version_by_head_dim.get(head_dim)
        if version is not None:
            return version
        _, get_flash_attn_version = _load_flash_attn()
        if get_flash_attn_version is None:
            raise RuntimeError(f"FlashAttention is not available: {_flash_attn_load_error()}")
        version = get_flash_attn_version(head_size=head_dim)
        if version is None:
            raise RuntimeError("FlashAttention bindings are present but no supported version was found")
        self._fa_version_by_head_dim[head_dim] = int(version)
        return int(version)

    def _write_current_kv(
        self,
        *,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        kv_cache,
        block_id_lists: list[list[int]],
        past_lens: torch.Tensor,
        query_lens: torch.Tensor,
        decode_slot_mapping: torch.Tensor | None,
        prefill_slot_mapping: torch.Tensor | None,
        prefill_slot_mapping_all_valid: bool,
    ) -> None:
        if decode_slot_mapping is not None:
            kv_cache.write_decode_slots(
                layer_idx=layer_idx,
                slot_mapping=decode_slot_mapping,
                k_tokens=k_new,
                v_tokens=v_new,
            )
            return

        if prefill_slot_mapping is not None and hasattr(kv_cache, "write_slot_mapping"):
            kv_cache.write_slot_mapping(
                layer_idx=layer_idx,
                slot_mapping=prefill_slot_mapping,
                k_tokens=k_new,
                v_tokens=v_new,
                assume_all_valid=prefill_slot_mapping_all_valid,
            )
            return

        if not block_id_lists:
            raise RuntimeError("FlashAttention K/V write needs either slot_mapping or block_id_lists")
        kv_cache.write_batch(
            layer_idx=layer_idx,
            block_id_lists=block_id_lists,
            start_tokens=past_lens.tolist(),
            valid_lengths=query_lens.tolist(),
            k_tokens=k_new,
            v_tokens=v_new,
        )

    def _flatten_queries(
        self,
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        total_queries: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if total_queries == q.shape[0] * q.shape[1]:
            return q.reshape(total_queries, q.shape[2], q.shape[3]).contiguous(), None

        query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        valid_query_rows = (
            torch.arange(q.shape[1], device=q.device).unsqueeze(0)
            < query_lens.to(torch.long).unsqueeze(1)
        )
        return q[valid_query_rows].contiguous(), valid_query_rows

    def forward(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        kv_cache,
        block_id_lists: list[list[int]],
        block_tables: torch.Tensor,
        query_lens: torch.Tensor,
        past_lens: torch.Tensor,
        key_lens: torch.Tensor,
        key_lens_i32: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        block_tables_i32: torch.Tensor | None = None,
        total_queries: int | None = None,
        max_key_len: int | None = None,
        triton_decode_metadata: TritonDecodeMetadata | None = None,
        decode_slot_mapping: torch.Tensor | None = None,
        prefill_slot_mapping: torch.Tensor | None = None,
        prefill_slot_mapping_all_valid: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        del triton_decode_metadata  # FlashAttention uses the common metadata tensors directly.

        flash_attn_varlen_func, _ = _load_flash_attn()
        if flash_attn_varlen_func is None:
            raise RuntimeError(f"FlashAttention is not available: {_flash_attn_load_error()}")
        if cu_seqlens_q is None:
            raise RuntimeError("FlashAttention needs cu_seqlens_q")

        key_lens_i32 = key_lens.to(dtype=torch.int32) if key_lens_i32 is None else key_lens_i32
        block_tables_i32 = block_tables.to(dtype=torch.int32) if block_tables_i32 is None else block_tables_i32
        total_queries = int(cu_seqlens_q[-1].item()) if total_queries is None else total_queries
        max_key_len = int(key_lens.max().item()) if max_key_len is None else max_key_len

        with self.profiler.section("model.kv_write") if self.profiler else nullcontext():
            self._write_current_kv(
                layer_idx=layer_idx,
                k_new=k_new,
                v_new=v_new,
                kv_cache=kv_cache,
                block_id_lists=block_id_lists,
                past_lens=past_lens,
                query_lens=query_lens,
                decode_slot_mapping=decode_slot_mapping,
                prefill_slot_mapping=prefill_slot_mapping,
                prefill_slot_mapping_all_valid=prefill_slot_mapping_all_valid,
            )

        with self.profiler.section("model.attention.flash_attn") if self.profiler else nullcontext():
            q_flat, valid_query_rows = self._flatten_queries(q, cu_seqlens_q, total_queries)
            out_flat = torch.empty_like(q_flat)
            flash_attn_varlen_func(
                q=q_flat,
                k=kv_cache.k_layers[layer_idx],
                v=kv_cache.v_layers[layer_idx],
                out=out_flat,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=q.shape[1],
                seqused_k=key_lens_i32,
                max_seqlen_k=max_key_len,
                softmax_scale=1.0 / math.sqrt(q.shape[3]),
                causal=True,
                block_table=block_tables_i32,
                fa_version=self._fa_version(q.shape[3]),
            )

            if valid_query_rows is None:
                return out_flat.view_as(q), True
            out = torch.zeros_like(q)
            out[valid_query_rows] = out_flat
            return out, True


def build_attention_backend(
    backend_name: str,
    *,
    num_attention_heads: int,
    profiler: SimpleProfiler | None = None,
) -> AttentionBackend:
    if backend_name == "flash_attn_paged":
        return FlashAttentionPagedBackend(num_attention_heads=num_attention_heads, profiler=profiler)
    raise ValueError(f"Unknown attention backend: {backend_name}")
