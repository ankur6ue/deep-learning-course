from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch

from .kernels import (
    batched_sdpa_attention,
    paged_sdpa_attention,
    paged_triton_decode_attention,
    repeat_kv,
)
from .profiling import SimpleProfiler


class AttentionBackend(Protocol):
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
    ) -> tuple[torch.Tensor, bool]: ...


@dataclass
class DenseReferenceAttentionBackend:
    """Reference backend that preserves the v3 dense-KV behavior."""

    num_attention_heads: int
    profiler: SimpleProfiler | None = None

    def _assemble_full_kv(
        self,
        k_past: torch.Tensor,
        v_past: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        query_lens: torch.Tensor,
        past_lens: torch.Tensor,
        key_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = k_new.shape[0]
        max_key_len = int(key_lens.max().item()) if batch_size > 0 else 0
        k_full = k_new.new_zeros((batch_size, max_key_len, k_new.shape[2], k_new.shape[3]))
        v_full = v_new.new_zeros((batch_size, max_key_len, v_new.shape[2], v_new.shape[3]))
        for req_idx in range(batch_size):
            past_len = int(past_lens[req_idx].item())
            query_len = int(query_lens[req_idx].item())
            if past_len > 0:
                k_full[req_idx, :past_len].copy_(k_past[req_idx, :past_len])
                v_full[req_idx, :past_len].copy_(v_past[req_idx, :past_len])
            if query_len > 0:
                k_full[req_idx, past_len : past_len + query_len].copy_(k_new[req_idx, :query_len])
                v_full[req_idx, past_len : past_len + query_len].copy_(v_new[req_idx, :query_len])
        return (
            repeat_kv(k_full, self.num_attention_heads),
            repeat_kv(v_full, self.num_attention_heads),
        )

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
    ) -> tuple[torch.Tensor, bool]:
        with self.profiler.section("model.kv_gather") if self.profiler else nullcontext():
            past_batch = kv_cache.gather_batch(layer_idx, block_id_lists, past_lens.tolist())
        with self.profiler.section("model.kv_assemble") if self.profiler else nullcontext():
            k_full, v_full = self._assemble_full_kv(
                k_past=past_batch.k,
                v_past=past_batch.v,
                k_new=k_new,
                v_new=v_new,
                query_lens=query_lens,
                past_lens=past_lens,
                key_lens=key_lens,
            )
        with self.profiler.section("model.attention") if self.profiler else nullcontext():
            return (
                batched_sdpa_attention(
                    q=q,
                    k=k_full,
                    v=v_full,
                    query_lens=query_lens,
                    key_lens=key_lens,
                    past_lens=past_lens,
                    enable_gqa=False,
                ),
                False,
            )


@dataclass
class PagedSDPAAttentionBackend:
    """More direct paged backend for v4.

    It writes the current chunk into the paged cache and then reads keys and
    values back through the request block tables, one logical block at a time.
    The implementation is still a Python reference path, but the kernel
    contract is now "paged cache + block tables" rather than "dense K/V tensor".
    """

    num_attention_heads: int
    profiler: SimpleProfiler | None = None

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
    ) -> tuple[torch.Tensor, bool]:
        if not torch.all(query_lens == 1):
            # Keep multi-token prefill on the dense reference path for now.
            dense_reference = DenseReferenceAttentionBackend(
                num_attention_heads=self.num_attention_heads,
                profiler=self.profiler,
            )
            return dense_reference.forward(
                layer_idx=layer_idx,
                q=q,
                k_new=k_new,
                v_new=v_new,
                kv_cache=kv_cache,
                block_id_lists=block_id_lists,
                block_tables=block_tables,
                query_lens=query_lens,
                past_lens=past_lens,
                key_lens=key_lens,
            )

        with self.profiler.section("model.kv_write") if self.profiler else nullcontext():
            kv_cache.write_batch(
                layer_idx=layer_idx,
                block_id_lists=block_id_lists,
                start_tokens=past_lens.tolist(),
                valid_lengths=query_lens.tolist(),
                k_tokens=k_new,
                v_tokens=v_new,
            )
        with self.profiler.section("model.attention") if self.profiler else nullcontext():
            triton_out = paged_triton_decode_attention(
                q=q,
                k_cache=kv_cache.k_layers[layer_idx],
                v_cache=kv_cache.v_layers[layer_idx],
                block_tables=block_tables,
                key_lens=key_lens,
                block_size=kv_cache.block_size,
            )
            if triton_out is not None:
                return triton_out, True
            return (
                paged_sdpa_attention(
                    q=q,
                    k_cache=kv_cache.k_layers[layer_idx],
                    v_cache=kv_cache.v_layers[layer_idx],
                    block_tables=block_tables,
                    query_lens=query_lens,
                    key_lens=key_lens,
                    past_lens=past_lens,
                    block_size=kv_cache.block_size,
                ),
                True,
            )


def build_attention_backend(
    backend_name: str,
    *,
    num_attention_heads: int,
    profiler: SimpleProfiler | None = None,
) -> AttentionBackend:
    if backend_name == "dense_reference":
        return DenseReferenceAttentionBackend(
            num_attention_heads=num_attention_heads,
            profiler=profiler,
        )
    if backend_name == "paged_sdpa":
        return PagedSDPAAttentionBackend(
            num_attention_heads=num_attention_heads,
            profiler=profiler,
        )
    raise ValueError(f"Unknown attention backend: {backend_name}")
