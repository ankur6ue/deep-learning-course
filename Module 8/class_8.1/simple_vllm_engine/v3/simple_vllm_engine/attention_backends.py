from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch

from common.profiling import SimpleProfiler
from .kernels import batched_sdpa_attention, repeat_kv


class AttentionBackend(Protocol):
    """Attention contract for v3.

    v3 teaches the cost of leaving the paged layout before attention. The model
    passes Q plus the new K/V chunk and paged-cache metadata. The backend returns
    `(attention_output, wrote_kv)`.

    `wrote_kv=False` means the model loop must append the current K/V chunk to
    the paged cache after attention finishes.
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
    ) -> tuple[torch.Tensor, bool]: ...


@dataclass
class GatheredSDPAAttention:
    """Gather paged K/V into dense tensors, then call PyTorch SDPA.

    This version is intentionally simple but no longer direct-paged:

    1. `kv_cache.gather_batch(...)` reads each request's cached prefix pages
       into a padded dense tensor shaped `[B, max_past, Hkv, D]`.
    2. `_assemble_full_kv(...)` builds `[cached_prefix | current_chunk]` for
       every request, padded to the same `max_key_len`.
    3. `batched_sdpa_attention(...)` builds the causal/padding mask and calls
       `torch.nn.functional.scaled_dot_product_attention`.

    Example:

        request A: past_len=5, query_len=2 -> keys [0..6]
        request B: past_len=9, query_len=1 -> keys [0..9]

    The backend creates a dense K/V batch with width 10. A's row has real keys
    in columns 0..6 and padding in 7..9; B's row has real keys in 0..9.
    """

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
                start = past_len
                end = past_len + query_len
                k_full[req_idx, start:end].copy_(k_new[req_idx, :query_len])
                v_full[req_idx, start:end].copy_(v_new[req_idx, :query_len])

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
        del block_tables  # v3 gathers into dense K/V, so the SDPA call does not use page ids.
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
        with self.profiler.section("model.attention.gathered_sdpa") if self.profiler else nullcontext():
            out = batched_sdpa_attention(
                q=q,
                k=k_full,
                v=v_full,
                query_lens=query_lens,
                key_lens=key_lens,
                past_lens=past_lens,
                enable_gqa=False,
            )
        return out, False


def build_attention_backend(
    backend_name: str,
    *,
    num_attention_heads: int,
    profiler: SimpleProfiler | None = None,
) -> AttentionBackend:
    if backend_name == "gathered_sdpa":
        return GatheredSDPAAttention(num_attention_heads=num_attention_heads, profiler=profiler)
    raise ValueError(f"Unknown attention backend: {backend_name}")
