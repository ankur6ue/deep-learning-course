from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .kernels import (
    apply_rope,
    batched_sdpa_attention,
    build_cu_seqlens,
    repeat_kv,
    swiglu,
)
from .kv_cache import PagedKVCache
from .requests import RequestState


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        """Construct RMSNorm over the final hidden dimension."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each token vector along its hidden dimension."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


@dataclass
class LayerWeights:
    input_norm: RMSNorm
    post_norm: RMSNorm
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear


@dataclass
class AttentionBatchMetadata:
    query_lens: torch.Tensor
    past_lens: torch.Tensor
    key_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_tables: torch.Tensor


class MiniLlamaLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        """Build a compact random-initialized Llama-style decoder."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "input_norm": RMSNorm(config.hidden_size, config.rms_norm_eps),
                        "post_norm": RMSNorm(config.hidden_size, config.rms_norm_eps),
                        "q_proj": nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False),
                        "k_proj": nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False),
                        "v_proj": nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False),
                        "o_proj": nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False),
                        "gate_proj": nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                        "up_proj": nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                        "down_proj": nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                    }
                )
            )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.profiler = None

    def _project_qkv(
        self,
        layer: nn.ModuleDict,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into Q/K/V and apply RoPE to Q and K.

        Args:
            layer: One transformer layer's weights.
            hidden: Hidden states shaped `[B, T, hidden_size]`.
            positions: Absolute token positions shaped `[B, T]`. For a later
                prompt chunk these are not `0..T-1`; they continue from the
                earlier prompt prefix.
        """
        bsz, seqlen, _ = hidden.shape
        q = layer["q_proj"](hidden).view(
            bsz, seqlen, self.config.num_attention_heads, self.config.head_dim
        )
        k = layer["k_proj"](hidden).view(
            bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim
        )
        v = layer["v_proj"](hidden).view(
            bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim
        )
        q = apply_rope(q, positions, self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        k = apply_rope(k, positions, self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        return q, k, v

    def _masked_residual(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero out padded token rows before adding a residual branch."""
        return x * mask.unsqueeze(-1)

    def _build_attention_metadata(
        self,
        requests: list[RequestState],
        lengths: list[int],
        kv_cache: PagedKVCache,
    ) -> AttentionBatchMetadata:
        """Assemble the per-request lengths needed by the batched attention path.

        Args:
            requests: Requests represented by the current batch rows.
            lengths: Query tokens being processed now per request. During
                decode, this is all ones. During chunked prefill, lengths may be
                something like `[4, 4, 1]`.
            kv_cache: Used here to materialize the batched block table from each
                request's block ids.
        """
        device = next(self.parameters()).device
        # These lengths describe only the query tokens being processed in this
        # scheduler step. For a mid-prompt prefill chunk, this is the chunk
        # length, not the full prompt length.
        query_lens = torch.tensor(lengths, device=device, dtype=torch.long)
        # `cached_seq_len` is how many tokens for this request are already in
        # the KV cache before the current chunk runs.
        past_lens = torch.tensor([req.cached_seq_len for req in requests], device=device, dtype=torch.long)
        # Each query token attends over the previously cached tokens plus the
        # current chunk being appended in this step.
        key_lens = past_lens + query_lens
        # The block table keeps the logical-to-physical page mapping for each
        # request. The kernel path still needs this even though we gather into a
        # padded batch tensor for teaching clarity.
        block_tables = kv_cache.block_tables_tensor(
            [req.block_ids for req in requests],
            key_lens.tolist(),
        )
        return AttentionBatchMetadata(
            query_lens=query_lens,
            past_lens=past_lens,
            key_lens=key_lens,
            cu_seqlens_q=build_cu_seqlens(query_lens),
            block_tables=block_tables,
        )

    def _assemble_full_kv(
        self,
        k_past: torch.Tensor,
        v_past: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        metadata: AttentionBatchMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate cached prefix KV with current-chunk KV for the batch.

        Args:
            k_past: Gathered cached keys shaped `[B, Tpast_pad, Hkv, D]`.
            v_past: Gathered cached values shaped `[B, Tpast_pad, Hkv, D]`.
            k_new: Current chunk keys shaped `[B, Tq_pad, Hkv, D]`.
            v_new: Current chunk values shaped `[B, Tq_pad, Hkv, D]`.
            metadata: Per-request valid lengths describing how much of each row
                is real. This matters because both the past view and current
                chunk are padded.
        """
        batch_size = k_new.shape[0]
        max_key_len = int(metadata.key_lens.max().item()) if batch_size > 0 else 0
        # Build one padded K/V tensor for the entire request batch. Each row is
        # `[cached prefix | current chunk | pad]` for a single request.
        k_full = k_new.new_zeros((batch_size, max_key_len, k_new.shape[2], k_new.shape[3]))
        v_full = v_new.new_zeros((batch_size, max_key_len, v_new.shape[2], v_new.shape[3]))
        for req_idx in range(batch_size):
            past_len = int(metadata.past_lens[req_idx].item())
            query_len = int(metadata.query_lens[req_idx].item())
            if past_len > 0:
                k_full[req_idx, :past_len].copy_(k_past[req_idx, :past_len])
                v_full[req_idx, :past_len].copy_(v_past[req_idx, :past_len])
            if query_len > 0:
                k_full[req_idx, past_len : past_len + query_len].copy_(k_new[req_idx, :query_len])
                v_full[req_idx, past_len : past_len + query_len].copy_(v_new[req_idx, :query_len])
        return (
            repeat_kv(k_full, self.config.num_attention_heads),
            repeat_kv(v_full, self.config.num_attention_heads),
        )

    def _forward_request_batch(
        self,
        requests: list[RequestState],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lengths: list[int],
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run one batched model step covering multiple requests at once.

        Args:
            requests: Requests represented by this batch.
            input_ids: Tokens for the current step, padded to a common width.
                During decode this is `[B, 1]`. During prefill it is a prompt
                chunk such as `[B, max_chunk]`.
            positions: Absolute positions for those tokens. This is how a later
                prompt chunk keeps the right RoPE offsets.
            lengths: Valid query-token count per batch row.
            kv_cache: Shared paged KV cache storing all previously computed
                prefix tokens.
        """
        metadata = self._build_attention_metadata(requests, lengths, kv_cache)
        with self.profiler.section("model.embed") if self.profiler else nullcontext():
            hidden = self.embed_tokens(input_ids)
        # This mask is only about padding inside the current batched chunk.
        # Example: if a request already prefetched 5 prompt tokens and is now
        # processing a 4-token chunk, `query_lens` is 4 here. The absolute
        # position within the prompt is carried separately by `positions`, and
        # visibility into the cached prefix is carried by `past_lens`.
        mask = (
            torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            < metadata.query_lens.unsqueeze(1)
        ).to(hidden.dtype)
        hidden = self._masked_residual(hidden, mask)

        block_id_lists = [req.block_ids for req in requests]
        past_lens = metadata.past_lens.tolist()
        query_lens = metadata.query_lens.tolist()

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden
            with self.profiler.section("model.qkv_proj") if self.profiler else nullcontext():
                normed = layer["input_norm"](hidden)
            # `positions` holds absolute token indices, so RoPE still sees the
            # right offsets when we are processing a later prefill chunk.
                q, k_new, v_new = self._project_qkv(layer, normed, positions)
            # Gather all previously cached pages for the whole batch into one
            # padded `[B, Tpast, Hkv, D]` view.
            with self.profiler.section("model.kv_gather") if self.profiler else nullcontext():
                past_batch = kv_cache.gather_batch(layer_idx, block_id_lists, past_lens)
            with self.profiler.section("model.kv_assemble") if self.profiler else nullcontext():
                k_full, v_full = self._assemble_full_kv(
                    k_past=past_batch.k,
                    v_past=past_batch.v,
                    k_new=k_new,
                    v_new=v_new,
                    metadata=metadata,
                )
            # One attention call covers every request in the batch. The per-
            # request differences in chunk length and prefix length are carried
            # by `query_lens`, `key_lens`, and `past_lens`.
            with self.profiler.section("model.attention") if self.profiler else nullcontext():
                attn_out_heads = batched_sdpa_attention(
                    q=q,
                    k=k_full,
                    v=v_full,
                    query_lens=metadata.query_lens,
                    key_lens=metadata.key_lens,
                    past_lens=metadata.past_lens,
                )
            # After attention consumes the current chunk, append that chunk's
            # K/V into the paged cache so later decode steps can see it.
            with self.profiler.section("model.kv_write") if self.profiler else nullcontext():
                kv_cache.write_batch(
                    layer_idx=layer_idx,
                    block_id_lists=block_id_lists,
                    start_tokens=past_lens,
                    valid_lengths=query_lens,
                    k_tokens=k_new,
                    v_tokens=v_new,
                )

            with self.profiler.section("model.attn_out_proj") if self.profiler else nullcontext():
                attn_out = layer["o_proj"](
                    attn_out_heads.reshape(attn_out_heads.shape[0], attn_out_heads.shape[1], -1)
                )
                hidden = residual + self._masked_residual(attn_out, mask)
            residual = hidden
            with self.profiler.section("model.mlp") if self.profiler else nullcontext():
                normed = layer["post_norm"](hidden)
                ff = layer["down_proj"](swiglu(layer["gate_proj"](normed), layer["up_proj"](normed)))
                hidden = residual + self._masked_residual(ff, mask)

        with self.profiler.section("model.final_norm_lm_head") if self.profiler else nullcontext():
            hidden = self.norm(hidden)
        # For each request, take the final valid token from this chunk. During
        # prefill this is the last token of the chunk; during decode it is the
        # single decode token.
            last_hidden = hidden[
                torch.arange(hidden.shape[0], device=hidden.device),
                metadata.query_lens - 1,
            ]
            return self.lm_head(last_hidden)

    def prefill_chunk(
        self,
        requests: list[RequestState],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lengths: list[int],
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run one chunked-prefill forward pass through the batched path.

        Args:
            requests: Requests represented by the batch rows.
            input_ids: Prompt chunk tokens padded to a common width.
            positions: Absolute positions for those chunk tokens.
            lengths: Valid chunk length per request row.
            kv_cache: Shared paged KV cache to read the existing prefix from and
                append the new chunk to.
        """
        return self._forward_request_batch(
            requests=requests,
            input_ids=input_ids,
            positions=positions,
            lengths=lengths,
            kv_cache=kv_cache,
        )

    def decode_tokens(
        self,
        requests: list[RequestState],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run one decode forward pass for a batch of one-token queries.

        Args:
            requests: Requests decoding one new token each.
            input_ids: Input token ids shaped `[B, 1]`.
            positions: Absolute decode position per request, shaped `[B, 1]`.
            kv_cache: Shared paged KV cache containing the full prefix for each
                request.
        """
        return self._forward_request_batch(
            requests=requests,
            input_ids=input_ids,
            positions=positions,
            lengths=[1] * len(requests),
            kv_cache=kv_cache,
        )
