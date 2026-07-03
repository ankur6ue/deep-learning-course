from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kernels import (
    TritonDecodeMetadata,
    apply_rope,
    build_cu_seqlens,
    build_rope_cos_sin_cache,
    build_triton_decode_metadata,
    swiglu,
    triton_fused_add_rms_norm,
    triton_rms_norm,
    triton_apply_rope_from_cache,
    triton_silu_and_mul,
)
from .kv_cache import PagedKVCache
from .requests import RequestState


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        """Construct RMSNorm over the final hidden dimension."""
        super().__init__()
        # Initialized as a valid default; the HF checkpoint loader overwrites
        # this Parameter with the model's learned RMSNorm scale tensor.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm with native Triton fast paths and a plain PyTorch fallback.

        Llama-style blocks use pre-norm with residual accumulation. When a
        residual tensor is present, the fast path fuses:

            x + residual -> RMSNorm

        and returns both the normalized activations and the updated residual.
        Keeping this API close to vLLM's fused-add-RMSNorm shape makes the
        transformer loop easier to compare against production serving code.
        """
        if residual is not None:
            # Callers keep the branch output (`x`) separate from the residual
            # stream so this add can be fused with RMSNorm instead of launching
            # a standalone residual-add kernel.
            fused = triton_fused_add_rms_norm(x, residual, self.weight, self.eps)
            if fused is not None:
                return fused
            x = x + residual
            residual = x
        custom = triton_rms_norm(x, self.weight, self.eps)
        if custom is not None:
            if residual is None:
                return custom
            return custom, residual
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight
        if residual is None:
            return x
        return x, residual

class MiniLlamaLayer(nn.Module):
    """One decoder layer worth of weights.

    The engine packs Q/K/V into a single projection and gate/up into a single
    projection, matching common optimized Llama checkpoint layouts:

        qkv_proj      -> [Q | K | V]
        gate_up_proj  -> [gate | up]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        qkv_size = (
            config.num_attention_heads + 2 * config.num_key_value_heads
        ) * config.head_dim
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_size, bias=False)
        self.o_proj = nn.Linear(
            config.attention_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)


@dataclass
class AttentionBatchMetadata:
    """All attention metadata for one scheduler/model step.

    This object is the contract between the Python scheduler and the model
    body. The model loop should not have to inspect request objects while it is
    running through layers; it receives tensors describing:

    - query lengths for this step
    - already-cached prefix lengths
    - paged KV block tables
    - slot mappings for writing new K/V vectors

    During decode, every active request contributes exactly one query token.
    During chunked prefill, each request may contribute a short chunk of prompt
    tokens, and rows may be padded to a common width.
    """

    query_lens: torch.Tensor
    past_lens: torch.Tensor
    key_lens: torch.Tensor
    key_lens_i32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_tables: torch.Tensor
    block_tables_i32: torch.Tensor
    total_queries: int
    max_key_len: int
    triton_decode_metadata: TritonDecodeMetadata | None = None
    decode_slot_mapping: torch.Tensor | None = None
    prefill_slot_mapping: torch.Tensor | None = None
    prefill_slot_mapping_all_valid: bool = False
    query_all_valid: bool = False


class MiniLlamaLM(nn.Module):
    """Small Llama/Mistral-style decoder used by the teaching engine.

    This class is intentionally only the model body. It does not schedule
    requests, allocate blocks, sample tokens, or manage output state. Those
    responsibilities live in `engine.py`. Keeping the boundary explicit makes
    it easier to explain which work is pure transformer math and which work is
    serving-system control plane.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the decoder modules and reusable RoPE cache."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(MiniLlamaLayer(config))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Precompute the vLLM-compatible cos/sin table once. The old teaching
        # path rebuilt RoPE frequencies and sin/cos tensors repeatedly, which
        # caused many small kernels. This cache plus a fused Triton application
        # kernel is one of the major performance steps.
        rope_cache = build_rope_cos_sin_cache(
            head_size=config.head_dim,
            max_position=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("rope_cos_sin_cache", rope_cache, persistent=False)
        self.profiler = None
        self.attention_backend = None
        # Debug guard for RoPE kernel ordering. Normal execution keeps RoPE and
        # decode workspace updates on the same CUDA stream, so stream ordering is
        # sufficient and this should stay disabled for benchmarking.
        self._sync_after_rope = os.environ.get("SIMPLE_VLLM_SYNC_AFTER_ROPE") == "1"
        # v5 has no static decode workspace or CUDA graph replay yet, so the
        # eager path should not force a synchronization after every forward.
        # The env var remains as a debugging guard when inspecting async kernel
        # lifetime issues.
        self._sync_after_forward = os.environ.get("SIMPLE_VLLM_SYNC_AFTER_FORWARD", "0") != "0"

    def _apply_native_rope_to_flat_qk(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Apply cached RoPE to flat Q/K projection tensors when available.

        The linear projection produces flat Q/K/V columns. Attention kernels
        want `[tokens, heads, head_dim]` tensors with RoPE already applied to Q
        and K. The fast path below launches one native Triton RoPE kernel for Q
        and one for K, using the cached cos/sin table built during init.

        The fallback path later in `_split_projected_qkv*` keeps correctness on
        CPU or if the custom kernel is unavailable.
        """
        if q_flat.device.type != "cuda":
            return None
        q_shape = q_flat.shape
        k_shape = k_flat.shape
        q_2d = q_flat.reshape(-1, q_shape[-1])
        k_2d = k_flat.reshape(-1, k_shape[-1])
        # RoPE runs on the current CUDA stream. Decode may pass a view into a
        # reusable positions workspace, but the next workspace update is queued
        # on the same stream and therefore cannot overwrite values before this
        # RoPE kernel reads them.
        flat_positions = positions.reshape(-1).contiguous()
        if flat_positions.numel() != q_2d.shape[0]:
            return None
        try:
            cos_sin_cache = self.rope_cos_sin_cache
            q_triton = triton_apply_rope_from_cache(
                q_2d,
                flat_positions,
                cos_sin_cache,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.head_dim,
            )
            k_triton = triton_apply_rope_from_cache(
                k_2d,
                flat_positions,
                cos_sin_cache,
                num_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
            )
        except Exception:
            q_triton = None
            k_triton = None
        if q_triton is not None and k_triton is not None:
            if self._sync_after_rope:
                torch.cuda.current_stream(q_triton.device).synchronize()
            return q_triton.view(q_shape), k_triton.view(k_shape)
        return None

    def _split_projected_qkv(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split packed QKV for padded/chunked prefill tensors.

        Shape convention for the generic path:

            qkv: [batch, query_tokens, qkv_width]
            q:   [batch, query_tokens, num_q_heads, head_dim]
            k/v: [batch, query_tokens, num_kv_heads, head_dim]
        """
        bsz, seqlen, _ = qkv.shape
        q_size = self.config.num_attention_heads * self.config.head_dim
        kv_size = self.config.num_key_value_heads * self.config.head_dim
        q_flat, k_flat, v_flat = qkv.split(
            [q_size, kv_size, kv_size],
            dim=-1,
        )
        rotated = self._apply_native_rope_to_flat_qk(q_flat, k_flat, positions)
        if rotated is not None:
            q_flat, k_flat = rotated
            q = q_flat.reshape(bsz, seqlen, self.config.num_attention_heads, self.config.head_dim)
            k = k_flat.reshape(bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim)
            v = v_flat.reshape(bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim)
            return q, k, v
        q = q_flat.reshape(bsz, seqlen, self.config.num_attention_heads, self.config.head_dim)
        k = k_flat.reshape(bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim)
        v = v_flat.reshape(bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim)
        q = apply_rope(q, positions, self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        k = apply_rope(k, positions, self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        return q, k, v

    def _split_projected_qkv_decode(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split packed QKV for the optimized single-token decode path.

        Decode processes one token per active request, so the hot path keeps
        activations token-major and 2D where possible:

            qkv: [active_requests, qkv_width]
            q:   [active_requests, num_q_heads, head_dim]
            k/v: [active_requests, num_kv_heads, head_dim]
        """
        q_size = self.config.num_attention_heads * self.config.head_dim
        kv_size = self.config.num_key_value_heads * self.config.head_dim
        q_flat, k_flat, v_flat = qkv.split([q_size, kv_size, kv_size], dim=-1)
        rotated = self._apply_native_rope_to_flat_qk(q_flat, k_flat, positions.reshape(-1))
        if rotated is not None:
            q_flat, k_flat = rotated
            q = q_flat.reshape(-1, self.config.num_attention_heads, self.config.head_dim)
            k = k_flat.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
            v = v_flat.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
            return q, k, v
        q = q_flat.reshape(-1, self.config.num_attention_heads, self.config.head_dim)
        k = k_flat.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
        v = v_flat.reshape(-1, self.config.num_key_value_heads, self.config.head_dim)
        flat_positions = positions.reshape(-1)
        q = apply_rope(q, flat_positions, self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        k = apply_rope(k, flat_positions, self.config.rope_theta, rope_scaling=self.config.rope_scaling)
        return q, k, v

    def _project_qkv(
        self,
        layer: MiniLlamaLayer,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into Q/K/V and apply RoPE to Q and K."""
        qkv = F.linear(
            hidden.reshape(-1, self.config.hidden_size),
            layer.qkv_proj.weight,
        ).view(*hidden.shape[:-1], -1)
        return self._split_projected_qkv(qkv, positions)

    def _project_qkv_decode(
        self,
        layer: MiniLlamaLayer,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one decode token per request using vLLM-style 2D tensors."""
        qkv = F.linear(hidden, layer.qkv_proj.weight)
        return self._split_projected_qkv_decode(qkv, positions)

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
        past_lengths = [req.cached_seq_len for req in requests]
        past_lens = torch.tensor(past_lengths, device=device, dtype=torch.long)
        # Each row attends over `[past cached tokens | current query chunk]`.
        # Keep the values as a Python list for block-table sizing, then build
        # the GPU tensor form required by attention backends.
        #
        # Example: past_lengths=[10, 32], lengths=[4, 1]
        #          key_lengths=[14, 33]
        key_lengths = [
            past_len + query_len
            for past_len, query_len in zip(past_lengths, lengths, strict=True)
        ]
        key_lens = torch.tensor(key_lengths, device=device, dtype=torch.long)
        max_key_len = max(key_lengths) if key_lengths else 0
        # The block table keeps the logical-to-physical page mapping for each
        # request. Each row says, "logical block 0/1/2 for this request lives in
        # physical cache block X/Y/Z." This is what makes the KV cache paged:
        # request memory no longer has to be contiguous.
        block_tables = kv_cache.block_tables_tensor(
            [req.block_ids for req in requests],
            key_lengths,
        )
        cu_seqlens_q = build_cu_seqlens(query_lens)
        key_lens_i32 = key_lens.to(dtype=torch.int32)
        block_tables_i32 = block_tables.to(dtype=torch.int32)
        triton_decode_metadata = None
        decode_slot_mapping = None
        prefill_slot_mapping = None
        # Two simple shape facts drive most of the branching below:
        #
        # - Decode: every request contributes exactly one query token.
        # - All-valid: every row has the same query length, so there are no
        #   padded query cells to mask or skip.
        is_decode = bool(lengths and all(length == 1 for length in lengths))
        query_all_valid = bool(lengths and all(length == lengths[0] for length in lengths))
        if is_decode:
            # Decode case: one new token per request. The attention backend can
            # write K/V directly into the exact physical slot for that token.
            slots = []
            for req, past_len in zip(requests, past_lengths, strict=True):
                slots.append(kv_cache.physical_slot(req.block_ids, past_len))
            decode_slot_mapping = torch.tensor(slots, device=device, dtype=torch.long)
            triton_decode_metadata = build_triton_decode_metadata(
                cu_seqlens_q=cu_seqlens_q,
                key_lens=key_lens_i32,
                block_tables=block_tables_i32,
                max_seqlen_k=max_key_len,
                num_kv_heads=self.config.num_key_value_heads,
            )
        elif lengths:
            # Prefill case: each request contributes a chunk of prompt tokens.
            # Chunks may have different lengths, so the slot mapping is padded
            # with -1 for invalid cells.
            slot_rows = []
            max_query_len = max(lengths)
            for req, past_len, query_len in zip(requests, past_lengths, lengths, strict=True):
                row = []
                for offset in range(query_len):
                    token_pos = past_len + offset
                    row.append(kv_cache.physical_slot(req.block_ids, token_pos))
                row.extend([-1] * (max_query_len - query_len))
                slot_rows.append(row)
            prefill_slot_mapping = torch.tensor(slot_rows, device=device, dtype=torch.long)
        prefill_slot_mapping_all_valid = prefill_slot_mapping is not None and query_all_valid
        return AttentionBatchMetadata(
            query_lens=query_lens,
            past_lens=past_lens,
            key_lens=key_lens,
            key_lens_i32=key_lens_i32,
            cu_seqlens_q=cu_seqlens_q,
            block_tables=block_tables,
            block_tables_i32=block_tables_i32,
            total_queries=sum(lengths),
            max_key_len=max_key_len,
            triton_decode_metadata=triton_decode_metadata,
            decode_slot_mapping=decode_slot_mapping,
            prefill_slot_mapping=prefill_slot_mapping,
            prefill_slot_mapping_all_valid=prefill_slot_mapping_all_valid,
            query_all_valid=query_all_valid,
        )

    def _forward_with_metadata(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
        block_id_lists: list[list[int]] | None = None,
    ) -> torch.Tensor:
        return self._forward_with_metadata_impl(
            input_ids=input_ids,
            positions=positions,
            metadata=metadata,
            kv_cache=kv_cache,
            block_id_lists=block_id_lists,
        )

    def _forward_with_metadata_impl(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
        block_id_lists: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Run the generic batched transformer path.

        This path handles chunked prefill and any decode batch that cannot use
        the specialized single-token decode layout. It keeps `[B, T, H]`
        activations because prefill chunks can contain multiple query tokens per
        request and may need padding masks.
        """
        with self.profiler.section("model.embed") if self.profiler else nullcontext():
            hidden = self.embed_tokens(input_ids)
        # This mask is only about padding inside the current batched chunk.
        # Example: if a request already prefetched 5 prompt tokens and is now
        # processing a 4-token chunk, `query_lens` is 4 here. The absolute
        # position within the prompt is carried separately by `positions`, and
        # visibility into the cached prefix is carried by `past_lens`.
        mask = None
        if not metadata.query_all_valid:
            mask = (
                torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                < metadata.query_lens.unsqueeze(1)
            ).to(hidden.dtype)
            hidden = self._masked_residual(hidden, mask)

        block_id_lists = block_id_lists or []

        # Llama keeps two values in flight: `hidden` is the newest branch output
        # and `residual` is the accumulated residual stream. We defer residual
        # adds until RMSNorm sites so add+norm can use the fused kernel.
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            # 1. Input RMSNorm/residual + QKV projection + RoPE.
            with self.profiler.section("model.qkv_proj") if self.profiler else nullcontext():
                if residual is None:
                    # First layer: initialize the residual stream from embeddings.
                    # There is no previous MLP branch to add yet.
                    residual = hidden
                    normed = layer.input_norm(hidden)
                else:
                    # Later layers: `hidden` is the previous layer's MLP output.
                    # `input_norm` adds it to the residual stream and normalizes in
                    # one fused add+RMSNorm operation.
                    normed, residual = layer.input_norm(hidden, residual)
                # `positions` holds absolute token indices, so RoPE still sees
                # the right offsets during later prefill chunks.
                q, k_new, v_new = self._project_qkv(layer, normed, positions)
            # 2. Paged attention boundary.
            #
            # Attention is delegated to the selected backend. The model passes
            # Q plus the newly computed K/V and all metadata needed to read old
            # K/V from the paged cache. Some backends also write the new K/V
            # inside the attention call; others ask us to write it afterward.
            attn_out_heads = self.attention_backend.forward(
                layer_idx=layer_idx,
                q=q,
                k_new=k_new,
                v_new=v_new,
                kv_cache=kv_cache,
                block_id_lists=block_id_lists,
                block_tables=metadata.block_tables,
                query_lens=metadata.query_lens,
                past_lens=metadata.past_lens,
                key_lens=metadata.key_lens,
                key_lens_i32=metadata.key_lens_i32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                block_tables_i32=metadata.block_tables_i32,
                total_queries=metadata.total_queries,
                max_key_len=metadata.max_key_len,
                triton_decode_metadata=metadata.triton_decode_metadata,
                decode_slot_mapping=metadata.decode_slot_mapping,
                prefill_slot_mapping=metadata.prefill_slot_mapping,
                prefill_slot_mapping_all_valid=metadata.prefill_slot_mapping_all_valid,
            )

            # 3. Output projection + post-attention norm + packed SwiGLU MLP.
            with self.profiler.section("model.attn_out_proj") if self.profiler else nullcontext():
                attn_out = F.linear(
                    attn_out_heads.reshape(-1, self.config.attention_hidden_size),
                    layer.o_proj.weight,
                ).view(attn_out_heads.shape[0], attn_out_heads.shape[1], -1)
                hidden = attn_out if mask is None else self._masked_residual(attn_out, mask)
            with self.profiler.section("model.post_attn_norm") if self.profiler else nullcontext():
                # Attention output is the other Llama residual branch. Add it at
                # the post-attention norm boundary to keep add+RMSNorm fused.
                normed, residual = layer.post_norm(hidden, residual)
            with self.profiler.section("model.mlp") if self.profiler else nullcontext():
                gate_up = F.linear(
                    normed.reshape(-1, self.config.hidden_size),
                    layer.gate_up_proj.weight,
                )
                activated = triton_silu_and_mul(gate_up)
                if activated is None:
                    gate, up = gate_up.chunk(2, dim=-1)
                    activated = swiglu(gate, up)
                hidden = F.linear(activated, layer.down_proj.weight).view_as(normed)
                if mask is not None:
                    hidden = self._masked_residual(hidden, mask)

        with self.profiler.section("model.final_norm_lm_head") if self.profiler else nullcontext():
            # Final normalization is applied to every valid token in the step,
            # but logits are only needed for the last token of each request.
            if residual is not None:
                # Flush the final layer's deferred MLP output into the residual
                # stream before the final RMSNorm.
                hidden, _ = self.norm(hidden, residual)
            else:
                hidden = self.norm(hidden)
        # For each request, take the final valid token from this chunk. During
        # prefill this is the last token of the chunk; during decode it is the
        # single decode token.
            if metadata.query_all_valid:
                last_hidden = hidden[:, -1, :]
            else:
                last_hidden = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device),
                    metadata.query_lens - 1,
                ]
            logits = self.lm_head(last_hidden)
            if self._sync_after_forward and logits.is_cuda:
                torch.cuda.current_stream(logits.device).synchronize()
            return logits

    def _forward_request_batch(
        self,
        requests: list[RequestState],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lengths: list[int],
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run one batched model step covering multiple requests at once."""
        metadata = self._build_attention_metadata(requests, lengths, kv_cache)
        return self._forward_with_metadata(
            input_ids=input_ids,
            positions=positions,
            metadata=metadata,
            kv_cache=kv_cache,
            block_id_lists=[req.block_ids for req in requests],
        )

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
