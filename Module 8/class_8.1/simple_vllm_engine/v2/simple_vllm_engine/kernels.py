from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F



def rotary_frequencies(
    seq_positions: torch.Tensor,
    head_dim: int,
    theta: float,
    rope_scaling: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cosine and sine tables for the requested positions.

    Args:
        seq_positions: Absolute token positions such as `[0, 1, 2]` or, during
            chunked prefill, `[5, 6, 7, 8]`.
        head_dim: Per-head dimension. Must be even because RoPE rotates pairs
            of channels.
        theta: RoPE base frequency.
        rope_scaling: Optional rope scaling metadata from a pretrained config.
            `v3` supports the Llama 3 scaling rule in addition to default RoPE.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=seq_positions.device).float() / head_dim)
    )
    attention_scaling = 1.0
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type")
        if rope_type == "llama3":
            factor = float(rope_scaling["factor"])
            low_freq_factor = float(rope_scaling["low_freq_factor"])
            high_freq_factor = float(rope_scaling["high_freq_factor"])
            old_context_len = float(rope_scaling["original_max_position_embeddings"])

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_scaled / factor + smooth_factor * inv_freq_scaled
            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_scaled)
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported rope_scaling rope_type: {rope_type}")
    freqs = torch.outer(seq_positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb) * attention_scaling, torch.sin(emb) * attention_scaling


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the Llama-style half-dimension rotation used by HF models."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    rope_scaling: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Apply rotary position embedding to Q or K.

    Args:
        x: Tensor shaped `[B, T, H, D]`.
        positions: Absolute token positions for each batch element. For a later
            prefill chunk, these positions continue from the earlier chunk.
        theta: RoPE base frequency.
        rope_scaling: Optional pretrained-model rope scaling metadata.
    """
    # x: [B, T, H, D]
    cos, sin = rotary_frequencies(positions.view(-1), x.shape[-1], theta, rope_scaling=rope_scaling)
    cos = cos.view(*positions.shape, 1, -1).to(dtype=x.dtype)
    sin = sin.view(*positions.shape, 1, -1).to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def causal_attn_mask(query_len: int, key_len: int, past_len: int, device: torch.device) -> torch.Tensor:
    """Build an additive causal mask for one request.

    Args:
        query_len: Number of query tokens in the current step.
        key_len: Total visible KV tokens, including cached prefix plus current
            chunk.
        past_len: Number of cached prefix tokens already in the KV cache.
        device: Target device for the mask tensor.
    """
    # Query token i in the current chunk can see all past tokens and query tokens <= i.
    q_idx = torch.arange(query_len, device=device).unsqueeze(1)
    k_idx = torch.arange(key_len, device=device).unsqueeze(0)
    allowed = k_idx <= (past_len + q_idx)
    mask = torch.full((query_len, key_len), float("-inf"), device=device)
    mask = mask.masked_fill(allowed, 0.0)
    return mask


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_len: int,
) -> torch.Tensor:
    """Run scaled dot-product attention for one request.

    Args:
        q: Query tensor shaped `[Tq, Hq, D]`.
        k: Key tensor shaped `[Tk, Hq, D]`.
        v: Value tensor shaped `[Tk, Hq, D]`.
        past_len: Number of cached tokens that existed before the current query
            chunk.
    """
    # q: [Tq, Hq, D], k/v: [Tk, Hq, D]
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    mask = causal_attn_mask(q.shape[0], k.shape[0], past_len, q.device).to(dtype=q.dtype)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=mask, dropout_p=0.0)
    return out.squeeze(0).transpose(0, 1)


def build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    """Build cumulative query offsets for a packed-query view.

    Args:
        lengths: Per-request query lengths. Example: `[4, 2, 1]` becomes
            `[0, 4, 6, 7]`. During decode, batch size 3 becomes `[0, 1, 2, 3]`.
    """
    # Packed-query style metadata: request i occupies
    # `[cu_seqlens[i], cu_seqlens[i + 1])` in a flattened query buffer.
    cu_seqlens = torch.zeros(lengths.shape[0] + 1, device=lengths.device, dtype=torch.int32)
    if lengths.numel() > 0:
        cu_seqlens[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    return cu_seqlens



def paged_sdpa_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    past_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Readable direct-paged attention used by v2.

    This function is deliberately not fast. It teaches the core idea of paged
    attention before later versions introduce optimized kernels:

    - K/V vectors live in fixed-size physical pages.
    - Each request owns a logical-to-physical `block_table`.
    - Attention reads only the pages touched by that request's current sequence.

    Example with `block_size = 16` and `block_table[row] = [7, 12]`:

        logical token 0..15  -> physical block 7
        logical token 16..31 -> physical block 12

    If a query token can see keys 0..18, this function reads all of block 7 and
    the first three rows of block 12. It never first builds a dense
    `[cached_prefix | current_chunk]` K/V tensor.
    """
    batch_size, max_query_len, num_attention_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    group_size = num_attention_heads // num_kv_heads
    scale = head_dim ** -0.5
    out = torch.zeros_like(q)

    for req_idx in range(batch_size):
        query_len = int(query_lens[req_idx].item())
        key_len = int(key_lens[req_idx].item())
        past_len = int(past_lens[req_idx].item())
        if query_len == 0 or key_len == 0:
            continue

        q_req = q[req_idx, :query_len]
        q_grouped = (
            q_req.view(query_len, num_kv_heads, group_size, head_dim)
            .permute(1, 2, 0, 3)
            .to(torch.float32)
        )
        running_max = torch.full(
            (num_kv_heads, group_size, query_len),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        running_lse = torch.zeros(
            (num_kv_heads, group_size, query_len),
            device=q.device,
            dtype=torch.float32,
        )
        running_out = torch.zeros(
            (num_kv_heads, group_size, query_len, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        query_positions = past_len + torch.arange(query_len, device=q.device, dtype=torch.long)
        blocks_needed = (key_len + block_size - 1) // block_size

        for logical_block in range(blocks_needed):
            physical_block = int(block_tables[req_idx, logical_block].item())
            if physical_block < 0:
                raise RuntimeError("Missing block table entry for paged attention")
            block_start = logical_block * block_size
            block_end = min(block_start + block_size, key_len)
            tokens_in_block = block_end - block_start
            if tokens_in_block <= 0:
                continue

            k_block = k_cache[physical_block, :tokens_in_block].permute(1, 0, 2).to(torch.float32)
            v_block = v_cache[physical_block, :tokens_in_block].permute(1, 0, 2).to(torch.float32)
            scores = torch.einsum("hgtd,hsd->hgts", q_grouped, k_block) * scale

            key_positions = torch.arange(block_start, block_end, device=q.device, dtype=torch.long)
            causal = key_positions.view(1, 1, 1, tokens_in_block) <= query_positions.view(
                1, 1, query_len, 1
            )
            scores = scores.masked_fill(~causal, float("-inf"))

            # Online softmax update. Each block contributes to the same final
            # softmax as if we had materialized every key first, but only one
            # page-sized K/V slice is live at a time.
            block_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            prev_scale = torch.exp(running_max - new_max)
            block_scale = torch.exp(scores - new_max.unsqueeze(-1))
            running_lse = prev_scale * running_lse + block_scale.sum(dim=-1)
            running_out = (
                prev_scale.unsqueeze(-1) * running_out
                + torch.einsum("hgts,hsd->hgtd", block_scale, v_block)
            )
            running_max = new_max

        req_out = running_out / running_lse.unsqueeze(-1)
        req_out = req_out.permute(2, 0, 1, 3).reshape(query_len, num_attention_heads, head_dim)
        out[req_idx, :query_len] = req_out.to(dtype=q.dtype)

    return out


def swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """Apply the SwiGLU activation used by Llama-style MLPs."""
    return F.silu(x_gate) * x_up


def describe_kernel_stack(device: str) -> str:
    """Summarize which backend stack this teaching engine will exercise."""
    if device.startswith("cuda"):
        return (
            "Linear layers dispatch GEMM kernels via PyTorch/cuBLAS, while "
            "v2 attention is the readable direct-paged reference: K/V rows are "
            "written to cache pages first, then attention walks each request's "
            "block table without materializing dense K/V tensors."
        )
    return "CPU fallback: same software flow, without CUDA kernel dispatch."
