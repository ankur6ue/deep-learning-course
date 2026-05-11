from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def repeat_kv(kv: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads.

    Args:
        kv: Tensor shaped `[T, Hkv, D]`. If `Hkv < Hq`, the same KV heads are
            repeated as in grouped-query attention.
        num_attention_heads: Target number of query heads `Hq`.
    """
    # [T, Hkv, D] -> [T, Hq, D]
    if kv.shape[1] == num_attention_heads:
        return kv
    num_kv_heads = kv.shape[1]
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    repeats = num_attention_heads // num_kv_heads
    return kv.repeat_interleave(repeats, dim=1)


def rotary_frequencies(
    seq_positions: torch.Tensor,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cosine and sine tables for the requested positions.

    Args:
        seq_positions: Absolute token positions such as `[0, 1, 2]` or, during
            chunked prefill, `[5, 6, 7, 8]`.
        head_dim: Per-head dimension. Must be even because RoPE rotates pairs
            of channels.
        theta: RoPE base frequency.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=seq_positions.device).float() / head_dim))
    freqs = torch.outer(seq_positions.float(), inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply rotary position embedding to Q or K.

    Args:
        x: Tensor shaped `[B, T, H, D]`.
        positions: Absolute token positions for each batch element. For a later
            prefill chunk, these positions continue from the earlier chunk.
        theta: RoPE base frequency.
    """
    # x: [B, T, H, D]
    cos, sin = rotary_frequencies(positions.view(-1), x.shape[-1], theta)
    cos = cos.view(*positions.shape, 1, -1)
    sin = sin.view(*positions.shape, 1, -1)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = x_rot_even
    out[..., 1::2] = x_rot_odd
    return out


def causal_attn_mask(query_len: int, key_len: int, past_len: int, device: torch.device) -> torch.Tensor:
    """Build an additive causal mask for one request.

    Args:
        query_len: Number of query tokens in the current step.
        key_len: Total visible KV tokens, including cached prefix plus current
            chunk.
        past_len: Number of cached prefix tokens already in the KV cache. If
            `past_len=5` and `query_len=4`, query token 0 can see keys `0..5`
            and query token 3 can see `0..8`.
        device: Target device for the mask tensor.
    """
    # Query token i in the current chunk can see all past tokens and query
    # tokens <= i. This returns `[Tq, Tk]` because v1 calls attention one
    # request at a time, so there is no batch dimension and no need to
    # broadcast across heads here.
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
            chunk. This is what lets chunked prefill remain causal.
    """
    # q: [Tq, Hq, D], k/v: [Tk, Hq, D]
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    # SDPA consumes `[B, H, Tq, Tk]`-style inputs, so the per-request tensors
    # are reshaped to batch size 1 here. Unlike v2, v1 does not need a padded-
    # query cleanup step after attention because this helper only sees the real
    # query tokens for one request, not a padded batch with invalid query rows.
    mask = causal_attn_mask(q.shape[0], k.shape[0], past_len, q.device).to(dtype=q.dtype)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=mask, dropout_p=0.0)
    return out.squeeze(0).transpose(0, 1)


def swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """Apply the SwiGLU activation used by Llama-style MLPs."""
    return F.silu(x_gate) * x_up


def describe_kernel_stack(device: str) -> str:
    """Summarize which backend stack this teaching engine will exercise."""
    if device.startswith("cuda"):
        return (
            "Linear layers dispatch GEMM kernels via PyTorch/cuBLAS, while "
            "scaled_dot_product_attention dispatches CUDA attention kernels "
            "when supported. Page gather/write remains explicit serving logic."
        )
    return "CPU fallback: same software flow, without CUDA kernel dispatch."
