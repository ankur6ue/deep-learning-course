from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# This file contains the native kernels used by the optimized teaching engine:
# cached RoPE, RMSNorm/fused add+RMSNorm, SwiGLU, paged K/V writes, and
# decode metadata helpers. Fallback RoPE remains as readable PyTorch code.
#
# The model/engine code should call the named Python wrappers below, not the
# underscored Triton kernels directly.


@dataclass(frozen=True)
class TritonDecodeMetadata:
    """Layer-invariant metadata for one single-token decode batch.

    The K/V cache tensor changes per transformer layer, but the decode batch's
    sequence lengths, block table, and cumulative query offsets are shared by
    every layer. Building this once per scheduler step avoids repeated Python
    tensor construction inside the layer loop.
    """

    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seqlen_k: int
    k_descale: torch.Tensor
    v_descale: torch.Tensor


def build_triton_decode_metadata(
    *,
    cu_seqlens_q: torch.Tensor,
    key_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seqlen_k: int,
    num_kv_heads: int,
) -> TritonDecodeMetadata:
    """Build layer-invariant inputs for paged decode attention.

    The K/V cache tensors are layer-specific, but sequence lengths, packed-query
    offsets, and block tables are shared by all layers in one decode batch.
    """
    batch_size = key_lens.shape[0]
    descale_shape = (batch_size, num_kv_heads)
    return TritonDecodeMetadata(
        cu_seqlens_q=cu_seqlens_q.to(dtype=torch.int32).contiguous(),
        seq_lens=key_lens.to(dtype=torch.int32).contiguous(),
        block_table=block_tables.to(dtype=torch.int32).contiguous(),
        max_seqlen_k=max_seqlen_k,
        k_descale=torch.ones(descale_shape, device=key_lens.device, dtype=torch.float32),
        v_descale=torch.ones(descale_shape, device=key_lens.device, dtype=torch.float32),
    )



def build_rope_cos_sin_cache(
    *,
    head_size: int,
    max_position: int,
    rope_theta: float,
    rope_scaling: dict[str, Any] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a vLLM-compatible RoPE cos/sin cache without vLLM RoPE modules."""
    inv_freq, attention_scaling, _ = _rope_inv_freq_and_attention_scaling(
        head_size,
        rope_theta,
        rope_scaling,
        device=torch.device("cpu"),
    )
    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * attention_scaling
    return cache.to(dtype)


def _rope_inv_freq_and_attention_scaling(
    head_size: int,
    rope_theta: float,
    rope_scaling: dict[str, Any] | None,
    *,
    device: torch.device | None,
) -> tuple[torch.Tensor, float, int]:
    """Return RoPE inverse frequencies and optional YaRN attention scaling."""
    rope_parameters = dict(rope_scaling or {})
    base = float(rope_parameters.get("rope_theta", rope_theta))
    rope_type = rope_parameters.get("rope_type", "default")
    partial_rotary_factor = float(rope_parameters.get("partial_rotary_factor", 1.0))
    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
        raise ValueError("partial_rotary_factor must be in the interval (0, 1]")
    rotary_dim = int(head_size * partial_rotary_factor)
    if rotary_dim % 2 != 0:
        raise ValueError("RoPE rotary_dim must be even")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )
    attention_scaling = 1.0
    if rope_type == "llama3":
        scaling_factor = float(rope_parameters["factor"])
        low_freq_factor = float(rope_parameters["low_freq_factor"])
        high_freq_factor = float(rope_parameters["high_freq_factor"])
        original_max_position = float(rope_parameters["original_max_position_embeddings"])

        low_freq_wavelen = original_max_position / low_freq_factor
        high_freq_wavelen = original_max_position / high_freq_factor
        wave_len = 2 * math.pi / inv_freq
        if low_freq_factor != high_freq_factor:
            smooth = (original_max_position / wave_len - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
        else:
            smooth = 0
        inv_freq = torch.where(
            wave_len < high_freq_wavelen,
            inv_freq,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freq / scaling_factor,
                (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
            ),
        )
    elif rope_type == "yarn":
        factor = float(rope_parameters["factor"])
        attention_factor = rope_parameters.get("attention_factor")
        mscale = rope_parameters.get("mscale")
        mscale_all_dim = rope_parameters.get("mscale_all_dim")
        original_max_position = float(rope_parameters["original_max_position_embeddings"])

        def get_mscale(scale: float, multiplier: float = 1.0) -> float:
            if scale <= 1.0:
                return 1.0
            return 0.1 * multiplier * math.log(scale) + 1.0

        if attention_factor is None:
            if mscale and mscale_all_dim:
                attention_scaling = get_mscale(factor, float(mscale)) / get_mscale(
                    factor,
                    float(mscale_all_dim),
                )
            else:
                attention_scaling = get_mscale(factor)
        else:
            attention_scaling = float(attention_factor)

        beta_fast = float(rope_parameters.get("beta_fast") or 32.0)
        beta_slow = float(rope_parameters.get("beta_slow") or 1.0)

        def find_correction_dim(num_rotations: float) -> float:
            return (
                rotary_dim
                * math.log(original_max_position / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        truncate = bool(rope_parameters.get("truncate", True))
        low = find_correction_dim(beta_fast)
        high = find_correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        low = max(low, 0)
        high = min(high, rotary_dim - 1)
        if low == high:
            high += 0.001

        ramp = (
            torch.arange(rotary_dim // 2, device=device, dtype=torch.float32) - low
        ) / (high - low)
        ramp = torch.clamp(ramp, 0, 1)
        extrapolation_weight = 1 - ramp

        pos_freqs = base ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        inv_freq = (
            inv_freq_interpolation * (1 - extrapolation_weight)
            + inv_freq_extrapolation * extrapolation_weight
        )
    elif rope_type not in (None, "default"):
        raise ValueError(f"Unsupported rope_scaling rope_type: {rope_type}")
    return inv_freq, attention_scaling, rotary_dim


@triton.jit
def _rope_neox_from_cache_kernel(
    x_ptr,
    out_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    x_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_PASS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    half_dim = rotary_dim // 2
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim

    pos = tl.load(positions_ptr + token_idx)
    x_base = x_ptr + token_idx * x_row_stride + head_idx * head_dim
    out_base = out_ptr + token_idx * out_row_stride + head_idx * head_dim
    cache_base = cos_sin_cache_ptr + pos * rotary_dim

    x0 = tl.load(x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_base + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cache_base + offsets, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(cache_base + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_base + offsets, x0 * cos - x1 * sin, mask=mask)
    tl.store(out_base + half_dim + offsets, x1 * cos + x0 * sin, mask=mask)

    pass_offsets = tl.arange(0, BLOCK_PASS)
    pass_mask = pass_offsets < (head_dim - rotary_dim)
    pass_values = tl.load(x_base + rotary_dim + pass_offsets, mask=pass_mask, other=0.0)
    tl.store(out_base + rotary_dim + pass_offsets, pass_values, mask=pass_mask)


def triton_apply_rope_from_cache(
    x_flat: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor | None:
    """Apply Neox/Llama RoPE using a precomputed cos/sin cache."""
    if x_flat.device.type != "cuda":
        return None
    x_shape = x_flat.shape
    if x_shape[-1] != num_heads * head_dim:
        return None

    x_2d = x_flat.reshape(-1, x_shape[-1])
    flat_positions = positions.reshape(-1).contiguous()
    if flat_positions.numel() != x_2d.shape[0]:
        return None

    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        return None

    out = torch.empty(
        (x_2d.shape[0], x_2d.shape[1]),
        device=x_2d.device,
        dtype=x_2d.dtype,
    )
    block_half = triton.next_power_of_2(rotary_dim // 2)
    pass_dim = head_dim - rotary_dim
    block_pass = 1 if pass_dim <= 0 else triton.next_power_of_2(pass_dim)
    _rope_neox_from_cache_kernel[(x_2d.shape[0], num_heads)](
        x_2d,
        out,
        flat_positions,
        cos_sin_cache,
        x_2d.stride(0),
        out.stride(0),
        head_dim,
        rotary_dim,
        BLOCK_HALF=block_half,
        BLOCK_PASS=block_pass,
    )
    return out.view(x_shape)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    row_stride: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size
    x = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    out = x * tl.rsqrt(variance + eps) * weight
    tl.store(out_ptr + row * row_stride + offsets, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Run RMSNorm with a small native Triton kernel."""
    if x.device.type != "cuda":
        return None
    if x.shape[-1] != weight.shape[0]:
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    weight = weight.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    hidden_size = x_2d.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    _rms_norm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight,
        out,
        x_2d.stride(0),
        hidden_size,
        eps,
        BLOCK_H=block_h,
    )
    return out.view_as(x)


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    out_residual_ptr,
    x_row_stride: tl.constexpr,
    residual_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    out_residual_row_stride: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size
    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * residual_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    summed = x + residual
    tl.store(
        out_residual_ptr + row * out_residual_row_stride + offsets,
        summed,
        mask=mask,
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(summed * summed, axis=0) / hidden_size
    out = summed * tl.rsqrt(variance + eps) * weight
    tl.store(out_ptr + row * out_row_stride + offsets, out, mask=mask)


def triton_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run residual add plus RMSNorm with a native Triton kernel."""
    if x.device.type != "cuda":
        return None
    if x.shape != residual.shape or x.shape[-1] != weight.shape[0]:
        return None
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    residual_2d = residual.reshape(-1, residual.shape[-1])
    out = torch.empty_like(x_2d)
    out_residual = torch.empty_like(residual_2d)
    hidden_size = x_2d.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    _fused_add_rms_norm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight,
        out,
        out_residual,
        x_2d.stride(0),
        residual_2d.stride(0),
        out.stride(0),
        out_residual.stride(0),
        hidden_size,
        eps,
        BLOCK_H=block_h,
    )
    return out.view_as(x), out_residual.view_as(residual)


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    x_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < half_size
    gate = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        x_ptr + row * x_row_stride + half_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = (gate * tl.sigmoid(gate)) * up
    tl.store(out_ptr + row * out_row_stride + offsets, out, mask=mask)


def triton_silu_and_mul(x: torch.Tensor) -> torch.Tensor | None:
    """Run Llama SwiGLU activation with a native Triton kernel.

    Input layout is `[... , 2 * intermediate_size]`, where the last dimension is
    `[gate | up]`. The output is `silu(gate) * up`.
    """
    if x.device.type != "cuda" or x.shape[-1] % 2 != 0:
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    half_size = x_2d.shape[-1] // 2
    out = torch.empty((x_2d.shape[0], half_size), device=x.device, dtype=x.dtype)
    block_h = triton.next_power_of_2(half_size)
    _silu_and_mul_kernel[(x_2d.shape[0],)](
        x_2d,
        out,
        x_2d.stride(0),
        out.stride(0),
        half_size,
        BLOCK_H=block_h,
    )
    return out.view(*x.shape[:-1], half_size)



@triton.jit
def _reshape_and_cache_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    cache_slot_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_idx, mask=token_idx < num_tokens, other=-1)
    offsets = tl.arange(0, BLOCK_D)
    mask = (token_idx < num_tokens) & (slot >= 0) & (offsets < head_dim)
    key = tl.load(
        key_ptr + token_idx * key_token_stride + head_idx * key_head_stride + offsets,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_ptr + token_idx * value_token_stride + head_idx * value_head_stride + offsets,
        mask=mask,
        other=0.0,
    )
    cache_base = slot * cache_slot_stride + head_idx * cache_head_stride + offsets
    tl.store(key_cache_ptr + cache_base, key, mask=mask)
    tl.store(value_cache_ptr + cache_base, value, mask=mask)


def triton_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> bool:
    """Write K/V tokens into the paged cache with a native Triton kernel.

    `slot_mapping[token]` is the flattened physical cache slot:

        physical_block_id * block_size + block_offset

    The wrapper returns `False` when the inputs are not compatible with the
    Triton path, allowing callers to fall back to PyTorch `index_copy_`.
    """
    if key.device.type != "cuda" or key.numel() == 0:
        return False
    if key.shape != value.shape or key.ndim != 3:
        return False
    if slot_mapping.dtype != torch.long:
        slot_mapping = slot_mapping.to(dtype=torch.long)
    if not slot_mapping.is_contiguous():
        slot_mapping = slot_mapping.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    num_tokens, num_heads, head_dim = key.shape
    if slot_mapping.numel() != num_tokens:
        return False
    if key_cache.shape[2] != num_heads or key_cache.shape[3] != head_dim:
        return False
    if value_cache.shape != key_cache.shape:
        return False
    key_cache_flat = key_cache.view(-1, num_heads, head_dim)
    value_cache_flat = value_cache.view(-1, num_heads, head_dim)
    block_d = triton.next_power_of_2(head_dim)
    _reshape_and_cache_kernel[(num_tokens, num_heads)](
        key,
        value,
        key_cache_flat,
        value_cache_flat,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_cache_flat.stride(0),
        key_cache_flat.stride(1),
        num_tokens,
        head_dim,
        BLOCK_D=block_d,
    )
    return True



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
            This implementation supports the Llama 3 and YaRN scaling metadata
            used by the local test models in addition to default RoPE.
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq, attention_scaling, rotary_dim = _rope_inv_freq_and_attention_scaling(
        head_dim,
        theta,
        rope_scaling,
        device=seq_positions.device,
    )
    freqs = torch.outer(seq_positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    if rotary_dim != head_dim:
        raise ValueError("Fallback apply_rope path does not support partial rotary dimensions")
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
    # Fallback path used on CPU or when the cached Triton RoPE path is not
    # available. It is intentionally readable, not optimized.
    cos, sin = rotary_frequencies(positions.view(-1), x.shape[-1], theta, rope_scaling=rope_scaling)
    cos = cos.view(*positions.shape, 1, -1).to(dtype=x.dtype)
    sin = sin.view(*positions.shape, 1, -1).to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


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



def swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """Apply the SwiGLU activation used by Llama-style MLPs."""
    return F.silu(x_gate) * x_up


def describe_kernel_stack(device: str) -> str:
    """Summarize which backend stack this teaching engine will exercise."""
    if device.startswith("cuda"):
        return ('FlashAttention handles paged attention from block tables. v5 adds packed projections, cached RoPE, fused local kernels, and slot-mapped cache writes.')
    return "CPU fallback: same software flow, without CUDA kernel dispatch."
