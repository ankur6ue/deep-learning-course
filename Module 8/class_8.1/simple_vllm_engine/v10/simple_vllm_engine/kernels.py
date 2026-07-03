from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# This file contains the native kernels used by the optimized teaching engine:
# decode metadata preparation, GPU decode-state update, cached RoPE,
# RMSNorm/fused add+RMSNorm, SwiGLU, and paged K/V writes. Fallback RoPE remains
# as readable PyTorch code.
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


@triton.jit
def _prepare_decode_inputs_kernel(
    req_slots_ptr,
    cached_seq_lens_ptr,
    block_tables_ptr,
    last_token_ids_ptr,
    input_ids_ptr,
    positions_ptr,
    past_lens_ptr,
    key_lens_ptr,
    seq_lens_i32_ptr,
    slot_mapping_ptr,
    block_tables_i32_ptr,
    actual_count: tl.constexpr,
    max_blocks: tl.constexpr,
    block_size: tl.constexpr,
    pad_token_id: tl.constexpr,
    scratch_block_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    active = row < actual_count

    past_len = tl.load(cached_seq_lens_ptr + row, mask=active, other=0)
    # vLLM-style graph padding:
    #
    # Active rows are real decode requests, so appending one query token means
    # the key length is `past_len + 1`.
    #
    # Inactive rows are only there to satisfy the fixed CUDA graph shape. They
    # should not look like real one-token requests, and they must not write K/V.
    # Use key_len=0 plus slot_mapping=-1, matching vLLM's PAD_SLOT_ID behavior.
    key_len = tl.where(active, past_len + 1, 0)
    req_slot = tl.load(req_slots_ptr + row, mask=active, other=0)
    token = tl.load(last_token_ids_ptr + req_slot, mask=active, other=pad_token_id)

    logical_block = past_len // block_size
    block_offset = past_len - logical_block * block_size
    physical_block = tl.load(
        block_tables_ptr + row * max_blocks + logical_block,
        mask=active,
        other=scratch_block_id,
    )
    physical_slot = physical_block * block_size + block_offset

    tl.store(input_ids_ptr + row, token)
    tl.store(positions_ptr + row, past_len)
    tl.store(past_lens_ptr + row, past_len)
    tl.store(key_lens_ptr + row, key_len)
    tl.store(seq_lens_i32_ptr + row, key_len)
    tl.store(slot_mapping_ptr + row, tl.where(active, physical_slot, -1))

    offsets = tl.arange(0, BLOCK_N)
    block_mask = offsets < max_blocks
    block_ids = tl.load(
        block_tables_ptr + row * max_blocks + offsets,
        mask=block_mask,
        other=scratch_block_id,
    )
    tl.store(block_tables_i32_ptr + row * max_blocks + offsets, block_ids, mask=block_mask)


def prepare_decode_inputs(
    *,
    req_slots: torch.Tensor,
    cached_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    last_token_ids: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    past_lens: torch.Tensor,
    key_lens: torch.Tensor,
    seq_lens_i32: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables_i32: torch.Tensor,
    actual_count: int,
    block_size: int,
    pad_token_id: int,
    scratch_block_id: int,
) -> None:
    """Fill decode input/metadata buffers with one Triton kernel.

    Most arguments here are output workspaces, not source data. The reusable
    tensors `input_ids`, `positions`, `past_lens`, `key_lens`, `seq_lens_i32`,
    `slot_mapping`, and `block_tables_i32` may contain zeros or stale values
    before this call. This kernel overwrites the active rows for the current
    decode step.

    Source-of-truth inputs:

    - `req_slots`: maps each decode batch row to a stable GPU decode-state slot.
    - `cached_seq_lens`: current sequence length before appending this token.
    - `block_tables`: per-request logical block -> physical block mapping.
    - `last_token_ids`: latest sampled token per active request slot, stored on
      GPU so decode does not need a synchronous CPU token readback.

    For each active request row the kernel derives and writes:

    - `input_ids`: next token id, read from `last_token_ids[request_slot]`
    - `positions`: absolute decode position
    - `past_lens` / `key_lens`
    - `slot_mapping`: physical KV cache slot for the new token
    - `block_tables_i32`: FlashAttention-friendly block table dtype

    CUDA graph padding follows vLLM's convention: rows beyond `actual_count`
    are shape padding, not fake decode requests. The kernel writes:

        input_ids[row] = pad_token_id
        positions[row] = 0
        key_lens[row] = 0
        slot_mapping[row] = -1

    The `-1` slot is important. The KV-write kernel treats it as
    "do not write", so padded graph rows cannot mutate a scratch cache page or
    introduce batch-shape-dependent cache history.

    Example with `block_size=16`:

        req_slots[row] = 3
        last_token_ids[3] = 1287
        cached_seq_lens[row] = 42
        block_tables[row] = [7, 12, 31]

    Then the kernel writes:

        input_ids[row] = 1287
        positions[row] = 42
        past_lens[row] = 42
        key_lens[row] = 43

        logical_block = 42 // 16 = 2
        block_offset = 42 % 16 = 10
        physical_block = block_tables[row, 2] = 31
        slot_mapping[row] = 31 * 16 + 10 = 506

    Doing this on GPU avoids per-token CPU reads/writes in the decode loop.
    """
    total_rows, max_blocks = block_tables.shape
    block_n = triton.next_power_of_2(max_blocks)
    _prepare_decode_inputs_kernel[(total_rows,)](
        req_slots,
        cached_seq_lens,
        block_tables,
        last_token_ids,
        input_ids.view(-1),
        positions.view(-1),
        past_lens,
        key_lens,
        seq_lens_i32,
        slot_mapping,
        block_tables_i32,
        actual_count,
        max_blocks,
        block_size,
        pad_token_id,
        scratch_block_id,
        BLOCK_N=block_n,
    )


@triton.jit
def _prepare_decode_inputs_from_gpu_block_table_kernel(
    req_slots_ptr,
    cached_seq_lens_ptr,
    request_block_tables_ptr,
    last_token_ids_ptr,
    input_ids_ptr,
    positions_ptr,
    past_lens_ptr,
    key_lens_ptr,
    seq_lens_i32_ptr,
    slot_mapping_ptr,
    block_tables_ptr,
    block_tables_i32_ptr,
    actual_count: tl.constexpr,
    output_max_blocks: tl.constexpr,
    request_block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    pad_token_id: tl.constexpr,
    scratch_block_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    active = row < actual_count

    past_len = tl.load(cached_seq_lens_ptr + row, mask=active, other=0)
    key_len = tl.where(active, past_len + 1, 0)
    req_slot = tl.load(req_slots_ptr + row, mask=active, other=0)
    token = tl.load(last_token_ids_ptr + req_slot, mask=active, other=pad_token_id)

    # `request_block_tables` is the persistent GPU table:
    #
    #   request slot -> logical KV block -> physical KV block
    #
    # The decode batch only carries request slots and sequence lengths. This
    # kernel gathers each active request's block-table row on GPU instead of
    # asking Python to rebuild and copy a rectangular CPU block table every
    # decode step.
    request_row_base = request_block_tables_ptr + req_slot * request_block_table_stride
    logical_block = past_len // block_size
    block_offset = past_len - logical_block * block_size
    physical_block = tl.load(
        request_row_base + logical_block,
        mask=active,
        other=scratch_block_id,
    )
    physical_slot = physical_block * block_size + block_offset

    tl.store(input_ids_ptr + row, token)
    tl.store(positions_ptr + row, past_len)
    tl.store(past_lens_ptr + row, past_len)
    tl.store(key_lens_ptr + row, key_len)
    tl.store(seq_lens_i32_ptr + row, key_len)
    tl.store(slot_mapping_ptr + row, tl.where(active, physical_slot, -1))

    offsets = tl.arange(0, BLOCK_N)
    block_mask = offsets < output_max_blocks
    block_ids = tl.load(
        request_row_base + offsets,
        mask=block_mask & active,
        other=scratch_block_id,
    )
    output_base = row * output_max_blocks + offsets
    tl.store(block_tables_ptr + output_base, block_ids, mask=block_mask)
    tl.store(block_tables_i32_ptr + output_base, block_ids, mask=block_mask)


def prepare_decode_inputs_from_gpu_block_table(
    *,
    req_slots: torch.Tensor,
    cached_seq_lens: torch.Tensor,
    request_block_tables: torch.Tensor,
    last_token_ids: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    past_lens: torch.Tensor,
    key_lens: torch.Tensor,
    seq_lens_i32: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    block_tables_i32: torch.Tensor,
    actual_count: int,
    block_size: int,
    pad_token_id: int,
    scratch_block_id: int,
) -> None:
    """Prepare decode metadata from a persistent GPU request block table.

    `prepare_decode_inputs()` in `v8` still receives a batch-shaped
    `block_tables` tensor that Python rebuilt on CPU and copied to GPU each
    decode step. This variant uses the same output tensors, but changes the
    source of truth:

    - CPU sends only `req_slots` and `cached_seq_lens` for this decode batch.
    - `request_block_tables` already lives on GPU and is keyed by stable request
      slot, not by transient batch row.
    - The Triton kernel gathers the active rows from `request_block_tables` into
      the batch-shaped `block_tables` / `block_tables_i32` tensors used by
      FlashAttention.

    Example with `block_size=16`:

        request slot 5 has GPU row [7, 12, 31, scratch, ...]
        decode batch row 0 has req_slots[0] = 5
        cached_seq_lens[0] = 42

    The kernel derives:

        input_ids[0]    = last_token_ids[5]
        positions[0]    = 42
        key_lens[0]     = 43
        slot_mapping[0] = 31 * 16 + (42 % 16)
        block_tables[0] = [7, 12, 31, scratch, ...]

    This keeps the same general paged-cache abstraction. It only moves the
    repetitive "copy each request's block table into a batch table" work from
    Python/H2D copies to one GPU kernel.
    """
    total_rows, output_max_blocks = block_tables.shape
    block_n = triton.next_power_of_2(output_max_blocks)
    _prepare_decode_inputs_from_gpu_block_table_kernel[(total_rows,)](
        req_slots,
        cached_seq_lens,
        request_block_tables,
        last_token_ids,
        input_ids.view(-1),
        positions.view(-1),
        past_lens,
        key_lens,
        seq_lens_i32,
        slot_mapping,
        block_tables,
        block_tables_i32,
        actual_count,
        output_max_blocks,
        request_block_tables.shape[1],
        block_size,
        pad_token_id,
        scratch_block_id,
        BLOCK_N=block_n,
    )


@triton.jit
def _update_decode_state_kernel(
    req_slots_ptr,
    next_tokens_ptr,
    last_token_ids_ptr,
    count: tl.constexpr,
):
    idx = tl.program_id(0)
    active = idx < count
    req_slot = tl.load(req_slots_ptr + idx, mask=active, other=0)
    token = tl.load(next_tokens_ptr + idx, mask=active, other=0)
    tl.store(last_token_ids_ptr + req_slot, token, mask=active)


def update_decode_state(
    *,
    req_slots: torch.Tensor,
    next_tokens: torch.Tensor,
    last_token_ids: torch.Tensor,
    count: int,
) -> None:
    """Update `last_token_ids[request_slot]` without torch index_copy overhead."""
    if count == 0:
        return
    _update_decode_state_kernel[(count,)](
        req_slots,
        next_tokens.view(-1),
        last_token_ids,
        count,
    )


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
    if rope_type == "proportional":
        # Gemma 4 full-attention layers use "proportional" RoPE. This is
        # subtly different from the usual "rotate the first N dimensions" form
        # of partial RoPE.
        #
        # Example for head_dim=512 and partial_rotary_factor=0.25:
        #
        #   normal partial RoPE would rotate dimensions 0..127
        #   proportional RoPE rotates 64 pairs across the full Neox split:
        #       dim 0 with dim 256
        #       dim 1 with dim 257
        #       ...
        #       dim 63 with dim 319
        #
        # The remaining pairs get frequency 0, so cos=1 and sin=0. The same
        # Triton RoPE kernel can then process the full head dimension while
        # leaving the non-rotary pairs unchanged.
        rope_angles = int(partial_rotary_factor * head_size // 2)
        rotated = 1.0 / (
            base
            ** (
                torch.arange(0, 2 * rope_angles, 2, device=device, dtype=torch.float32)
                / head_size
            )
        )
        nope_angles = head_size // 2 - rope_angles
        if nope_angles > 0:
            zeros = torch.zeros(nope_angles, device=device, dtype=torch.float32)
            inv_freq = torch.cat((rotated, zeros), dim=0)
        else:
            inv_freq = rotated
        inv_freq = inv_freq / float(rope_parameters.get("factor", 1.0))
        rotary_dim = head_size
    elif rope_type == "llama3":
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
def _qk_rope_neox_from_cache_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_row_stride: tl.constexpr,
    k_row_stride: tl.constexpr,
    q_out_row_stride: tl.constexpr,
    k_out_row_stride: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
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
    cache_base = cos_sin_cache_ptr + pos * rotary_dim
    cos = tl.load(cache_base + offsets, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(cache_base + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    if head_idx < num_q_heads:
        q_base = q_ptr + token_idx * q_row_stride + head_idx * head_dim
        q_out_base = q_out_ptr + token_idx * q_out_row_stride + head_idx * head_dim
        q0 = tl.load(q_base + offsets, mask=mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_base + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(q_out_base + offsets, q0 * cos - q1 * sin, mask=mask)
        tl.store(q_out_base + half_dim + offsets, q1 * cos + q0 * sin, mask=mask)

        pass_offsets = tl.arange(0, BLOCK_PASS)
        pass_mask = pass_offsets < (head_dim - rotary_dim)
        pass_values = tl.load(q_base + rotary_dim + pass_offsets, mask=pass_mask, other=0.0)
        tl.store(q_out_base + rotary_dim + pass_offsets, pass_values, mask=pass_mask)

    if head_idx < num_k_heads:
        k_base = k_ptr + token_idx * k_row_stride + head_idx * head_dim
        k_out_base = k_out_ptr + token_idx * k_out_row_stride + head_idx * head_dim
        k0 = tl.load(k_base + offsets, mask=mask, other=0.0).to(tl.float32)
        k1 = tl.load(k_base + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(k_out_base + offsets, k0 * cos - k1 * sin, mask=mask)
        tl.store(k_out_base + half_dim + offsets, k1 * cos + k0 * sin, mask=mask)

        pass_offsets = tl.arange(0, BLOCK_PASS)
        pass_mask = pass_offsets < (head_dim - rotary_dim)
        pass_values = tl.load(k_base + rotary_dim + pass_offsets, mask=pass_mask, other=0.0)
        tl.store(k_out_base + rotary_dim + pass_offsets, pass_values, mask=pass_mask)


def triton_apply_qk_rope_from_cache(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    q_num_heads: int,
    k_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply cached Neox/Llama RoPE to Q and K in one Triton launch."""
    if q_flat.device.type != "cuda" or q_flat.device != k_flat.device:
        return None
    q_shape = q_flat.shape
    k_shape = k_flat.shape
    if q_shape[:-1] != k_shape[:-1]:
        return None
    if q_shape[-1] != q_num_heads * head_dim:
        return None
    if k_shape[-1] != k_num_heads * head_dim:
        return None

    q_2d = q_flat.reshape(-1, q_shape[-1])
    k_2d = k_flat.reshape(-1, k_shape[-1])
    flat_positions = positions.reshape(-1).contiguous()
    if flat_positions.numel() != q_2d.shape[0]:
        return None

    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        return None

    q_out = torch.empty_like(q_2d)
    k_out = torch.empty_like(k_2d)
    block_half = triton.next_power_of_2(rotary_dim // 2)
    pass_dim = head_dim - rotary_dim
    block_pass = 1 if pass_dim <= 0 else triton.next_power_of_2(pass_dim)
    _qk_rope_neox_from_cache_kernel[(q_2d.shape[0], max(q_num_heads, k_num_heads))](
        q_2d,
        k_2d,
        q_out,
        k_out,
        flat_positions,
        cos_sin_cache,
        q_2d.stride(0),
        k_2d.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        q_num_heads,
        k_num_heads,
        head_dim,
        rotary_dim,
        BLOCK_HALF=block_half,
        BLOCK_PASS=block_pass,
    )
    return q_out.view(q_shape), k_out.view(k_shape)


@triton.jit
def _gemma_qkv_norm_rope_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_row_stride: tl.constexpr,
    k_row_stride: tl.constexpr,
    v_row_stride: tl.constexpr,
    q_out_row_stride: tl.constexpr,
    k_out_row_stride: tl.constexpr,
    v_out_row_stride: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    apply_rope: tl.constexpr,
    kv_shared: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < head_dim
    q_head_mask = head_idx < num_q_heads
    kv_head_mask = head_idx < num_kv_heads
    q_mask = mask & q_head_mask
    kv_mask = mask & kv_head_mask

    q_base = q_ptr + token_idx * q_row_stride + head_idx * head_dim
    q_out_base = q_out_ptr + token_idx * q_out_row_stride + head_idx * head_dim
    q = tl.load(q_base + offsets, mask=q_mask, other=0.0).to(tl.float32)
    q_weight = tl.load(q_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q_var = tl.sum(q * q, axis=0) / head_dim
    q_norm = q * tl.rsqrt(q_var + eps) * q_weight
    if apply_rope:
        tl.store(q_out_base + offsets, q_norm, mask=q_mask & (offsets >= rotary_dim))
    else:
        tl.store(q_out_base + offsets, q_norm, mask=q_mask)

    k_base = k_ptr + token_idx * k_row_stride + head_idx * head_dim
    k_out_base = k_out_ptr + token_idx * k_out_row_stride + head_idx * head_dim
    k = tl.load(k_base + offsets, mask=kv_mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    k_var = tl.sum(k * k, axis=0) / head_dim
    k_inv = tl.rsqrt(k_var + eps)
    k_norm = k * k_inv * k_weight
    if apply_rope:
        tl.store(k_out_base + offsets, k_norm, mask=kv_mask & (offsets >= rotary_dim))
    else:
        tl.store(k_out_base + offsets, k_norm, mask=kv_mask)

    if apply_rope:
        half_offsets = tl.arange(0, BLOCK_HALF)
        half_dim = rotary_dim // 2
        half_mask = half_offsets < half_dim
        pos = tl.load(positions_ptr + token_idx)
        cache_base = cos_sin_cache_ptr + pos * rotary_dim
        cos = tl.load(cache_base + half_offsets, mask=half_mask, other=1.0).to(tl.float32)
        sin = tl.load(
            cache_base + half_dim + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)

        q_half_mask = half_mask & q_head_mask
        q0 = tl.load(q_base + half_offsets, mask=q_half_mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_base + half_dim + half_offsets, mask=q_half_mask, other=0.0).to(tl.float32)
        qw0 = tl.load(q_weight_ptr + half_offsets, mask=half_mask, other=0.0).to(tl.float32)
        qw1 = tl.load(
            q_weight_ptr + half_dim + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        q_inv = tl.rsqrt(q_var + eps)
        q0 = q0 * q_inv * qw0
        q1 = q1 * q_inv * qw1
        tl.store(q_out_base + half_offsets, q0 * cos - q1 * sin, mask=q_half_mask)
        tl.store(
            q_out_base + half_dim + half_offsets,
            q1 * cos + q0 * sin,
            mask=q_half_mask,
        )

        kv_half_mask = half_mask & kv_head_mask
        k0 = tl.load(k_base + half_offsets, mask=kv_half_mask, other=0.0).to(tl.float32)
        k1 = tl.load(k_base + half_dim + half_offsets, mask=kv_half_mask, other=0.0).to(tl.float32)
        kw0 = tl.load(k_weight_ptr + half_offsets, mask=half_mask, other=0.0).to(tl.float32)
        kw1 = tl.load(
            k_weight_ptr + half_dim + half_offsets,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        k0 = k0 * k_inv * kw0
        k1 = k1 * k_inv * kw1
        tl.store(k_out_base + half_offsets, k0 * cos - k1 * sin, mask=kv_half_mask)
        tl.store(
            k_out_base + half_dim + half_offsets,
            k1 * cos + k0 * sin,
            mask=kv_half_mask,
        )

    v_out_base = v_out_ptr + token_idx * v_out_row_stride + head_idx * head_dim
    if kv_shared:
        v_norm = k * k_inv
    else:
        v_base = v_ptr + token_idx * v_row_stride + head_idx * head_dim
        v = tl.load(v_base + offsets, mask=kv_mask, other=0.0).to(tl.float32)
        v_var = tl.sum(v * v, axis=0) / head_dim
        v_norm = v * tl.rsqrt(v_var + eps)
    tl.store(v_out_base + offsets, v_norm, mask=kv_mask)


def triton_gemma_qkv_norm_rope(
    q_flat: torch.Tensor,
    k_flat_raw: torch.Tensor,
    v_flat_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    apply_rope: bool,
    kv_shared: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fuse Gemma per-head Q/K/V RMSNorm, optionally applying RoPE to Q/K.

    Q and K use learned per-head RMSNorm weights. V is unweighted. When Gemma's
    full-attention layer reuses K as raw V, `kv_shared=True` lets the kernel use
    K's already-computed variance for V normalization.
    """
    if q_flat.device.type != "cuda":
        return None
    if q_flat.shape[:-1] != k_flat_raw.shape[:-1] or k_flat_raw.shape != v_flat_raw.shape:
        return None
    if q_flat.shape[-1] != num_q_heads * head_dim:
        return None
    if k_flat_raw.shape[-1] != num_kv_heads * head_dim:
        return None
    if q_weight.shape[0] != head_dim or k_weight.shape[0] != head_dim:
        return None

    q_2d = q_flat.reshape(-1, q_flat.shape[-1])
    k_2d = k_flat_raw.reshape(-1, k_flat_raw.shape[-1])
    v_2d = v_flat_raw.reshape(-1, v_flat_raw.shape[-1])
    flat_positions = positions.reshape(-1).contiguous()
    if apply_rope and flat_positions.numel() != q_2d.shape[0]:
        return None

    rotary_dim = cos_sin_cache.shape[-1]
    if apply_rope and (rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0):
        return None

    q_out = torch.empty_like(q_2d)
    k_out = torch.empty_like(k_2d)
    v_out = torch.empty_like(v_2d)
    block_d = triton.next_power_of_2(head_dim)
    block_half = 1 if rotary_dim <= 0 else triton.next_power_of_2(rotary_dim // 2)
    _gemma_qkv_norm_rope_kernel[(q_2d.shape[0], max(num_q_heads, num_kv_heads))](
        q_2d,
        k_2d,
        v_2d,
        q_weight.contiguous(),
        k_weight.contiguous(),
        flat_positions,
        cos_sin_cache,
        q_out,
        k_out,
        v_out,
        q_2d.stride(0),
        k_2d.stride(0),
        v_2d.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        v_out.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        apply_rope,
        kv_shared,
        BLOCK_D=block_d,
        BLOCK_HALF=block_half,
    )
    return q_out.view_as(q_flat), k_out.view_as(k_flat_raw), v_out.view_as(v_flat_raw)


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
def _rms_norm_no_weight_kernel(
    x_ptr,
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
    variance = tl.sum(x * x, axis=0) / hidden_size
    out = x * tl.rsqrt(variance + eps)
    tl.store(out_ptr + row * row_stride + offsets, out, mask=mask)


def triton_rms_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Run RMSNorm without a learned weight using a native Triton kernel."""
    if x.device.type != "cuda":
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    hidden_size = x_2d.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    _rms_norm_no_weight_kernel[(x_2d.shape[0],)](
        x_2d,
        out,
        x_2d.stride(0),
        hidden_size,
        eps,
        BLOCK_H=block_h,
    )
    return out.view_as(x)


@triton.jit
def _gemma_post_norm_residual_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    x_row_stride: tl.constexpr,
    residual_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    apply_scale: tl.constexpr,
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
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    out = x * tl.rsqrt(variance + eps) * weight + residual
    if apply_scale:
        scale = tl.load(scale_ptr).to(tl.float32)
        out = out * scale
    tl.store(out_ptr + row * out_row_stride + offsets, out, mask=mask)


def triton_gemma_post_norm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fuse Gemma post-branch RMSNorm, residual add, and optional layer scalar."""
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
    hidden_size = x_2d.shape[-1]
    block_h = triton.next_power_of_2(hidden_size)
    scale_arg = weight if scale is None else scale.contiguous()
    _gemma_post_norm_residual_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight,
        scale_arg,
        out,
        x_2d.stride(0),
        residual_2d.stride(0),
        out.stride(0),
        hidden_size,
        eps,
        scale is not None,
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
def _gelu_tanh_and_mul_kernel(
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
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    gelu = 0.5 * gate * (1.0 + tanh_inner)
    tl.store(out_ptr + row * out_row_stride + offsets, gelu * up, mask=mask)


def triton_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor | None:
    """Run Gemma GELU-tanh gated activation with a native Triton kernel.

    Input layout is `[... , 2 * intermediate_size]`, with `[gate | up]` in the
    last dimension. The output is `gelu(gate, approximate="tanh") * up`.
    """
    if x.device.type != "cuda" or x.shape[-1] % 2 != 0:
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    half_size = x_2d.shape[-1] // 2
    out = torch.empty((x_2d.shape[0], half_size), device=x.device, dtype=x.dtype)
    block_h = triton.next_power_of_2(half_size)
    _gelu_tanh_and_mul_kernel[(x_2d.shape[0],)](
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


def describe_kernel_stack(device: str, attention_backend: str | None = None) -> str:
    """Summarize which backend stack this teaching engine will exercise."""
    del attention_backend
    if not device.startswith("cuda"):
        return "CPU fallback: same software flow, without CUDA kernel dispatch."
    return (
        "Linear layers dispatch GEMM kernels via PyTorch/cuBLAS, while "
        "FlashAttention varlen handles paged prefill/decode attention from "
        "block tables. v9 keeps decode graph replay and gathers decode "
        "block-table metadata from persistent GPU request rows."
    )
