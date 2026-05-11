from __future__ import annotations

import importlib
import math
from typing import Any

import torch
import torch.nn.functional as F


_VLLM_UNIFIED_ATTENTION = None
_VLLM_UNIFIED_ATTENTION_LOAD_ATTEMPTED = False


def repeat_kv(kv: torch.Tensor, num_attention_heads: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads.

    Args:
        kv: Tensor shaped `[..., Hkv, D]`. In v2 this may be a full batched
            tensor such as `[B, T, Hkv, D]`.
        num_attention_heads: Target number of query heads `Hq`.
    """
    # [..., Hkv, D] -> [..., Hq, D]
    if kv.shape[-2] == num_attention_heads:
        return kv
    num_kv_heads = kv.shape[-2]
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    repeats = num_attention_heads // num_kv_heads
    return kv.repeat_interleave(repeats, dim=-2)


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


def batched_causal_attn_mask(
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    past_lens: torch.Tensor,
    max_query_len: int,
    max_key_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build one additive causal mask for an entire padded request batch.

    Args:
        query_lens: Query tokens being processed now for each request.
        key_lens: Total visible KV tokens per request after appending this
            chunk.
        past_lens: Cached prefix tokens per request before this chunk.
        max_query_len: Padding width for the query dimension in this batch.
        max_key_len: Padding width for the key/value dimension in this batch.
        device: Device on which to allocate the mask.
    """
    batch_size = query_lens.shape[0]
    q_idx = torch.arange(max_query_len, device=device).view(1, max_query_len, 1)
    k_idx = torch.arange(max_key_len, device=device).view(1, 1, max_key_len)
    # `query_lens` is only the current chunk length per request. `key_lens`
    # includes both the cached prefix and the current chunk.
    q_valid = q_idx < query_lens.view(batch_size, 1, 1)
    k_valid = k_idx < key_lens.view(batch_size, 1, 1)
    # Query token j in the current chunk can see all cached prefix tokens plus
    # chunk-local query tokens up to j.
    causal = k_idx <= (past_lens.view(batch_size, 1, 1) + q_idx)
    disallowed = (q_valid & ~(k_valid & causal)).unsqueeze(1)
    # SDPA consumes `[B, H, Tq, Tk]`, so we build `[B, 1, Tq, Tk]` here and let
    # it broadcast across attention heads. This is not `[B, Tk, Tk]` because in
    # general query length and key length differ during chunked prefill and
    # decode.
    #
    # Floating-point additive mask for SDPA: 0 means visible, -inf means mask.
    mask = torch.zeros((batch_size, 1, max_query_len, max_key_len), device=device, dtype=torch.float32)
    mask = mask.masked_fill(disallowed, float("-inf"))
    return mask


def batched_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    past_lens: torch.Tensor,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Run one attention call for a padded batch of requests.

    Args:
        q: Query tensor `[B, Tq_max, Hq, D]`.
        k: Key tensor `[B, Tk_max, H, D]`, where `H` can be either `Hq` or
            `Hkv` when `enable_gqa=True`.
        v: Value tensor `[B, Tk_max, H, D]`.
        query_lens: Valid query tokens per request in this step.
        key_lens: Valid visible KV tokens per request.
        past_lens: Cached prefix length per request before the current chunk.
        enable_gqa: If true, let SDPA handle grouped-query attention directly
            instead of explicitly repeating KV heads.
    """
    # SDPA expects `[B, H, T, D]`; the rest of the engine keeps `[B, T, H, D]`
    # because that is easier to reason about alongside token positions.
    q_b = q.permute(0, 2, 1, 3)
    k_b = k.permute(0, 2, 1, 3)
    v_b = v.permute(0, 2, 1, 3)
    mask = batched_causal_attn_mask(
        query_lens=query_lens,
        key_lens=key_lens,
        past_lens=past_lens,
        max_query_len=q.shape[1],
        max_key_len=k.shape[1],
        device=q.device,
    ).to(dtype=q.dtype)
    out = F.scaled_dot_product_attention(
        q_b,
        k_b,
        v_b,
        attn_mask=mask,
        dropout_p=0.0,
        enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)
    # Zero out padded query rows explicitly. In this implementation, padded
    # query rows are treated as "don't care" rows in the mask rather than fully
    # masked rows, so this multiply is what enforces the invariant that invalid
    # query positions produce zero output in the padded batch representation.
    valid = (
        torch.arange(q.shape[1], device=q.device).unsqueeze(0) < query_lens.unsqueeze(1)
    ).to(dtype=out.dtype)
    return out * valid.unsqueeze(-1).unsqueeze(-1)


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
    """Reference paged attention that reads directly from paged KV tensors.

    Args:
        q: Query tensor shaped `[B, Tq_max, Hq, D]`.
        k_cache: Paged key cache shaped `[num_blocks, block_size, Hkv, D]`.
        v_cache: Paged value cache shaped `[num_blocks, block_size, Hkv, D]`.
        block_tables: Logical-to-physical page mapping shaped
            `[B, max_blocks_per_seq]`. Row `i` maps logical blocks of request
            `i` to physical page ids in the KV cache.
        query_lens: Valid query tokens per request in the current step.
        key_lens: Total visible KV tokens per request after appending this
            chunk.
        past_lens: Cached prefix length per request before this chunk. This is
            used to place the current query rows at their absolute logical
            positions when applying causality.
        block_size: Tokens stored in one physical KV page.

    This is still a teaching/reference implementation. It does not launch a
    custom fused paged-attention kernel. Instead, it walks the block table and
    accumulates attention block by block with an online softmax update, so the
    function consumes the paged cache directly without first materializing a
    dense `[B, Tk, ...]` KV tensor.
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


def load_vllm_triton_unified_attention():
    """Best-effort loader for vLLM's Triton unified attention op.

    The teaching engine should still run without vLLM installed, so this
    loader returns `None` when the operator is unavailable. When the user runs
    through the shared `.venv-llm`, the import usually resolves to
    `vllm.v1.attention.ops.triton_unified_attention.unified_attention`.
    """
    global _VLLM_UNIFIED_ATTENTION
    global _VLLM_UNIFIED_ATTENTION_LOAD_ATTEMPTED
    if _VLLM_UNIFIED_ATTENTION_LOAD_ATTEMPTED:
        return _VLLM_UNIFIED_ATTENTION
    _VLLM_UNIFIED_ATTENTION_LOAD_ATTEMPTED = True
    module_candidates = [
        "vllm.v1.attention.ops.triton_unified_attention",
        "vllm.attention.ops.triton_unified_attention",
    ]
    for mod_name in module_candidates:
        try:
            mod = importlib.import_module(mod_name)
            _VLLM_UNIFIED_ATTENTION = getattr(mod, "unified_attention")
            return _VLLM_UNIFIED_ATTENTION
        except Exception:
            continue
    return None


def paged_triton_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    key_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    """Run vLLM's Triton unified attention op for single-token paged decode.

    Args:
        q: Query tensor `[B, 1, Hq, D]`.
        k_cache: Paged K cache `[num_blocks, block_size, Hkv, D]`.
        v_cache: Paged V cache `[num_blocks, block_size, Hkv, D]`.
        block_tables: Logical-to-physical page map `[B, max_blocks_per_seq]`.
        key_lens: Visible KV length per request.
        block_size: Tokens per page. vLLM's Triton path requires this to be a
            multiple of 16.

    Returns:
        `[B, 1, Hq, D]` on success, or `None` when the Triton kernel is not
        available or the current shape does not satisfy its requirements.
    """
    unified_attention = load_vllm_triton_unified_attention()
    if unified_attention is None:
        return None
    if q.device.type != "cuda":
        return None
    if q.shape[1] != 1:
        return None
    if block_size % 16 != 0:
        return None
    _, _, num_attention_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    if head_dim < 32:
        return None
    if num_attention_heads % num_kv_heads != 0:
        return None

    batch_size = q.shape[0]
    q_flat = q[:, 0].contiguous()
    out_flat = torch.empty_like(q_flat)
    cu_seqlens_q = torch.arange(
        0,
        batch_size + 1,
        device=q.device,
        dtype=torch.int32,
    )
    seq_lens = key_lens.to(dtype=torch.int32).contiguous()
    block_table = block_tables.to(dtype=torch.int32).contiguous()
    descale_shape = (batch_size, num_kv_heads)
    k_descale = torch.ones(descale_shape, device=q.device, dtype=torch.float32)
    v_descale = torch.ones(descale_shape, device=q.device, dtype=torch.float32)
    unified_attention(
        q=q_flat,
        k=k_cache,
        v=v_cache,
        out=out_flat,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return out_flat.view(batch_size, 1, num_attention_heads, head_dim)


def swiglu(x_gate: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
    """Apply the SwiGLU activation used by Llama-style MLPs."""
    return F.silu(x_gate) * x_up


def describe_kernel_stack(device: str) -> str:
    """Summarize which backend stack this teaching engine will exercise."""
    if device.startswith("cuda"):
        return (
            "Linear layers dispatch GEMM kernels via PyTorch/cuBLAS, while "
            "scaled_dot_product_attention dispatches CUDA attention kernels "
            "when supported. v4 batches requests into one attention call per "
            "layer, and its paged backend can route single-token decode "
            "through vLLM's Triton unified attention op when available."
        )
    return "CPU fallback: same software flow, without CUDA kernel dispatch."
