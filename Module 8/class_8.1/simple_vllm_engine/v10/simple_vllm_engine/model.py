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
    triton_gelu_tanh_and_mul,
    triton_gemma_post_norm_residual,
    triton_gemma_qkv_norm_rope,
    triton_apply_qk_rope_from_cache,
    triton_rms_norm,
    triton_rms_norm_no_weight,
    triton_apply_rope_from_cache,
    triton_silu_and_mul,
)
from .kv_cache import PagedKVCache
from .requests import RequestState


# ---------------------------------------------------------------------------
# Compile-safe tensor helpers
# ---------------------------------------------------------------------------
#
# These functions deliberately accept only tensors, weights, and scalar
# constants. They do not touch the request objects, KV cache allocator, block
# tables, attention planner state, or sampling/output state. That is what makes
# them safe targets for torch.compile: Dynamo/Inductor can see a pure tensor
# subgraph with stable semantics instead of the full serving loop.
#
# The teaching point is important:
#   - torch.compile works well on pure model-body tensor regions.
#   - paged attention and KV-cache mutation are kept as explicit boundaries.
#   - full decode graph replay is a separate, stricter optimization.


def _mlp_tensor_forward(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU MLP used by the default, safest compile scope."""
    gate, up = F.linear(x, gate_up_weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_weight)


def _gemma4_mlp_tensor_forward(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Gemma 4 GELU-tanh MLP used by the default compile scope."""
    gate, up = F.linear(x, gate_up_weight).chunk(2, dim=-1)
    return F.linear(F.gelu(gate, approximate="tanh") * up, down_weight)


def _vllm_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor | None:
    """Use vLLM's native GELU-tanh-and-mul op when the extension is present."""
    if x.device.type != "cuda" or x.shape[-1] % 2 != 0:
        return None
    op = getattr(torch.ops._C, "gelu_tanh_and_mul", None)
    if op is None:
        return None
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    try:
        op(out, x)
    except Exception:
        return None
    return out


def _rms_norm_tensor_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Small RMSNorm expression for experimental compile scopes."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _input_norm_qkv_no_residual_tensor_forward(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor input RMSNorm + QKV projection for the first layer."""
    normed = _rms_norm_tensor_forward(hidden, norm_weight, eps)
    qkv = F.linear(normed.reshape(-1, hidden.shape[-1]), qkv_weight)
    # Layer 0 has no accumulated residual yet. Returning `hidden` initializes
    # the residual stream without accidentally adding embeddings to themselves.
    return qkv.view(*hidden.shape[:-1], -1), hidden


def _input_norm_qkv_residual_tensor_forward(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor residual add + input RMSNorm + QKV projection."""
    # `hidden` is the previous layer's MLP branch output. Add it here, directly
    # next to RMSNorm, so torch.compile can keep the residual add and norm in
    # one tensor region.
    residual = hidden + residual
    normed = _rms_norm_tensor_forward(residual, norm_weight, eps)
    qkv = F.linear(normed.reshape(-1, hidden.shape[-1]), qkv_weight)
    return qkv.view(*hidden.shape[:-1], -1), residual


def _attention_tail_tensor_forward(
    attn_out_heads: torch.Tensor,
    residual: torch.Tensor,
    o_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor O projection + post-attention RMSNorm + MLP."""
    attn_out = F.linear(
        attn_out_heads.reshape(-1, o_weight.shape[1]),
        o_weight,
    ).view(*attn_out_heads.shape[:-2], -1)
    # Attention's output projection is the second residual branch in a Llama
    # block. Keep the add adjacent to post-attention RMSNorm for fusion.
    residual = attn_out + residual
    normed = _rms_norm_tensor_forward(residual, post_norm_weight, eps)
    hidden = _mlp_tensor_forward(
        normed.reshape(-1, normed.shape[-1]),
        gate_up_weight,
        down_weight,
    ).view_as(normed)
    return hidden, residual


def _attention_tail_masked_tensor_forward(
    attn_out_heads: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
    o_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor tail for padded prefill chunks."""
    attn_out = F.linear(
        attn_out_heads.reshape(-1, o_weight.shape[1]),
        o_weight,
    ).view(*attn_out_heads.shape[:-2], -1)
    attn_out = attn_out * mask.unsqueeze(-1)
    # Same attention residual add as the unmasked tail; the mask only prevents
    # padded prefill cells from contributing to the carried residual stream.
    residual = attn_out + residual
    normed = _rms_norm_tensor_forward(residual, post_norm_weight, eps)
    hidden = _mlp_tensor_forward(
        normed.reshape(-1, normed.shape[-1]),
        gate_up_weight,
        down_weight,
    ).view_as(normed)
    return hidden * mask.unsqueeze(-1), residual


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        """Construct RMSNorm over the final hidden dimension."""
        super().__init__()
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


class RMSNormNoWeight(nn.Module):
    """RMSNorm variant used by Gemma 4 value normalization.

    Gemma 4 applies learned RMSNorm weights to Q and K, but value normalization
    is pure normalization with no learned scale. Keeping this as a separate
    class makes that model choice visible in the code.
    """

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        mean_squared = x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return (x_float * torch.pow(mean_squared, -0.5)).type_as(x)


def _gemma4_fast_rmsnorm_enabled(hidden_size: int, role: str) -> bool:
    fast_mode = os.environ.get("SIMPLE_VLLM_GEMMA_FAST_RMSNORM", "")
    if fast_mode in {"1", "all"}:
        return True
    if fast_mode == "hidden":
        return hidden_size > 1024
    if fast_mode == "head":
        return hidden_size <= 1024
    modes = {
        item.strip()
        for item in fast_mode.replace("+", ",").replace(" ", ",").split(",")
        if item.strip()
    }
    return (
        role in modes
        or ("hidden" in modes and hidden_size > 1024)
        or ("head" in modes and hidden_size <= 1024)
        or ("ff" in modes and role in {"pre_ff", "post_ff"})
        or ("attn" in modes and role in {"input", "post_attn", "q", "k", "v"})
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class Gemma4RMSNorm(nn.Module):
    """Gemma 4 RMSNorm matching the Hugging Face implementation."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        with_scale: bool = True,
        role: str = "hidden",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.with_scale = with_scale
        self.fast_math = _gemma4_fast_rmsnorm_enabled(hidden_size, role)
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer(
                "unit_weight",
                torch.ones(hidden_size),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast_math:
            if self.with_scale:
                custom = triton_rms_norm(x, self.weight, self.eps)
                if custom is not None:
                    return custom
            else:
                backend = os.environ.get(
                    "SIMPLE_VLLM_GEMMA_UNSCALED_RMSNORM",
                    "triton",
                )
                if backend in {"unit", "unit_weight", "weighted"}:
                    custom = triton_rms_norm(x, self.unit_weight, self.eps)
                    if custom is not None:
                        return custom
                elif backend in {"triton", "no_weight", "triton_no_weight"}:
                    custom = triton_rms_norm_no_weight(x, self.eps)
                    if custom is not None:
                        return custom
            x_float = x.float()
            mean_squared = x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
            out = x_float * torch.rsqrt(mean_squared)
            if self.with_scale:
                out = out * self.weight.float()
            return out.type_as(x)
        x_float = x.float()
        mean_squared = x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        out = x_float * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            out = out * self.weight.float()
        return out.type_as(x)


class Gemma4Layer(nn.Module):
    """One Gemma 4 text decoder layer.

    Gemma 4 is not just "Llama with different numbers":

    - Sliding layers and full layers can use different head dimensions.
    - Full layers in this checkpoint have no `v_proj`; the K projection output
      is reused as the raw V input, then K and V receive different norms.
    - Q, K, and V are normalized in head space before attention.
    - The MLP uses GELU-tanh gating instead of Llama's SiLU gate.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.kv_heads_for_layer(layer_idx)
        self.attention_hidden_size = self.num_attention_heads * self.head_dim
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim
        self.uses_k_projection_as_raw_v = (
            self.layer_type == "full_attention" and config.attention_k_eq_v
        )
        self.use_packed_qkv_projection = (
            _env_flag("SIMPLE_VLLM_GEMMA_FUSED_QKV_PROJ")
            or _env_flag("SIMPLE_VLLM_GEMMA_OPTIMIZE")
        )

        self.input_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, role="input"
        )
        self.post_attention_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, role="post_attn"
        )
        self.pre_feedforward_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, role="pre_ff"
        )
        self.post_feedforward_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, role="post_ff"
        )

        if self.use_packed_qkv_projection:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            if self.uses_k_projection_as_raw_v:
                self.qk_proj = nn.Linear(
                    config.hidden_size,
                    self.attention_hidden_size + self.kv_hidden_size,
                    bias=False,
                )
                self.qkv_proj = None
            else:
                self.qk_proj = None
                self.qkv_proj = nn.Linear(
                    config.hidden_size,
                    self.attention_hidden_size + 2 * self.kv_hidden_size,
                    bias=False,
                )
        else:
            self.qk_proj = None
            self.qkv_proj = None
            self.q_proj = nn.Linear(config.hidden_size, self.attention_hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, self.kv_hidden_size, bias=False)
            self.v_proj = (
                None
                if self.uses_k_projection_as_raw_v
                else nn.Linear(config.hidden_size, self.kv_hidden_size, bias=False)
            )
        self.o_proj = nn.Linear(self.attention_hidden_size, config.hidden_size, bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, role="q")
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, role="k")
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, config.rms_norm_eps, with_scale=False, role="v"
        )
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Gemma 4 checkpoints include a learned scalar per decoder layer. Some
        # layers use very small values, so omitting this is not a minor numeric
        # difference; activations become much too large.
        self.register_buffer("layer_scalar", torch.ones(1))


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
        for layer_idx in range(config.num_layers):
            if config.architecture == "gemma4":
                self.layers.append(Gemma4Layer(config, layer_idx))
            else:
                self.layers.append(MiniLlamaLayer(config))
        if config.architecture == "gemma4":
            self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps, role="final")
        else:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Gemma 4 text checkpoints tie the LM head to the token embedding
            # table. Assigning the same Parameter keeps the model structure
            # explicit and avoids loading a nonexistent lm_head tensor.
            self.lm_head.weight = self.embed_tokens.weight
        if config.embedding_scale is not None:
            self.register_buffer(
                "embedding_scale",
                torch.tensor(config.embedding_scale, dtype=torch.get_default_dtype()),
                persistent=False,
            )
        else:
            self.embedding_scale = None
        # Precompute vLLM-compatible cos/sin tables once. Llama/Mistral need one
        # table. Gemma 4 needs one table per layer type because sliding and full
        # layers use different RoPE parameters and head dimensions.
        if config.architecture == "gemma4":
            for layer_type in sorted(set(config.layer_types or ())):
                layer_idx = (config.layer_types or ()).index(layer_type)
                rope_cache = build_rope_cos_sin_cache(
                    head_size=config.head_dim_for_layer(layer_idx),
                    max_position=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    rope_scaling=config.rope_scaling_for_layer(layer_idx),
                    dtype=torch.get_default_dtype(),
                )
                self.register_buffer(
                    f"rope_cos_sin_cache_{layer_type}",
                    rope_cache,
                    persistent=False,
                )
        else:
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
        # These fields are populated only when --torch-compile-model-body is
        # enabled. The default scope is `mlp`; broader scopes remain available
        # for experiments but are not universally faster across model families.
        self._compiled_mlp = None
        self._compiled_gemma4_mlp = None
        self._compiled_input_norm_qkv_no_residual = None
        self._compiled_input_norm_qkv_residual = None
        self._compiled_attention_tail = None
        self._compiled_attention_tail_masked = None
        self._enable_gemma4_compiled_mlp = (
            os.environ.get("SIMPLE_VLLM_GEMMA_COMPILE_MLP") == "1"
        )
        self._gemma_fast_rope = os.environ.get("SIMPLE_VLLM_GEMMA_FAST_ROPE") == "1"
        self._gemma_fused_qk_norm_rope = (
            _env_flag("SIMPLE_VLLM_GEMMA_FUSED_QK_NORM_ROPE")
            or _env_flag("SIMPLE_VLLM_GEMMA_OPTIMIZE")
        )
        self._gemma_fused_qkv_norm = (
            _env_flag("SIMPLE_VLLM_GEMMA_FUSED_QKV_NORM")
            or _env_flag("SIMPLE_VLLM_GEMMA_OPTIMIZE")
        )
        self._gemma_gelu_gate_triton = (
            _env_flag("SIMPLE_VLLM_GEMMA_GELU_GATE_TRITON")
            or _env_flag("SIMPLE_VLLM_GEMMA_OPTIMIZE")
        )
        self._gemma_vllm_gelu_gate = _env_flag("SIMPLE_VLLM_GEMMA_VLLM_GELU_GATE")
        self._gemma_fused_post_norm = (
            _env_flag("SIMPLE_VLLM_GEMMA_FUSED_POST_NORM")
            or _env_flag("SIMPLE_VLLM_GEMMA_OPTIMIZE")
        )
        # Debug guard for RoPE kernel ordering. Normal execution keeps RoPE and
        # decode workspace updates on the same CUDA stream, so stream ordering is
        # sufficient and this should stay disabled for benchmarking.
        self._sync_after_rope = os.environ.get("SIMPLE_VLLM_SYNC_AFTER_ROPE") == "1"
        self._sync_after_forward = os.environ.get("SIMPLE_VLLM_SYNC_AFTER_FORWARD", "0") != "0"

    def enable_torch_compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = True,
        scope: str = "mlp",
    ) -> None:
        """Compile selected pure tensor model-body blocks.

        The full metadata forward includes Python scheduler objects, KV cache
        mutation, and external paged-attention calls. Compiling the whole
        method causes graph breaks and recompiles, so these scopes only cover
        deterministic tensor work around attention.
        """
        # Scope meanings:
        #   mlp       - safest/default: only the SwiGLU MLP subgraph.
        #   input_qkv - input norm + QKV projection, with RoPE still outside.
        #   tail      - O projection + post-attention norm + MLP.
        #   all       - input_qkv + tail.
        #
        # None of these scopes include paged attention or KV-cache writes.
        valid_scopes = {"mlp", "input_qkv", "tail", "all"}
        if scope not in valid_scopes:
            raise ValueError(f"torch.compile scope must be one of {sorted(valid_scopes)}")
        compile_kwargs = {
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": "inductor",
        }
        if scope in {"mlp", "all"}:
            self._compiled_mlp = torch.compile(_mlp_tensor_forward, **compile_kwargs)
            if self.config.architecture == "gemma4" and self._enable_gemma4_compiled_mlp:
                self._compiled_gemma4_mlp = torch.compile(
                    _gemma4_mlp_tensor_forward,
                    **compile_kwargs,
                )
        if scope in {"input_qkv", "all"}:
            self._compiled_input_norm_qkv_no_residual = torch.compile(
                _input_norm_qkv_no_residual_tensor_forward,
                **compile_kwargs,
            )
            self._compiled_input_norm_qkv_residual = torch.compile(
                _input_norm_qkv_residual_tensor_forward,
                **compile_kwargs,
            )
        if scope in {"tail", "all"}:
            self._compiled_attention_tail = torch.compile(
                _attention_tail_tensor_forward,
                **compile_kwargs,
            )
            self._compiled_attention_tail_masked = torch.compile(
                _attention_tail_masked_tensor_forward,
                **compile_kwargs,
            )

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
        return self._apply_cached_rope_to_flat_qk(
            q_flat,
            k_flat,
            positions,
            q_num_heads=self.config.num_attention_heads,
            k_num_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            cos_sin_cache=self.rope_cos_sin_cache,
        )

    def _apply_cached_rope_to_flat_qk(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        positions: torch.Tensor,
        *,
        q_num_heads: int,
        k_num_heads: int,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Apply a specific cached RoPE table to flat Q/K tensors."""
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
            fused = triton_apply_qk_rope_from_cache(
                q_2d,
                k_2d,
                flat_positions,
                cos_sin_cache,
                q_num_heads=q_num_heads,
                k_num_heads=k_num_heads,
                head_dim=head_dim,
            )
            if fused is not None:
                q_triton, k_triton = fused
            else:
                q_triton = triton_apply_rope_from_cache(
                    q_2d,
                    flat_positions,
                    cos_sin_cache,
                    num_heads=q_num_heads,
                    head_dim=head_dim,
                )
                k_triton = triton_apply_rope_from_cache(
                    k_2d,
                    flat_positions,
                    cos_sin_cache,
                    num_heads=k_num_heads,
                    head_dim=head_dim,
                )
        except Exception:
            q_triton = None
            k_triton = None
        if q_triton is not None and k_triton is not None:
            if self._sync_after_rope:
                torch.cuda.current_stream(q_triton.device).synchronize()
            return q_triton.view(q_shape), k_triton.view(k_shape)
        return None

    def _apply_rope_from_cache_readable(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Readable CPU/fallback RoPE using the same cache as the Triton path."""
        cache = cos_sin_cache
        rotary_dim = cache.shape[-1]
        half_dim = rotary_dim // 2
        flat_positions = positions.reshape(-1)
        x_shape = x.shape
        x_flat = x.reshape(flat_positions.numel(), -1, x.shape[-1])
        cos = cache.index_select(0, flat_positions)[:, :half_dim].unsqueeze(1)
        sin = cache.index_select(0, flat_positions)[:, half_dim:].unsqueeze(1)
        x_rot = x_flat[..., :rotary_dim]
        x_pass = x_flat[..., rotary_dim:]
        x0 = x_rot[..., :half_dim]
        x1 = x_rot[..., half_dim:]
        rotated = torch.cat((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)
        if x_pass.numel() > 0:
            rotated = torch.cat((rotated, x_pass), dim=-1)
        return rotated.view(x_shape)

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

    def _gemma_rope_cache(self, layer: Gemma4Layer) -> torch.Tensor:
        return getattr(self, f"rope_cos_sin_cache_{layer.layer_type}")

    def _project_gemma4_raw_qkv(
        self,
        layer: Gemma4Layer,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Gemma 4 raw Q/K/V using separate or packed weights."""
        if layer.qkv_proj is not None:
            projected = F.linear(hidden, layer.qkv_proj.weight)
            q_flat, k_flat_raw, v_flat_raw = projected.split(
                [layer.attention_hidden_size, layer.kv_hidden_size, layer.kv_hidden_size],
                dim=-1,
            )
            return q_flat, k_flat_raw, v_flat_raw
        if layer.qk_proj is not None:
            projected = F.linear(hidden, layer.qk_proj.weight)
            q_flat, k_flat_raw = projected.split(
                [layer.attention_hidden_size, layer.kv_hidden_size],
                dim=-1,
            )
            return q_flat, k_flat_raw, k_flat_raw

        if layer.q_proj is None or layer.k_proj is None:
            raise RuntimeError("Gemma4 layer has neither packed nor separate Q/K projections")
        q_flat = F.linear(hidden, layer.q_proj.weight)
        k_flat_raw = F.linear(hidden, layer.k_proj.weight)
        if layer.v_proj is None:
            return q_flat, k_flat_raw, k_flat_raw
        return q_flat, k_flat_raw, F.linear(hidden, layer.v_proj.weight)

    def _finish_gemma4_qkv(
        self,
        layer: Gemma4Layer,
        q_flat: torch.Tensor,
        k_flat_raw: torch.Tensor,
        v_flat_raw: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize Gemma 4 Q/K/V and apply RoPE to Q/K."""
        prefix_shape = q_flat.shape[:-1]
        cache = self._gemma_rope_cache(layer)
        fused_rope = self._gemma_fused_qk_norm_rope
        fused_norm = fused_rope or self._gemma_fused_qkv_norm
        if fused_norm:
            flat_positions = positions.reshape(-1).contiguous()
            fused = triton_gemma_qkv_norm_rope(
                q_flat,
                k_flat_raw,
                v_flat_raw,
                layer.q_norm.weight,
                layer.k_norm.weight,
                layer.q_norm.eps,
                flat_positions,
                cache,
                num_q_heads=layer.num_attention_heads,
                num_kv_heads=layer.num_key_value_heads,
                head_dim=layer.head_dim,
                apply_rope=fused_rope,
                kv_shared=v_flat_raw is k_flat_raw,
            )
            if fused is not None:
                q_flat, k_flat, v_flat = fused
                q = q_flat.view(*prefix_shape, layer.num_attention_heads, layer.head_dim)
                k = k_flat.view(*prefix_shape, layer.num_key_value_heads, layer.head_dim)
                v = v_flat.view(*prefix_shape, layer.num_key_value_heads, layer.head_dim)
                if fused_rope:
                    if self._sync_after_rope:
                        torch.cuda.current_stream(q.device).synchronize()
                    return q, k, v
                q_normed = q
                k_normed = k
                v = v
            else:
                q_normed = None
                k_normed = None
                v = None
        else:
            q_normed = None
            k_normed = None
            v = None

        if q_normed is None or k_normed is None or v is None:
            q = q_flat.view(*prefix_shape, layer.num_attention_heads, layer.head_dim)
            k_raw = k_flat_raw.view(*prefix_shape, layer.num_key_value_heads, layer.head_dim)
            v_raw = v_flat_raw.view(*prefix_shape, layer.num_key_value_heads, layer.head_dim)

            q_normed = layer.q_norm(q)
            k_normed = layer.k_norm(k_raw)
            v = layer.v_norm(v_raw)

        if self._gemma_fast_rope:
            rotated = self._apply_cached_rope_to_flat_qk(
                q_normed.reshape(*prefix_shape, -1),
                k_normed.reshape(*prefix_shape, -1),
                positions,
                q_num_heads=layer.num_attention_heads,
                k_num_heads=layer.num_key_value_heads,
                head_dim=layer.head_dim,
                cos_sin_cache=cache,
            )
        else:
            rotated = None
        if rotated is not None:
            q_flat, k_flat = rotated
            q = q_flat.view(*prefix_shape, layer.num_attention_heads, layer.head_dim)
            k = k_flat.view(*prefix_shape, layer.num_key_value_heads, layer.head_dim)
        else:
            # Gemma 4 greedy outputs are sensitive to RoPE rounding. HF applies
            # rotary embedding through regular PyTorch bf16 operations; the
            # Triton cache kernel is opt-in because it can change argmaxes.
            q = self._apply_rope_from_cache_readable(q_normed, positions, cache)
            k = self._apply_rope_from_cache_readable(k_normed, positions, cache)
        return q, k, v

    def _project_gemma4_qkv(
        self,
        layer: Gemma4Layer,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Gemma 4 Q/K/V for a padded prefill-style tensor.

        The no-`v_proj` full-attention case is intentionally explicit:
        V starts from the raw K projection, then K and V receive different
        normalizations. K also receives RoPE; V does not.
        """
        q_flat, k_flat_raw, v_flat_raw = self._project_gemma4_raw_qkv(layer, hidden)
        return self._finish_gemma4_qkv(layer, q_flat, k_flat_raw, v_flat_raw, positions)

    def _project_gemma4_qkv_decode(
        self,
        layer: Gemma4Layer,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Gemma 4 Q/K/V for one decode token per request."""
        q_flat, k_flat_raw, v_flat_raw = self._project_gemma4_raw_qkv(layer, hidden)
        return self._finish_gemma4_qkv(
            layer,
            q_flat,
            k_flat_raw,
            v_flat_raw,
            positions.reshape(-1),
        )

    def _gemma4_mlp(self, layer: Gemma4Layer, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self.config.hidden_size)
        if self._compiled_gemma4_mlp is not None:
            return self._compiled_gemma4_mlp(
                x_flat,
                layer.gate_up_proj.weight,
                layer.down_proj.weight,
            ).view_as(x)
        gate_up = F.linear(x_flat, layer.gate_up_proj.weight)
        activated = None
        if self._gemma_vllm_gelu_gate:
            activated = _vllm_gelu_tanh_and_mul(gate_up)
        if activated is None and self._gemma_gelu_gate_triton:
            activated = triton_gelu_tanh_and_mul(gate_up)
        if activated is None:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.gelu(gate, approximate="tanh") * up
        return F.linear(activated, layer.down_proj.weight).view_as(x)

    def _gemma4_post_norm_residual(
        self,
        norm: Gemma4RMSNorm,
        x: torch.Tensor,
        residual: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._gemma_fused_post_norm:
            # Gemma's residual order is norm(branch) + residual. Keep that pair
            # together so the optional Triton kernel can fuse post-norm,
            # residual add, and layer scale when present.
            custom = triton_gemma_post_norm_residual(
                x,
                residual,
                norm.weight,
                norm.eps,
                scale,
            )
            if custom is not None:
                return custom
        # Fallback keeps the same Gemma ordering without the fused Triton kernel.
        out = norm(x) + residual
        if scale is not None:
            out = out * scale
        return out

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
        if self.config.architecture == "gemma4":
            return self._forward_gemma4_with_metadata_impl(
                input_ids=input_ids,
                positions=positions,
                metadata=metadata,
                kv_cache=kv_cache,
                block_id_lists=block_id_lists,
            )
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
            #
            # This is pure tensor work except for RoPE's custom kernel lifetime
            # handling. The optional `input_qkv` compile scope compiles norm and
            # QKV projection but intentionally leaves RoPE outside the compiled
            # function.
            with self.profiler.section("model.qkv_proj") if self.profiler else nullcontext():
                if self._compiled_input_norm_qkv_no_residual is not None:
                    if residual is None:
                        # First layer: initialize residual from embeddings without
                        # adding embeddings to themselves.
                        qkv, residual = self._compiled_input_norm_qkv_no_residual(
                            hidden,
                            layer.input_norm.weight,
                            layer.qkv_proj.weight,
                            layer.input_norm.eps,
                        )
                    else:
                        # Later layers: fold the previous MLP output (`hidden`)
                        # into the residual stream at the input RMSNorm boundary.
                        qkv, residual = self._compiled_input_norm_qkv_residual(
                            hidden,
                            residual,
                            layer.input_norm.weight,
                            layer.qkv_proj.weight,
                            layer.input_norm.eps,
                        )
                    q, k_new, v_new = self._split_projected_qkv(qkv, positions)
                else:
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
                    # `positions` holds absolute token indices, so RoPE still
                    # sees the right offsets during later prefill chunks.
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

            if self._compiled_attention_tail is not None:
                # 3a. Experimental compiled tail.
                #
                # This fuses the post-attention pure tensor region into one
                # compile target: O projection, residual add/RMSNorm, and MLP.
                # It helped Llama slightly but was not universally faster.
                with self.profiler.section("model.attn_tail") if self.profiler else nullcontext():
                    if mask is None:
                        hidden, residual = self._compiled_attention_tail(
                            attn_out_heads,
                            residual,
                            layer.o_proj.weight,
                            layer.post_norm.weight,
                            layer.gate_up_proj.weight,
                            layer.down_proj.weight,
                            layer.post_norm.eps,
                        )
                    else:
                        hidden, residual = self._compiled_attention_tail_masked(
                            attn_out_heads,
                            residual,
                            mask,
                            layer.o_proj.weight,
                            layer.post_norm.weight,
                            layer.gate_up_proj.weight,
                            layer.down_proj.weight,
                            layer.post_norm.eps,
                        )
            else:
                # 3b. Default tail.
                #
                # The default keeps O projection, post-attention RMSNorm, and
                # MLP visible as separate profiler sections. If MLP compile is
                # enabled, only the SwiGLU MLP subgraph is compiled.
                with self.profiler.section("model.attn_out_proj") if self.profiler else nullcontext():
                    attn_out = F.linear(
                        attn_out_heads.reshape(-1, self.config.attention_hidden_size),
                        layer.o_proj.weight,
                    ).view(attn_out_heads.shape[0], attn_out_heads.shape[1], -1)
                    hidden = attn_out if mask is None else self._masked_residual(attn_out, mask)
                with self.profiler.section("model.post_attn_norm") if self.profiler else nullcontext():
                    # Attention output is the other Llama residual branch. Add it
                    # at post-attention norm so add+RMSNorm stays fused.
                    normed, residual = layer.post_norm(hidden, residual)
                with self.profiler.section("model.mlp") if self.profiler else nullcontext():
                    if self._compiled_mlp is not None:
                        normed_shape = normed.shape
                        hidden = self._compiled_mlp(
                            normed.reshape(-1, self.config.hidden_size),
                            layer.gate_up_proj.weight,
                            layer.down_proj.weight,
                        ).view(normed_shape)
                    else:
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

    def _forward_gemma4_with_metadata_impl(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
        block_id_lists: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Run Gemma 4 chunked prefill or generic batched decode.

        This mirrors the HF/vLLM Gemma 4 residual pattern directly:

            residual = x
            x = input_norm(x)
            x = attention(x)
            x = post_attention_norm(x) + residual

            residual = x
            x = pre_feedforward_norm(x)
            x = mlp(x)
            x = post_feedforward_norm(x) + residual

        That differs from the Llama path above, which keeps a separate residual
        accumulator and uses fused add+RMSNorm.
        """
        with self.profiler.section("model.embed") if self.profiler else nullcontext():
            hidden = self.embed_tokens(input_ids)
            if self.embedding_scale is not None:
                hidden = hidden * self.embedding_scale

        mask = None
        if not metadata.query_all_valid:
            mask = (
                torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                < metadata.query_lens.unsqueeze(1)
            ).to(hidden.dtype)
            hidden = self._masked_residual(hidden, mask)

        block_id_lists = block_id_lists or []
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, Gemma4Layer)
            residual = hidden
            with self.profiler.section("model.qkv_proj") if self.profiler else nullcontext():
                normed = layer.input_norm(hidden)
                q, k_new, v_new = self._project_gemma4_qkv(layer, normed, positions)

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
                softmax_scale=1.0,
                sliding_window=self.config.sliding_window_for_layer(layer_idx),
                logits_soft_cap=None,
            )
            with self.profiler.section("model.attn_out_proj") if self.profiler else nullcontext():
                attn_out = F.linear(
                    attn_out_heads.reshape(-1, layer.attention_hidden_size),
                    layer.o_proj.weight,
                ).view(attn_out_heads.shape[0], attn_out_heads.shape[1], -1)
                if mask is not None:
                    attn_out = self._masked_residual(attn_out, mask)
            with self.profiler.section("model.post_attn_norm") if self.profiler else nullcontext():
                hidden = self._gemma4_post_norm_residual(
                    layer.post_attention_norm,
                    attn_out,
                    residual,
                )
            with self.profiler.section("model.mlp") if self.profiler else nullcontext():
                residual = hidden
                normed = layer.pre_feedforward_norm(hidden)
                hidden = self._gemma4_mlp(layer, normed)
                if mask is not None:
                    hidden = self._masked_residual(hidden, mask)
                hidden = self._gemma4_post_norm_residual(
                    layer.post_feedforward_norm,
                    hidden,
                    residual,
                    scale=layer.layer_scalar,
                )

        with self.profiler.section("model.final_norm_lm_head") if self.profiler else nullcontext():
            hidden = self.norm(hidden)
            if metadata.query_all_valid:
                last_hidden = hidden[:, -1, :]
            else:
                last_hidden = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device),
                    metadata.query_lens - 1,
                ]
            logits = self.lm_head(last_hidden)
            if self.config.final_logit_softcapping is not None:
                cap = float(self.config.final_logit_softcapping)
                logits = torch.tanh(logits / cap) * cap
            if self._sync_after_forward and logits.is_cuda:
                torch.cuda.current_stream(logits.device).synchronize()
            return logits

    def _forward_decode_with_metadata_impl(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
        block_id_lists: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Run the optimized decode path.

        Decode has a simpler shape than prefill: every active request supplies
        exactly one token. That lets us keep most activations as
        `[active_requests, hidden]` instead of `[batch, 1, hidden]`, avoiding
        some reshape/mask work in the hottest loop.
        """
        if self.config.architecture == "gemma4":
            return self._forward_gemma4_decode_with_metadata_impl(
                input_ids=input_ids,
                positions=positions,
                metadata=metadata,
                kv_cache=kv_cache,
                block_id_lists=block_id_lists,
            )
        with self.profiler.section("model.embed") if self.profiler else nullcontext():
            hidden = self.embed_tokens(input_ids.reshape(-1))

        block_id_lists = block_id_lists or []
        # Same deferred-residual scheme as prefill, but with token-major decode
        # tensors. Adds are still delayed to RMSNorm boundaries for fusion.
        residual = None
        batch_size = hidden.shape[0]
        for layer_idx, layer in enumerate(self.layers):
            # Same layer structure as prefill, but with token-major shapes.
            with self.profiler.section("model.qkv_proj") if self.profiler else nullcontext():
                if self._compiled_input_norm_qkv_no_residual is not None:
                    if residual is None:
                        # First layer: initialize residual from embeddings without
                        # adding embeddings to themselves.
                        qkv, residual = self._compiled_input_norm_qkv_no_residual(
                            hidden,
                            layer.input_norm.weight,
                            layer.qkv_proj.weight,
                            layer.input_norm.eps,
                        )
                    else:
                        # Later layers: fold the previous MLP output (`hidden`)
                        # into the residual stream at the input RMSNorm boundary.
                        qkv, residual = self._compiled_input_norm_qkv_residual(
                            hidden,
                            residual,
                            layer.input_norm.weight,
                            layer.qkv_proj.weight,
                            layer.input_norm.eps,
                        )
                    q, k_new, v_new = self._split_projected_qkv_decode(qkv, positions)
                else:
                    if residual is None:
                        # First layer: initialize the residual stream from embeddings.
                        # There is no previous MLP branch to add yet.
                        residual = hidden
                        normed = layer.input_norm(hidden)
                    else:
                        # Later layers: add the deferred MLP output and normalize
                        # in one fused add+RMSNorm call.
                        normed, residual = layer.input_norm(hidden, residual)
                    q, k_new, v_new = self._project_qkv_decode(layer, normed, positions)

            # The decode attention backend is expected to write the new K/V
            # directly into the paged cache using `decode_slot_mapping`.
            attn_out_heads = self.attention_backend.forward(
                layer_idx=layer_idx,
                q=q.view(batch_size, 1, self.config.num_attention_heads, self.config.head_dim),
                k_new=k_new.view(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim),
                v_new=v_new.view(batch_size, 1, self.config.num_key_value_heads, self.config.head_dim),
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
                prefill_slot_mapping=None,
                prefill_slot_mapping_all_valid=False,
            )
            if self._compiled_attention_tail is not None:
                # Experimental compiled post-attention tail. The helper expects
                # an attention-head-shaped tensor so we restore that view here.
                with self.profiler.section("model.attn_tail") if self.profiler else nullcontext():
                    hidden, residual = self._compiled_attention_tail(
                        attn_out_heads.reshape(
                            batch_size,
                            self.config.num_attention_heads,
                            self.config.head_dim,
                        ),
                        residual,
                        layer.o_proj.weight,
                        layer.post_norm.weight,
                        layer.gate_up_proj.weight,
                        layer.down_proj.weight,
                        layer.post_norm.eps,
                    )
            else:
                with self.profiler.section("model.attn_out_proj") if self.profiler else nullcontext():
                    hidden = F.linear(
                        attn_out_heads.reshape(batch_size, self.config.attention_hidden_size),
                        layer.o_proj.weight,
                    )
                with self.profiler.section("model.post_attn_norm") if self.profiler else nullcontext():
                    # Add the attention branch at post-attention RMSNorm so the
                    # residual add can stay fused with the norm.
                    normed, residual = layer.post_norm(hidden, residual)
                with self.profiler.section("model.mlp") if self.profiler else nullcontext():
                    if self._compiled_mlp is not None:
                        hidden = self._compiled_mlp(
                            normed,
                            layer.gate_up_proj.weight,
                            layer.down_proj.weight,
                        )
                    else:
                        gate_up = F.linear(normed, layer.gate_up_proj.weight)
                        activated = triton_silu_and_mul(gate_up)
                        if activated is None:
                            gate, up = gate_up.chunk(2, dim=-1)
                            activated = swiglu(gate, up)
                        hidden = F.linear(activated, layer.down_proj.weight)

        with self.profiler.section("model.final_norm_lm_head") if self.profiler else nullcontext():
            if residual is not None:
                # Flush the final layer's deferred MLP output into the residual
                # stream before the final RMSNorm.
                hidden, _ = self.norm(hidden, residual)
            else:
                hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
            if self._sync_after_forward and logits.is_cuda:
                torch.cuda.current_stream(logits.device).synchronize()
            return logits

    def _forward_gemma4_decode_with_metadata_impl(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
        block_id_lists: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Run Gemma 4's optimized one-token decode path."""
        with self.profiler.section("model.embed") if self.profiler else nullcontext():
            hidden = self.embed_tokens(input_ids.reshape(-1))
            if self.embedding_scale is not None:
                hidden = hidden * self.embedding_scale

        block_id_lists = block_id_lists or []
        batch_size = hidden.shape[0]
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, Gemma4Layer)
            residual = hidden
            with self.profiler.section("model.qkv_proj") if self.profiler else nullcontext():
                normed = layer.input_norm(hidden)
                q, k_new, v_new = self._project_gemma4_qkv_decode(layer, normed, positions)

            attn_out_heads = self.attention_backend.forward(
                layer_idx=layer_idx,
                q=q.view(batch_size, 1, layer.num_attention_heads, layer.head_dim),
                k_new=k_new.view(batch_size, 1, layer.num_key_value_heads, layer.head_dim),
                v_new=v_new.view(batch_size, 1, layer.num_key_value_heads, layer.head_dim),
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
                prefill_slot_mapping=None,
                prefill_slot_mapping_all_valid=False,
                softmax_scale=1.0,
                sliding_window=self.config.sliding_window_for_layer(layer_idx),
                logits_soft_cap=None,
            )
            with self.profiler.section("model.attn_out_proj") if self.profiler else nullcontext():
                hidden = F.linear(
                    attn_out_heads.reshape(batch_size, layer.attention_hidden_size),
                    layer.o_proj.weight,
                )
            with self.profiler.section("model.post_attn_norm") if self.profiler else nullcontext():
                hidden = self._gemma4_post_norm_residual(
                    layer.post_attention_norm,
                    hidden,
                    residual,
                )
            with self.profiler.section("model.mlp") if self.profiler else nullcontext():
                residual = hidden
                normed = layer.pre_feedforward_norm(hidden)
                hidden = self._gemma4_mlp(layer, normed)
                hidden = self._gemma4_post_norm_residual(
                    layer.post_feedforward_norm,
                    hidden,
                    residual,
                    scale=layer.layer_scalar,
                )

        with self.profiler.section("model.final_norm_lm_head") if self.profiler else nullcontext():
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
            if self.config.final_logit_softcapping is not None:
                cap = float(self.config.final_logit_softcapping)
                logits = torch.tanh(logits / cap) * cap
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

    def prefill_chunk_prebuilt(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run prefill using externally managed graph-safe metadata tensors."""
        return self._forward_with_metadata(
            input_ids=input_ids,
            positions=positions,
            metadata=metadata,
            kv_cache=kv_cache,
            block_id_lists=None,
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

    def decode_tokens_prebuilt(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionBatchMetadata,
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run one decode forward pass using externally managed metadata tensors."""
        if (
            input_ids.shape[1] == 1
            and metadata.query_all_valid
            and metadata.triton_decode_metadata is not None
        ):
            return self._forward_decode_with_metadata_impl(
                input_ids=input_ids,
                positions=positions,
                metadata=metadata,
                kv_cache=kv_cache,
                block_id_lists=None,
            )
        return self._forward_with_metadata(
            input_ids=input_ids,
            positions=positions,
            metadata=metadata,
            kv_cache=kv_cache,
            block_id_lists=None,
        )
