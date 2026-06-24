from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters for a Llama/Mistral-style decoder.

    These values describe tensor shapes only. They do not control serving
    policy or memory allocation; those live in `EngineConfig`.
    """

    vocab_size: int
    hidden_size: int = 256
    intermediate_size: int = 768
    num_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    attention_head_dim: int | None = None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden size used by attention projections."""
        if self.attention_head_dim is not None:
            return self.attention_head_dim
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_hidden_size(self) -> int:
        """Return the width of concatenated query attention heads."""
        return self.num_attention_heads * self.head_dim


@dataclass(frozen=True)
class EngineConfig:
    """Serving/runtime knobs for the teaching engine.

    The most important controls are:

    - `block_size`: number of token slots in one KV-cache page.
    - `num_blocks`: total physical pages in the KV cache.
    - `max_batch_tokens`: per-step token budget shared by decode and prefill.
    - `max_prefill_chunk_tokens`: largest prompt chunk for one request.
    - `max_decode_batch_size`: max active decode requests per step.

    Example:

        max_batch_tokens=128 and max_decode_batch_size=8 means eight decode
        requests consume eight slots, leaving up to 120 token slots for prompt
        chunks in the same scheduler step.
    """

    # Version-specific defaults. `run_real.py` uses these same choices, so
    # running a folder demonstrates that folder's optimization without requiring
    # flags that belong to other versions.
    block_size: int = 16
    num_blocks: int = 2048
    max_batch_tokens: int = 128
    max_prefill_chunk_tokens: int = 64
    max_decode_batch_size: int = 16
    max_model_len: int | None = None
    enable_prefix_cache: bool = False
    attention_backend: str = "flash_attn_paged"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    enable_timing: bool = False
    enable_decode_cuda_graphs: bool = True
    unsafe_decode_cuda_graphs: bool = False
    enable_torch_compile_model_body: bool = True
    torch_compile_scope: str = "mlp"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = True
    prewarm_torch_compile: bool = True
    enable_gpu_decode_state: bool = True
    enable_triton_decode_metadata_kernel: bool = True
    enable_eager_decode_workspace: bool = True
    enable_async_output_processing: bool = True
    ignore_eos: bool = False
    eos_token_id: int = 2
    pad_token_id: int = 0

    def validate(self, model_config: ModelConfig) -> None:
        """Check that engine settings are consistent with the model settings.

        Args:
            model_config: The model architecture being served. For example, the
                engine block size can be arbitrary, but the number of attention
                heads must still be divisible by the number of KV heads.
        """
        if model_config.num_attention_heads % model_config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive when set")
        valid_compile_scopes = {"mlp", "input_qkv", "tail", "all"}
        if self.torch_compile_scope not in valid_compile_scopes:
            raise ValueError(
                f"torch_compile_scope must be one of {sorted(valid_compile_scopes)}"
            )
        valid_attention_backends = {"flash_attn_paged"}
        if self.attention_backend not in valid_attention_backends:
            raise ValueError(
                "attention_backend must be one of: "
                + ", ".join(sorted(valid_attention_backends))
            )
