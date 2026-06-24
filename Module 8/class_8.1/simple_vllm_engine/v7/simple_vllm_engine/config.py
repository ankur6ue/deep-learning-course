from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Shape-only decoder architecture settings."""

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
        if self.attention_head_dim is not None:
            return self.attention_head_dim
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_hidden_size(self) -> int:
        return self.num_attention_heads * self.head_dim


@dataclass(frozen=True)
class EngineConfig:
    """Runtime knobs exposed by this teaching step."""

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
        valid_attention_backends = {'flash_attn_paged'}
        if self.attention_backend not in valid_attention_backends:
            raise ValueError(
                "attention_backend must be one of: "
                + ", ".join(sorted(valid_attention_backends))
            )
