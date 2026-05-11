from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int = 256
    intermediate_size: int = 768
    num_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        """Return the per-head hidden size used by attention projections."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 2048
    max_batch_tokens: int = 128
    max_prefill_chunk_tokens: int = 64
    max_decode_batch_size: int = 16
    enable_prefix_cache: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    enable_timing: bool = False
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
