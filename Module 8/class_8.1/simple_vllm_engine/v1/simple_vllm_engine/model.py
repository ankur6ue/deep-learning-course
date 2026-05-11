from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .kernels import apply_rope, repeat_kv, sdpa_attention, swiglu
from .kv_cache import PagedKVCache
from .requests import RequestState


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        """Construct RMSNorm over the final hidden dimension."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each token vector along its hidden dimension."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


@dataclass
class LayerWeights:
    input_norm: RMSNorm
    post_norm: RMSNorm
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear


class MiniLlamaLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        """Build a compact random-initialized Llama-style decoder."""
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "input_norm": RMSNorm(config.hidden_size, config.rms_norm_eps),
                        "post_norm": RMSNorm(config.hidden_size, config.rms_norm_eps),
                        "q_proj": nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False),
                        "k_proj": nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False),
                        "v_proj": nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False),
                        "o_proj": nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False),
                        "gate_proj": nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                        "up_proj": nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                        "down_proj": nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                    }
                )
            )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _project_qkv(
        self,
        layer: nn.ModuleDict,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states into Q/K/V and apply RoPE to Q and K.

        Args:
            layer: One transformer layer's weights.
            hidden: Hidden states shaped `[B, T, hidden_size]`.
            positions: Absolute token positions shaped `[B, T]`. For a later
                prompt chunk these are not `0..T-1`; they continue from the
                earlier prompt prefix.
        """
        bsz, seqlen, _ = hidden.shape
        q = layer["q_proj"](hidden).view(
            bsz, seqlen, self.config.num_attention_heads, self.config.head_dim
        )
        k = layer["k_proj"](hidden).view(
            bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim
        )
        v = layer["v_proj"](hidden).view(
            bsz, seqlen, self.config.num_key_value_heads, self.config.head_dim
        )
        q = apply_rope(q, positions, self.config.rope_theta)
        k = apply_rope(k, positions, self.config.rope_theta)
        return q, k, v

    def _masked_residual(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Zero out padded token rows before adding a residual branch."""
        return x * mask.unsqueeze(-1)

    def prefill_chunk(
        self,
        requests: list[RequestState],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lengths: list[int],
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Run one chunked-prefill forward pass.

        Args:
            requests: Requests represented by the batch rows.
            input_ids: Prompt chunk tokens padded to a common width
                `[B, max_chunk]`.
            positions: Absolute positions for those chunk tokens. If one request
                already computed 5 prompt tokens and now processes 4 more, this
                row would be `[5, 6, 7, 8]`.
            lengths: Valid chunk length per request row.
            kv_cache: Shared paged KV cache to read the existing prefix from and
                append the new chunk to.
        """
        hidden = self.embed_tokens(input_ids)
        mask = (torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0) < torch.tensor(lengths, device=input_ids.device).unsqueeze(1)).to(hidden.dtype)
        hidden = self._masked_residual(hidden, mask)

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden
            normed = layer["input_norm"](hidden)
            q, k_new, v_new = self._project_qkv(layer, normed, positions)
            attn_out_heads = torch.zeros_like(q)

            for req_idx, req in enumerate(requests):
                chunk_len = lengths[req_idx]
                if chunk_len == 0:
                    continue
                past_len = req.prompt_tokens_computed
                k_past, v_past = kv_cache.gather_tokens(layer_idx, req.block_ids, req.cached_seq_len)
                q_i = q[req_idx, :chunk_len]
                k_new_i = k_new[req_idx, :chunk_len]
                v_new_i = v_new[req_idx, :chunk_len]
                k_full = torch.cat([repeat_kv(k_past, self.config.num_attention_heads), repeat_kv(k_new_i, self.config.num_attention_heads)], dim=0)
                v_full = torch.cat([repeat_kv(v_past, self.config.num_attention_heads), repeat_kv(v_new_i, self.config.num_attention_heads)], dim=0)
                attn_out_heads[req_idx, :chunk_len] = sdpa_attention(q_i, k_full, v_full, past_len)
                kv_cache.write_tokens(layer_idx, req.block_ids, req.cached_seq_len, k_new_i, v_new_i)

            attn_out = layer["o_proj"](attn_out_heads.reshape(attn_out_heads.shape[0], attn_out_heads.shape[1], -1))
            hidden = residual + self._masked_residual(attn_out, mask)
            residual = hidden
            normed = layer["post_norm"](hidden)
            ff = layer["down_proj"](swiglu(layer["gate_proj"](normed), layer["up_proj"](normed)))
            hidden = residual + self._masked_residual(ff, mask)

        hidden = self.norm(hidden)
        last_hidden = []
        for req_idx, chunk_len in enumerate(lengths):
            last_hidden.append(hidden[req_idx, chunk_len - 1])
        last_hidden_tensor = torch.stack(last_hidden, dim=0)
        return self.lm_head(last_hidden_tensor)

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
            input_ids: Input token ids shaped `[B, 1]`, usually the token that
                was just sampled on the previous step.
            positions: Absolute decode position per request, shaped `[B, 1]`.
                This is the current cached sequence length before appending the
                new token.
            kv_cache: Shared paged KV cache containing the full prefix for each
                request.
        """
        hidden = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden
            normed = layer["input_norm"](hidden)
            q, k_new, v_new = self._project_qkv(layer, normed, positions)
            attn_out_heads = torch.zeros_like(q)

            for req_idx, req in enumerate(requests):
                past_len = req.cached_seq_len
                k_past, v_past = kv_cache.gather_tokens(layer_idx, req.block_ids, req.cached_seq_len)
                q_i = q[req_idx, :1]
                k_new_i = k_new[req_idx, :1]
                v_new_i = v_new[req_idx, :1]
                k_full = torch.cat([repeat_kv(k_past, self.config.num_attention_heads), repeat_kv(k_new_i, self.config.num_attention_heads)], dim=0)
                v_full = torch.cat([repeat_kv(v_past, self.config.num_attention_heads), repeat_kv(v_new_i, self.config.num_attention_heads)], dim=0)
                attn_out_heads[req_idx, :1] = sdpa_attention(q_i, k_full, v_full, past_len)
                kv_cache.write_tokens(layer_idx, req.block_ids, req.cached_seq_len, k_new_i, v_new_i)

            attn_out = layer["o_proj"](attn_out_heads.reshape(attn_out_heads.shape[0], attn_out_heads.shape[1], -1))
            hidden = residual + attn_out
            residual = hidden
            normed = layer["post_norm"](hidden)
            ff = layer["down_proj"](swiglu(layer["gate_proj"](normed), layer["up_proj"](normed)))
            hidden = residual + ff

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden[:, -1])
        return logits
