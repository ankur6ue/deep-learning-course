#!/usr/bin/env python3
"""Shared fixtures for tutorials 1-6.

This module deliberately does not implement any of the numbered tutorial
backends. It only keeps the model weights, request data, paged-cache storage,
and dense correctness reference identical across the series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RequestExample:
    name: str
    context_tokens: tuple[int, ...]
    chunk_tokens: tuple[int, ...]
    decode_token: int


REQUESTS = (
    RequestExample(
        name="request_a",
        context_tokens=tuple(range(1, 19)),
        chunk_tokens=(19, 20),
        decode_token=21,
    ),
    RequestExample(
        name="request_b",
        context_tokens=tuple(range(22, 54)),
        chunk_tokens=(54,),
        decode_token=55,
    ),
    RequestExample(
        name="request_c",
        context_tokens=tuple(range(40, 55)),
        chunk_tokens=(56, 57, 58),
        decode_token=59,
    ),
)

BLOCK_SIZE = 16
BLOCK_TABLES = torch.tensor(
    [
        [8, 1, 10, 3],
        [0, 9, 4, 11],
        [6, 2, 7, 5],
    ],
    dtype=torch.long,
)
NUM_PHYSICAL_BLOCKS = int(BLOCK_TABLES.max().item()) + 1


@dataclass(frozen=True)
class TinyModelConfig:
    vocab_size: int = 64
    hidden_size: int = 16
    num_heads: int = 2
    num_layers: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class AttentionOnlyLayer(nn.Module):
    """One projection-attention-output-projection block.

    The tutorials omit normalization, positional encoding, and the MLP so the
    only changing part of the model is how attention obtains and consumes K/V.
    """

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def project(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_shape = (*hidden.shape[:-1], self.config.num_heads, self.config.head_dim)
        q = self.q_proj(hidden).view(output_shape)
        k = self.k_proj(hidden).view(output_shape)
        v = self.v_proj(hidden).view(output_shape)
        return q, k, v

    def finish(self, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        attention_flat = attention.reshape(*hidden.shape[:-1], self.config.hidden_size)
        return hidden + self.o_proj(attention_flat)


class AttentionOnlyTransformer(nn.Module):
    """A deterministic two-layer transformer scaffold without a fixed backend."""

    def __init__(self, config: TinyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            AttentionOnlyLayer(config) for _ in range(config.num_layers)
        )


def build_model() -> AttentionOnlyTransformer:
    """Build the same model weights without changing the caller's RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return AttentionOnlyTransformer(TinyModelConfig()).eval()


def context_sequences() -> list[list[int]]:
    return [list(request.context_tokens) for request in REQUESTS]


def chunk_sequences() -> list[list[int]]:
    return [list(request.chunk_tokens) for request in REQUESTS]


def decode_sequences() -> list[list[int]]:
    return [[request.decode_token] for request in REQUESTS]


def through_chunk_sequences() -> list[list[int]]:
    return [
        list(request.context_tokens + request.chunk_tokens)
        for request in REQUESTS
    ]


def through_decode_sequences() -> list[list[int]]:
    return [
        list(request.context_tokens + request.chunk_tokens + (request.decode_token,))
        for request in REQUESTS
    ]


def pad_token_sequences(
    sequences: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad token ids and return ids, lengths, and a valid-token mask."""
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max().item())
    token_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
    valid_tokens = torch.zeros((len(sequences), max_length), dtype=torch.bool)

    for row, sequence in enumerate(sequences):
        length = len(sequence)
        token_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
        valid_tokens[row, :length] = True

    return token_ids, lengths, valid_tokens


def embed_padded(
    model: AttentionOnlyTransformer,
    sequences: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids, lengths, valid_tokens = pad_token_sequences(sequences)
    device = model.token_embedding.weight.device
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)
    valid_tokens = valid_tokens.to(device)
    hidden = model.token_embedding(token_ids)
    hidden = hidden * valid_tokens.unsqueeze(-1)
    return hidden, lengths, valid_tokens


def physical_slot(
    block_table: torch.Tensor,
    token_position: int,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Map a logical token position to a flat physical cache slot."""
    logical_block, block_offset = divmod(token_position, block_size)
    physical_block = int(block_table[logical_block].item())
    return physical_block * block_size + block_offset


class PagedKVCache:
    """Per-layer K/V tensors in `[physical_block, block_offset, head, dim]` order."""

    def __init__(self, model: AttentionOnlyTransformer) -> None:
        config = model.config
        model_tensor = model.token_embedding.weight
        shape = (
            NUM_PHYSICAL_BLOCKS,
            BLOCK_SIZE,
            config.num_heads,
            config.head_dim,
        )
        self.k_layers = [
            torch.zeros(
                shape,
                device=model_tensor.device,
                dtype=model_tensor.dtype,
            )
            for _ in range(config.num_layers)
        ]
        self.v_layers = [
            torch.zeros(
                shape,
                device=model_tensor.device,
                dtype=model_tensor.dtype,
            )
            for _ in range(config.num_layers)
        ]

    def flat_k(self, layer_idx: int) -> torch.Tensor:
        layer = self.k_layers[layer_idx]
        return layer.view(NUM_PHYSICAL_BLOCKS * BLOCK_SIZE, *layer.shape[2:])

    def flat_v(self, layer_idx: int) -> torch.Tensor:
        layer = self.v_layers[layer_idx]
        return layer.view(NUM_PHYSICAL_BLOCKS * BLOCK_SIZE, *layer.shape[2:])


def _dense_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int | None,
) -> torch.Tensor:
    """Unbatched dense attention used only as a correctness oracle."""
    sequence_length = q.shape[0]
    query_positions = torch.arange(sequence_length, device=q.device).unsqueeze(1)
    key_positions = torch.arange(sequence_length, device=q.device).unsqueeze(0)
    allowed = key_positions <= query_positions
    if window_size is not None:
        allowed &= key_positions >= query_positions - window_size + 1

    scores = torch.einsum("thd,shd->hts", q, k)
    scores = scores / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probabilities, v)


@torch.no_grad()
def dense_reference_forward(
    model: AttentionOnlyTransformer,
    sequences: list[list[int]],
    layer_windows: tuple[int | None, ...] | None = None,
) -> list[torch.Tensor]:
    """Run each request independently with ordinary dense causal attention."""
    if layer_windows is None:
        layer_windows = (None,) * model.config.num_layers
    if len(layer_windows) != model.config.num_layers:
        raise ValueError("layer_windows must contain one entry per model layer")

    outputs = []
    for sequence in sequences:
        token_ids = torch.tensor(
            sequence,
            device=model.token_embedding.weight.device,
            dtype=torch.long,
        )
        hidden = model.token_embedding(token_ids)
        for layer, window_size in zip(model.layers, layer_windows):
            q, k, v = layer.project(hidden)
            attention = _dense_reference_attention(q, k, v, window_size)
            hidden = layer.finish(hidden, attention)
        outputs.append(hidden)
    return outputs


def valid_rows(padded: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
    return [
        padded[row, : int(length.item())]
        for row, length in enumerate(lengths)
    ]


def reference_suffixes(
    model: AttentionOnlyTransformer,
    full_sequences: list[list[int]],
    suffix_lengths: torch.Tensor,
    layer_windows: tuple[int | None, ...] | None = None,
) -> list[torch.Tensor]:
    full_outputs = dense_reference_forward(model, full_sequences, layer_windows)
    return [
        output[-int(length.item()):]
        for output, length in zip(full_outputs, suffix_lengths)
    ]


def assert_outputs_match(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Validate each ragged request separately and print useful error magnitudes."""
    for request, actual_output, expected_output in zip(REQUESTS, actual, expected):
        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=atol,
            rtol=rtol,
        )
        max_difference = float((actual_output - expected_output).abs().max().item())
        print(f"  {request.name}: PASS (max |difference|={max_difference:.3g})")


def print_request_examples() -> None:
    print("Requests used throughout tutorials 1-6")
    print("--------------------------------------")
    for request in REQUESTS:
        context = (
            f"{list(request.context_tokens[:4])} ... "
            f"{list(request.context_tokens[-3:])} "
            f"({len(request.context_tokens)} tokens)"
        )
        print(
            f"{request.name}: context={context}, "
            f"next chunk={list(request.chunk_tokens)}, "
            f"decode token={request.decode_token}"
        )


def print_block_tables() -> None:
    print("Logical-to-physical block tables")
    print("--------------------------------")
    for request, table in zip(REQUESTS, BLOCK_TABLES):
        print(f"{request.name}: {table.tolist()}")
