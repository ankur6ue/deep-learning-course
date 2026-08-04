#!/usr/bin/env python3
"""Tutorial 2: batched dense attention over cached contexts.

The serving step is explicit:

    Q     = projections of the current query chunk
    K/V   = [dense cached context | current chunk]
    output = attention for the current query rows only

The script first prefills each request's dense per-layer K/V cache. It then
processes a separate ragged query chunk through two attention-only layers.
"""

from __future__ import annotations

import math

import torch

from _attention_tutorial_common import (
    REQUESTS,
    assert_outputs_match,
    build_model,
    chunk_sequences,
    context_sequences,
    embed_padded,
    print_request_examples,
    reference_suffixes,
    through_chunk_sequences,
    valid_rows,
)


class DenseKVCache:
    """Ragged per-request K/V tensors without pages or block tables."""

    def __init__(self, model) -> None:
        config = model.config
        empty_shape = (0, config.num_heads, config.head_dim)
        self.k_layers = [
            [torch.empty(empty_shape) for _ in REQUESTS]
            for _ in range(config.num_layers)
        ]
        self.v_layers = [
            [torch.empty(empty_shape) for _ in REQUESTS]
            for _ in range(config.num_layers)
        ]


def assemble_context_and_current_kv(
    cache: DenseKVCache,
    layer_idx: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    query_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense `[cached context | current chunk | padding]` tensors."""
    past_lens = torch.tensor(
        [
            cache.k_layers[layer_idx][request_idx].shape[0]
            for request_idx in range(k_new.shape[0])
        ],
        dtype=torch.long,
    )
    key_lens = past_lens + query_lens
    max_key_len = int(key_lens.max().item())
    full_shape = (k_new.shape[0], max_key_len, k_new.shape[2], k_new.shape[3])
    k_full = torch.zeros(full_shape)
    v_full = torch.zeros(full_shape)

    for request_idx in range(k_new.shape[0]):
        past_len = int(past_lens[request_idx].item())
        query_len = int(query_lens[request_idx].item())
        k_full[request_idx, :past_len] = cache.k_layers[layer_idx][request_idx]
        v_full[request_idx, :past_len] = cache.v_layers[layer_idx][request_idx]
        k_full[request_idx, past_len:past_len + query_len] = k_new[
            request_idx,
            :query_len,
        ]
        v_full[request_idx, past_len:past_len + query_len] = v_new[
            request_idx,
            :query_len,
        ]

    return k_full, v_full, past_lens, key_lens


def dense_query_attention(
    q: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend current Q rows to dense context-plus-current K/V."""
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k_full)
    scores = scores / math.sqrt(q.shape[-1])

    query_offset = torch.arange(q.shape[1]).view(1, 1, q.shape[1], 1)
    key_position = torch.arange(k_full.shape[1]).view(1, 1, 1, k_full.shape[1])
    absolute_query_position = past_lens[:, None, None, None] + query_offset
    query_is_real = query_offset < query_lens[:, None, None, None]
    key_is_real = key_position < key_lens[:, None, None, None]
    causal = key_position <= absolute_query_position
    allowed = query_is_real & key_is_real & causal

    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    # Padded query rows have no valid keys. Give only those rows finite scores,
    # then zero their outputs after softmax.
    masked_scores = torch.where(
        query_is_real,
        masked_scores,
        torch.zeros_like(masked_scores),
    )
    probabilities = torch.softmax(masked_scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, v_full)
    output_is_real = query_is_real[:, 0].unsqueeze(-1)
    return output * output_is_real, allowed


def update_dense_cache(
    cache: DenseKVCache,
    layer_idx: int,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    key_lens: torch.Tensor,
) -> None:
    for request_idx in range(k_full.shape[0]):
        key_len = int(key_lens[request_idx].item())
        cache.k_layers[layer_idx][request_idx] = k_full[
            request_idx,
            :key_len,
        ].clone()
        cache.v_layers[layer_idx][request_idx] = v_full[
            request_idx,
            :key_len,
        ].clone()


@torch.no_grad()
def run_dense_chunk(
    model,
    cache: DenseKVCache,
    token_sequences: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Process one ragged chunk and append its K/V to the dense cache."""
    hidden, query_lens, valid_queries = embed_padded(model, token_sequences)
    first_layer_debug = {}

    for layer_idx, layer in enumerate(model.layers):
        q, k_new, v_new = layer.project(hidden)
        k_full, v_full, past_lens, key_lens = assemble_context_and_current_kv(
            cache,
            layer_idx,
            k_new,
            v_new,
            query_lens,
        )
        attention, allowed = dense_query_attention(
            q,
            k_full,
            v_full,
            past_lens,
            query_lens,
            key_lens,
        )
        update_dense_cache(
            cache,
            layer_idx,
            k_full,
            v_full,
            key_lens,
        )
        hidden = layer.finish(hidden, attention)
        hidden = hidden * valid_queries.unsqueeze(-1)

        if layer_idx == 0:
            first_layer_debug = {
                "q": q,
                "k_full": k_full,
                "v_full": v_full,
                "past_lens": past_lens,
                "key_lens": key_lens,
                "allowed": allowed,
            }

    return hidden, query_lens, first_layer_debug


def main() -> None:
    model = build_model()
    cache = DenseKVCache(model)

    # Stage 1: process only the context. This fills dense K/V for every layer.
    run_dense_chunk(model, cache, context_sequences())

    # Stage 2: these are the current query tokens. Their Q attends to context K/V
    # plus causally visible K/V from earlier rows in the same current chunk.
    query_output, query_lens, debug = run_dense_chunk(
        model,
        cache,
        chunk_sequences(),
    )

    print_request_examples()
    print()
    print("Dense context-plus-query attention")
    print("----------------------------------")
    print(f"past_lens:  {debug['past_lens'].tolist()}")
    print(f"query_lens: {query_lens.tolist()}")
    print(f"key_lens:   {debug['key_lens'].tolist()}")
    print(f"Q current:  {tuple(debug['q'].shape)}")
    print(f"K full:     {tuple(debug['k_full'].shape)}")
    print(f"V full:     {tuple(debug['v_full'].shape)}")

    print()
    print("Allowed keys for request_a's two current queries")
    print("------------------------------------------------")
    request_a_query_len = int(query_lens[0].item())
    print(debug["allowed"][0, 0, :request_a_query_len].to(torch.int32))

    print()
    print("Validation against full-sequence dense recomputation")
    print("----------------------------------------------------")
    expected = reference_suffixes(
        model,
        through_chunk_sequences(),
        query_lens,
    )
    assert_outputs_match(valid_rows(query_output, query_lens), expected)

    print()
    print("What the tensors mean")
    print("  - Q contains only the current query chunk.")
    print("  - Cached context contributes K/V, but is not queried again.")
    print("  - Current tokens also contribute K/V for causal chunk attention.")
    print("  - The cache is dense and ragged; paging starts in tutorial 3.")


if __name__ == "__main__":
    main()
