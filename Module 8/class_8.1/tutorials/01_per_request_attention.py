#!/usr/bin/env python3
"""Tutorial 1: calculate dense attention one request at a time.

There is no batch dimension and no padding in this tutorial. Each request owns:

    Q_current: [its query length, heads, head_dim]
    K/V_full:  [its context length + query length, heads, head_dim]

The requests therefore keep their natural, different tensor sizes. Tutorial 2
will combine these ragged calculations into one padded rectangular batch.
"""

from __future__ import annotations

import math

import torch

from _attention_tutorial_common import (
    REQUESTS,
    assert_outputs_match,
    build_model,
    print_request_examples,
    reference_suffixes,
    through_chunk_sequences,
)


class PerRequestKVCache:
    """One request's dense K/V history for every attention layer."""

    def __init__(self, model) -> None:
        config = model.config
        empty_shape = (0, config.num_heads, config.head_dim)
        self.k_layers = [
            torch.empty(empty_shape)
            for _ in range(config.num_layers)
        ]
        self.v_layers = [
            torch.empty(empty_shape)
            for _ in range(config.num_layers)
        ]


def attention_for_one_request(
    q_current: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    past_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate causal attention without batching or padding.

    Current query row `i` has absolute position `past_len + i`, so it may read
    all context keys plus current-chunk keys through row `i`.
    """
    scores = torch.einsum("qhd,khd->hqk", q_current, k_full)
    scores = scores / math.sqrt(q_current.shape[-1])

    query_positions = past_len + torch.arange(q_current.shape[0]).unsqueeze(1)
    key_positions = torch.arange(k_full.shape[0]).unsqueeze(0)
    allowed = key_positions <= query_positions

    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, v_full)
    return output, allowed


@torch.no_grad()
def run_request_chunk(
    model,
    cache: PerRequestKVCache,
    token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Process one request's current chunk through both model layers."""
    hidden = model.token_embedding(torch.tensor(token_ids, dtype=torch.long))
    first_layer_debug = {}

    for layer_idx, layer in enumerate(model.layers):
        q_current, k_current, v_current = layer.project(hidden)
        past_len = cache.k_layers[layer_idx].shape[0]
        k_full = torch.cat([cache.k_layers[layer_idx], k_current], dim=0)
        v_full = torch.cat([cache.v_layers[layer_idx], v_current], dim=0)

        attention, allowed = attention_for_one_request(
            q_current,
            k_full,
            v_full,
            past_len,
        )
        cache.k_layers[layer_idx] = k_full.clone()
        cache.v_layers[layer_idx] = v_full.clone()
        hidden = layer.finish(hidden, attention)

        if layer_idx == 0:
            first_layer_debug = {
                "past_len": past_len,
                "q_current": q_current,
                "k_full": k_full,
                "v_full": v_full,
                "allowed": allowed,
            }

    return hidden, first_layer_debug


def main() -> None:
    model = build_model()
    query_outputs = []
    debug_rows = []

    for request in REQUESTS:
        cache = PerRequestKVCache(model)

        # Stage 1 fills this request's dense context K/V.
        run_request_chunk(model, cache, request.context_tokens)

        # Stage 2 computes only this request's current query rows.
        query_output, debug = run_request_chunk(
            model,
            cache,
            request.chunk_tokens,
        )
        query_outputs.append(query_output)
        debug_rows.append(debug)

    print_request_examples()
    print()
    print("Per-request tensor shapes: no batch dimension, no padding")
    print("---------------------------------------------------------")
    for request, debug in zip(REQUESTS, debug_rows):
        q_current = debug["q_current"]
        k_full = debug["k_full"]
        v_full = debug["v_full"]
        print(
            f"{request.name}: past_len={debug['past_len']}, "
            f"Q={tuple(q_current.shape)}, "
            f"K={tuple(k_full.shape)}, V={tuple(v_full.shape)}"
        )
        print("  causal visibility:")
        print(debug["allowed"].to(torch.int32))

    print()
    print("Validation against full-sequence dense recomputation")
    print("----------------------------------------------------")
    query_lens = torch.tensor(
        [len(request.chunk_tokens) for request in REQUESTS],
        dtype=torch.long,
    )
    expected = reference_suffixes(
        model,
        through_chunk_sequences(),
        query_lens,
    )
    assert_outputs_match(query_outputs, expected)

    print()
    print("What tutorial 1 deliberately does not do")
    print("  - It does not stack requests into a batch.")
    print("  - It does not pad Q, K, V, or attention masks.")
    print("  - It does not use pages or block tables.")
    print("  - Tutorial 2 introduces one padded dense batch for all requests.")


if __name__ == "__main__":
    main()
