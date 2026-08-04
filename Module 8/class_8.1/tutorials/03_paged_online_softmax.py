#!/usr/bin/env python3
"""Tutorial 3: direct paged attention with block-wise online softmax.

The permanent K/V cache is now paged. Attention walks each request's block
table directly and updates softmax statistics one physical block at a time.
No dense `[batch, max_key_length, ...]` K/V tensor is assembled.

This is intentionally a slow Python reference. Its structure mirrors the
direct-paged reference attention introduced by simple_vllm_engine v2.
"""

from __future__ import annotations

import math

import torch

from _attention_tutorial_common import (
    BLOCK_SIZE,
    BLOCK_TABLES,
    REQUESTS,
    PagedKVCache,
    assert_outputs_match,
    build_model,
    chunk_sequences,
    context_sequences,
    embed_padded,
    physical_slot,
    print_block_tables,
    print_request_examples,
    reference_suffixes,
    through_chunk_sequences,
    valid_rows,
)


def write_chunk_to_cache(
    cache: PagedKVCache,
    layer_idx: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> None:
    """Write each valid current token through its request's block table."""
    flat_k = cache.flat_k(layer_idx)
    flat_v = cache.flat_v(layer_idx)

    for request_idx in range(k_new.shape[0]):
        past_len = int(past_lens[request_idx].item())
        query_len = int(query_lens[request_idx].item())
        for query_offset in range(query_len):
            token_position = past_len + query_offset
            slot = physical_slot(BLOCK_TABLES[request_idx], token_position)
            flat_k[slot] = k_new[request_idx, query_offset]
            flat_v[slot] = v_new[request_idx, query_offset]


def paged_online_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
    trace_query: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Read paged K/V and update softmax state block by block.

    `trace_query=(request_idx, query_offset)` records the running state for one
    query so the numerical update is visible in the printed tutorial output.
    """
    output = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    trace = []

    for request_idx in range(q.shape[0]):
        query_len = int(query_lens[request_idx].item())
        past_len = int(past_lens[request_idx].item())

        for query_offset in range(query_len):
            query_position = past_len + query_offset
            query = q[request_idx, query_offset]

            running_max = torch.full((q.shape[2],), float("-inf"))
            running_denominator = torch.zeros(q.shape[2])
            running_numerator = torch.zeros_like(query)

            final_logical_block = query_position // BLOCK_SIZE
            for logical_block in range(final_logical_block + 1):
                physical_block = int(
                    BLOCK_TABLES[request_idx, logical_block].item()
                )
                first_key_position = logical_block * BLOCK_SIZE
                keys_in_block = min(
                    BLOCK_SIZE,
                    query_position + 1 - first_key_position,
                )

                k_block = cache.k_layers[layer_idx][
                    physical_block,
                    :keys_in_block,
                ]
                v_block = cache.v_layers[layer_idx][
                    physical_block,
                    :keys_in_block,
                ]
                block_scores = torch.einsum("hd,khd->hk", query, k_block) * scale

                block_max = block_scores.max(dim=-1).values
                new_max = torch.maximum(running_max, block_max)
                old_rescale = torch.exp(running_max - new_max)
                block_exponentials = torch.exp(block_scores - new_max[:, None])

                running_numerator = (
                    running_numerator * old_rescale[:, None]
                    + torch.einsum("hk,khd->hd", block_exponentials, v_block)
                )
                running_denominator = (
                    running_denominator * old_rescale
                    + block_exponentials.sum(dim=-1)
                )
                running_max = new_max

                if trace_query == (request_idx, query_offset):
                    trace.append(
                        {
                            "logical_block": logical_block,
                            "physical_block": physical_block,
                            "key_positions": list(
                                range(
                                    first_key_position,
                                    first_key_position + keys_in_block,
                                )
                            ),
                            "running_max": running_max.tolist(),
                            "running_denominator": running_denominator.tolist(),
                        }
                    )

            output[request_idx, query_offset] = (
                running_numerator / running_denominator[:, None]
            )

    return output, trace


@torch.no_grad()
def run_chunk(
    model,
    cache: PagedKVCache,
    token_sequences: list[list[int]],
    past_lens: torch.Tensor,
    trace_layer_zero_query: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    """Run one ragged prefill chunk through both attention-only layers."""
    hidden, query_lens, valid_tokens = embed_padded(model, token_sequences)
    recorded_trace = []

    for layer_idx, layer in enumerate(model.layers):
        q, k_new, v_new = layer.project(hidden)
        write_chunk_to_cache(
            cache,
            layer_idx,
            k_new,
            v_new,
            past_lens,
            query_lens,
        )
        attention, trace = paged_online_attention(
            q,
            cache,
            layer_idx,
            past_lens,
            query_lens,
            trace_query=trace_layer_zero_query if layer_idx == 0 else None,
        )
        hidden = layer.finish(hidden, attention)
        hidden = hidden * valid_tokens.unsqueeze(-1)
        if trace:
            recorded_trace = trace

    return hidden, query_lens, recorded_trace


def print_chunk_slots(past_lens: torch.Tensor, query_lens: torch.Tensor) -> None:
    print("Physical slots used by the appended chunk")
    print("-----------------------------------------")
    for request_idx, request in enumerate(REQUESTS):
        past_len = int(past_lens[request_idx].item())
        query_len = int(query_lens[request_idx].item())
        slots = [
            physical_slot(BLOCK_TABLES[request_idx], past_len + offset)
            for offset in range(query_len)
        ]
        print(
            f"{request.name}: past_len={past_len}, query_len={query_len}, "
            f"key_len={past_len + query_len}, slots={slots}"
        )


def main() -> None:
    model = build_model()
    cache = PagedKVCache(model)

    contexts = context_sequences()
    context_lens = torch.tensor([len(tokens) for tokens in contexts])
    run_chunk(
        model,
        cache,
        contexts,
        past_lens=torch.zeros_like(context_lens),
    )

    chunks = chunk_sequences()
    chunk_output, chunk_lens, trace = run_chunk(
        model,
        cache,
        chunks,
        past_lens=context_lens,
        # Trace request_a's second new query. It sees 20 keys over two pages.
        trace_layer_zero_query=(0, 1),
    )

    print_request_examples()
    print()
    print_block_tables()
    print()
    print_chunk_slots(context_lens, chunk_lens)

    print()
    print("Online-softmax trace: layer 0, request_a, second chunk query")
    print("----------------------------------------------------------------")
    for row in trace:
        print(
            f"logical block {row['logical_block']} -> physical block "
            f"{row['physical_block']}, keys={row['key_positions']}, "
            f"running max={row['running_max']}, "
            f"running denominator={row['running_denominator']}"
        )

    print()
    print("Validation against full dense recomputation")
    print("-------------------------------------------")
    expected = reference_suffixes(
        model,
        through_chunk_sequences(),
        chunk_lens,
    )
    assert_outputs_match(valid_rows(chunk_output, chunk_lens), expected)

    print()
    print("Modification introduced in tutorial 3")
    print("  - K/V live in non-contiguous physical pages.")
    print("  - Each query walks its request's logical-to-physical block table.")
    print("  - Softmax keeps a running max, denominator, and weighted V sum.")
    print("  - Memory stays paged, but Python loops make this a reference path.")


if __name__ == "__main__":
    main()
