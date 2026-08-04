#!/usr/bin/env python3
"""Tutorial 4: gather paged K/V into dense tensors and call PyTorch SDPA.

The permanent cache remains paged, but attention temporarily materializes:

    k_past/v_past -> k_full/v_full -> additive mask -> PyTorch SDPA

This removes the Python attention math from tutorial 3 at the cost of gathering
and padding K/V on every layer and every scheduler step. It mirrors
simple_vllm_engine v3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from _attention_tutorial_common import (
    BLOCK_TABLES,
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


def gather_past_kv(
    cache: PagedKVCache,
    layer_idx: int,
    past_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather each request's paged prefix into a padded dense tensor."""
    batch_size = past_lens.numel()
    max_past_len = int(past_lens.max().item())
    cache_shape = cache.k_layers[layer_idx].shape
    dense_shape = (batch_size, max_past_len, cache_shape[2], cache_shape[3])
    k_past = torch.zeros(dense_shape)
    v_past = torch.zeros(dense_shape)
    flat_k = cache.flat_k(layer_idx)
    flat_v = cache.flat_v(layer_idx)

    for request_idx in range(batch_size):
        past_len = int(past_lens[request_idx].item())
        for token_position in range(past_len):
            slot = physical_slot(BLOCK_TABLES[request_idx], token_position)
            k_past[request_idx, token_position] = flat_k[slot]
            v_past[request_idx, token_position] = flat_v[slot]

    return k_past, v_past


def assemble_full_kv(
    k_past: torch.Tensor,
    v_past: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create `[cached prefix | current chunk | padding]` for every request."""
    key_lens = past_lens + query_lens
    max_key_len = int(key_lens.max().item())
    full_shape = (k_new.shape[0], max_key_len, k_new.shape[2], k_new.shape[3])
    k_full = torch.zeros(full_shape)
    v_full = torch.zeros(full_shape)

    for request_idx in range(k_new.shape[0]):
        past_len = int(past_lens[request_idx].item())
        query_len = int(query_lens[request_idx].item())
        k_full[request_idx, :past_len] = k_past[request_idx, :past_len]
        v_full[request_idx, :past_len] = v_past[request_idx, :past_len]
        k_full[request_idx, past_len:past_len + query_len] = k_new[
            request_idx,
            :query_len,
        ]
        v_full[request_idx, past_len:past_len + query_len] = v_new[
            request_idx,
            :query_len,
        ]

    return k_full, v_full, key_lens


def build_sdpa_mask(
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    max_query_len: int,
    max_key_len: int,
) -> torch.Tensor:
    """Combine ragged-query, key-padding, and causal visibility rules."""
    query_offset = torch.arange(max_query_len).view(1, max_query_len, 1)
    key_position = torch.arange(max_key_len).view(1, 1, max_key_len)
    query_position = past_lens[:, None, None] + query_offset

    query_is_real = query_offset < query_lens[:, None, None]
    key_is_real = key_position < key_lens[:, None, None]
    causal = key_position <= query_position
    allowed = query_is_real & key_is_real & causal

    mask = torch.where(
        allowed[:, None],
        torch.tensor(0.0),
        torch.tensor(float("-inf")),
    )
    # SDPA cannot softmax an all-negative-infinity padded query row. Its output
    # is discarded, so give only those rows a harmless all-zero mask.
    mask = torch.where(
        query_is_real[:, None],
        mask,
        torch.zeros_like(mask),
    )
    return mask


def write_chunk_to_cache(
    cache: PagedKVCache,
    layer_idx: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> None:
    """Persist current K/V after SDPA has consumed the temporary dense tensors."""
    flat_k = cache.flat_k(layer_idx)
    flat_v = cache.flat_v(layer_idx)
    for request_idx in range(k_new.shape[0]):
        past_len = int(past_lens[request_idx].item())
        query_len = int(query_lens[request_idx].item())
        for query_offset in range(query_len):
            slot = physical_slot(
                BLOCK_TABLES[request_idx],
                past_len + query_offset,
            )
            flat_k[slot] = k_new[request_idx, query_offset]
            flat_v[slot] = v_new[request_idx, query_offset]


@torch.no_grad()
def run_chunk(
    model,
    cache: PagedKVCache,
    token_sequences: list[list[int]],
    past_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    hidden, query_lens, valid_tokens = embed_padded(model, token_sequences)
    first_layer_debug = {}

    for layer_idx, layer in enumerate(model.layers):
        q, k_new, v_new = layer.project(hidden)
        k_past, v_past = gather_past_kv(cache, layer_idx, past_lens)
        k_full, v_full, key_lens = assemble_full_kv(
            k_past,
            v_past,
            k_new,
            v_new,
            past_lens,
            query_lens,
        )
        mask = build_sdpa_mask(
            past_lens,
            query_lens,
            key_lens,
            max_query_len=q.shape[1],
            max_key_len=k_full.shape[1],
        )

        attention = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_full.transpose(1, 2),
            v_full.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
        ).transpose(1, 2)
        attention = attention * valid_tokens[:, :, None, None]

        write_chunk_to_cache(
            cache,
            layer_idx,
            k_new,
            v_new,
            past_lens,
            query_lens,
        )
        hidden = layer.finish(hidden, attention)
        hidden = hidden * valid_tokens.unsqueeze(-1)

        if layer_idx == 0:
            first_layer_debug = {
                "k_past": k_past,
                "k_full": k_full,
                "key_lens": key_lens,
                "mask": mask,
            }

    return hidden, query_lens, first_layer_debug


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

    chunk_output, chunk_lens, debug = run_chunk(
        model,
        cache,
        chunk_sequences(),
        past_lens=context_lens,
    )

    print_request_examples()
    print()
    print_block_tables()
    print()
    print("Dense temporary tensors for the appended chunk")
    print("----------------------------------------------")
    print(f"past_lens:   {context_lens.tolist()}")
    print(f"query_lens:  {chunk_lens.tolist()}")
    print(f"key_lens:    {debug['key_lens'].tolist()}")
    print(f"k_past:      {tuple(debug['k_past'].shape)}")
    print(f"k_full:      {tuple(debug['k_full'].shape)}")
    print(f"SDPA mask:   {tuple(debug['mask'].shape)}")
    print()
    print("Finite mask entries for request_a (1=visible, 0=masked)")
    request_a_query_len = int(chunk_lens[0].item())
    print(
        torch.isfinite(debug["mask"][0, 0, :request_a_query_len]).to(torch.int32)
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
    print("Modification introduced in tutorial 4")
    print("  - Paged K/V remain the long-lived storage format.")
    print("  - Cached pages are gathered into padded k_past/v_past tensors.")
    print("  - Current K/V are appended to form dense k_full/v_full tensors.")
    print("  - PyTorch SDPA replaces the Python online-softmax calculation.")


if __name__ == "__main__":
    main()
