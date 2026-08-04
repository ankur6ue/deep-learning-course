#!/usr/bin/env python3
"""Tutorial 5: paged varlen attention and slot-mapped K/V writes.

Tutorial 4 gathered a rectangular K/V batch before calling SDPA. This tutorial
keeps K/V paged and switches to the metadata contract used by paged
FlashAttention-style kernels:

    q_flat, cu_seqlens_q, seqused_k, block_table

It also builds physical cache-write metadata:

    prefill_slot_mapping: [batch, max_query_tokens], with -1 padding

A decode iteration is the same operation with one query token per request, so
this tutorial does not introduce or run a separate decode path.

The default attention loop is a CPU reference for that contract. Optional
``flash-attn`` and ``compare`` modes call the same vLLM FlashAttention binding
used by simple_vllm_engine v5.
"""

from __future__ import annotations

import argparse
import math

import torch

from _attention_tutorial_common import (
    BLOCK_SIZE,
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


ATTENTION_BACKENDS = ("reference", "flash-attn", "compare")
FLASH_ATTN_ATOL = 5e-3
FLASH_ATTN_RTOL = 5e-3


class FlashAttentionKernel:
    """Call v5's paged-varlen CUDA binding on the resident paged cache."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FlashAttention mode requires a CUDA-capable PyTorch runtime."
            )

        try:
            from vllm.v1.attention.backends.fa_utils import (
                get_flash_attn_version,
            )
            from vllm.vllm_flash_attn.flash_attn_interface import (
                flash_attn_varlen_func,
            )
        except Exception as exc:
            raise RuntimeError(
                "FlashAttention mode requires the vLLM FlashAttention bindings "
                "used by simple_vllm_engine v5."
            ) from exc

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.get_flash_attn_version = get_flash_attn_version
        self.version_by_head_dim: dict[int, int] = {}

    def _version(self, head_dim: int) -> int:
        version = self.version_by_head_dim.get(head_dim)
        if version is not None:
            return version

        version = self.get_flash_attn_version(head_size=head_dim)
        if version is None:
            raise RuntimeError(
                f"No installed FlashAttention kernel supports head_dim={head_dim}."
            )
        self.version_by_head_dim[head_dim] = int(version)
        return int(version)

    def __call__(
        self,
        q_flat: torch.Tensor,
        cache: PagedKVCache,
        layer_idx: int,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        if not q_flat.is_cuda:
            raise RuntimeError("FlashAttention inputs must be CUDA tensors")
        if q_flat.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("FlashAttention inputs must use float16 or bfloat16")
        if cache.k_layers[layer_idx].shape[1] != BLOCK_SIZE:
            raise RuntimeError("FlashAttention cache page size does not match BLOCK_SIZE")
        if (
            block_tables.device != q_flat.device
            or block_tables.dtype != torch.int32
        ):
            raise RuntimeError("FlashAttention block tables must be CUDA int32 tensors")

        output_flat = torch.empty_like(q_flat)

        try:
            self.flash_attn_varlen_func(
                q=q_flat,
                k=cache.k_layers[layer_idx],
                v=cache.v_layers[layer_idx],
                out=output_flat,
                cu_seqlens_q=cu_seqlens_q.to(
                    device=q_flat.device,
                    dtype=torch.int32,
                ).contiguous(),
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k.to(
                    device=q_flat.device,
                    dtype=torch.int32,
                ).contiguous(),
                max_seqlen_k=int(seqused_k.max().item()),
                softmax_scale=1.0 / math.sqrt(q_flat.shape[-1]),
                causal=True,
                block_table=block_tables,
                fa_version=self._version(q_flat.shape[-1]),
            )
        except Exception as exc:
            raise RuntimeError(
                "The vLLM paged-varlen FlashAttention call failed. Verify that "
                "the installed vLLM wheel supports this GPU and PyTorch build."
            ) from exc

        return output_flat


def build_batch_block_tables(key_lens: torch.Tensor) -> torch.Tensor:
    """Return only block-table rows and columns needed by this active batch."""
    max_blocks = (
        int(key_lens.max().item()) + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    batch_size = key_lens.numel()
    if max_blocks > BLOCK_TABLES.shape[1]:
        raise RuntimeError("The fixture block tables cannot hold this sequence")
    return BLOCK_TABLES[:batch_size, :max_blocks].to(
        device=key_lens.device,
        dtype=torch.int32,
    ).contiguous()


def build_slot_mapping(
    past_lens: torch.Tensor,
    query_lens: torch.Tensor,
    max_query_len: int,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Map padded current-token rows to flat physical cache slots."""
    slot_mapping = torch.full(
        (past_lens.numel(), max_query_len),
        -1,
        device=past_lens.device,
        dtype=torch.long,
    )
    for request_idx in range(past_lens.numel()):
        past_len = int(past_lens[request_idx].item())
        query_len = int(query_lens[request_idx].item())
        for query_offset in range(query_len):
            slot_mapping[request_idx, query_offset] = physical_slot(
                block_tables[request_idx],
                past_len + query_offset,
            )
    return slot_mapping


def write_mapped_kv(
    cache: PagedKVCache,
    layer_idx: int,
    slot_mapping: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> None:
    """Copy valid K/V rows using one shared physical-slot table."""
    slots = slot_mapping.reshape(-1)
    k_rows = k_new.reshape(-1, k_new.shape[-2], k_new.shape[-1])
    v_rows = v_new.reshape(-1, v_new.shape[-2], v_new.shape[-1])
    valid = slots >= 0

    cache.flat_k(layer_idx).index_copy_(0, slots[valid], k_rows[valid])
    cache.flat_v(layer_idx).index_copy_(0, slots[valid], v_rows[valid])


def pack_queries(
    q: torch.Tensor,
    query_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove padded query rows and record each request's flat range."""
    valid_query_rows = (
        torch.arange(q.shape[1], device=q.device).unsqueeze(0)
        < query_lens.unsqueeze(1)
    )
    q_flat = q[valid_query_rows].contiguous()
    cu_seqlens_q = torch.cat(
        [
            torch.zeros(1, device=q.device, dtype=torch.int32),
            query_lens.to(torch.int32).cumsum(dim=0),
        ]
    )
    return q_flat, cu_seqlens_q, valid_query_rows


def read_logical_prefix(
    flat_cache: torch.Tensor,
    block_table: torch.Tensor,
    end_position_inclusive: int,
) -> torch.Tensor:
    """Resolve logical key positions through a block table."""
    slots = [
        physical_slot(block_table, token_position)
        for token_position in range(end_position_inclusive + 1)
    ]
    indices = torch.tensor(slots, device=flat_cache.device, dtype=torch.long)
    return flat_cache[indices]


def paged_varlen_attention_reference(
    q_flat: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    query_lens: torch.Tensor,
    seqused_k: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Readable stand-in for a causal paged varlen kernel.

    `seqused_k` gives each request's total valid K/V length. Causal alignment
    places its packed query rows at the end of that sequence. For example,
    query_len=2 and seqused_k=20 means query rows are positions 18 and 19.
    """
    output_flat = torch.empty_like(q_flat)
    flat_k = cache.flat_k(layer_idx)
    flat_v = cache.flat_v(layer_idx)
    scale = 1.0 / math.sqrt(q_flat.shape[-1])

    for request_idx in range(query_lens.numel()):
        query_start = int(cu_seqlens_q[request_idx].item())
        query_end = int(cu_seqlens_q[request_idx + 1].item())
        query_len = int(query_lens[request_idx].item())
        key_len = int(seqused_k[request_idx].item())
        first_query_position = key_len - query_len

        for flat_idx in range(query_start, query_end):
            local_query_idx = flat_idx - query_start
            query_position = first_query_position + local_query_idx
            k_visible = read_logical_prefix(
                flat_k,
                block_tables[request_idx],
                query_position,
            )
            v_visible = read_logical_prefix(
                flat_v,
                block_tables[request_idx],
                query_position,
            )
            scores = torch.einsum(
                "hd,khd->hk",
                q_flat[flat_idx],
                k_visible,
            ) * scale
            probabilities = torch.softmax(scores, dim=-1)
            output_flat[flat_idx] = torch.einsum(
                "hk,khd->hd",
                probabilities,
                v_visible,
            )

    return output_flat


def unpack_queries(
    output_flat: torch.Tensor,
    valid_query_rows: torch.Tensor,
    padded_shape: torch.Size,
) -> torch.Tensor:
    output = torch.zeros(
        padded_shape,
        device=output_flat.device,
        dtype=output_flat.dtype,
    )
    output[valid_query_rows] = output_flat
    return output


@torch.no_grad()
def run_chunk(
    model,
    cache: PagedKVCache,
    token_sequences: list[list[int]],
    past_lens: torch.Tensor,
    attention_backend: str,
    flash_attention: FlashAttentionKernel | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    list[float],
]:
    hidden, query_lens, valid_tokens = embed_padded(model, token_sequences)
    key_lens = past_lens + query_lens
    # FlashAttention needs this host-side launch parameter. The padded batch
    # width is exactly the largest request query length.
    max_seqlen_q = hidden.shape[1]
    block_tables = build_batch_block_tables(key_lens)
    slot_mapping = build_slot_mapping(
        past_lens,
        query_lens,
        max_query_len=hidden.shape[1],
        block_tables=block_tables,
    )
    first_layer_debug = {}
    comparison_differences = []

    for layer_idx, layer in enumerate(model.layers):
        q, k_new, v_new = layer.project(hidden)

        # The map depends on request layout, not layer weights, so every layer
        # reuses the same slots while supplying different K/V values.
        write_mapped_kv(
            cache,
            layer_idx,
            slot_mapping,
            k_new,
            v_new,
        )
        q_flat, cu_seqlens_q, valid_query_rows = pack_queries(q, query_lens)
        reference_output = None
        if attention_backend in ("reference", "compare"):
            reference_output = paged_varlen_attention_reference(
                q_flat,
                cache,
                layer_idx,
                cu_seqlens_q,
                query_lens,
                seqused_k=key_lens,
                block_tables=block_tables,
            )

        kernel_output = None
        if attention_backend in ("flash-attn", "compare"):
            if flash_attention is None:
                raise RuntimeError("FlashAttention backend was not initialized")
            kernel_output = flash_attention(
                q_flat,
                cache,
                layer_idx,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=key_lens,
                block_tables=block_tables,
            )

        if attention_backend == "compare":
            if reference_output is None or kernel_output is None:
                raise RuntimeError("Compare mode requires both attention outputs")
            torch.testing.assert_close(
                kernel_output,
                reference_output,
                atol=FLASH_ATTN_ATOL,
                rtol=FLASH_ATTN_RTOL,
            )
            comparison_differences.append(
                float((kernel_output - reference_output).abs().max().item())
            )

        output_flat = (
            reference_output
            if attention_backend == "reference"
            else kernel_output
        )
        if output_flat is None:
            raise RuntimeError(f"Unsupported attention backend: {attention_backend}")
        attention = unpack_queries(output_flat, valid_query_rows, q.shape)
        hidden = layer.finish(hidden, attention)
        hidden = hidden * valid_tokens.unsqueeze(-1)

        if layer_idx == 0:
            first_layer_debug = {
                "slot_mapping": slot_mapping,
                "cu_seqlens_q": cu_seqlens_q,
                "key_lens": key_lens,
                "q_flat": q_flat,
                "block_tables": block_tables,
            }

    return hidden, query_lens, first_layer_debug, comparison_differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a readable paged-varlen attention reference with the "
            "vLLM FlashAttention kernel used by simple_vllm_engine v5."
        )
    )
    parser.add_argument(
        "--attention-backend",
        "--backend",
        choices=ATTENTION_BACKENDS,
        default="reference",
        help=(
            "reference runs the readable path on CPU; flash-attn runs the CUDA "
            "kernel; compare runs both on CUDA and checks every layer "
            "(default: reference)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uses_flash_attention = args.attention_backend in ("flash-attn", "compare")
    flash_attention = (
        FlashAttentionKernel()
        if uses_flash_attention
        else None
    )
    device = torch.device("cuda" if uses_flash_attention else "cpu")
    dtype = torch.float16 if uses_flash_attention else torch.float32
    model = build_model().to(device=device, dtype=dtype)
    cache = PagedKVCache(model)

    contexts = context_sequences()
    context_lens = torch.tensor(
        [len(tokens) for tokens in contexts],
        device=device,
    )
    _, _, context_debug, context_comparisons = run_chunk(
        model,
        cache,
        contexts,
        past_lens=torch.zeros_like(context_lens),
        attention_backend=args.attention_backend,
        flash_attention=flash_attention,
    )

    chunk_output, chunk_lens, prefill_debug, chunk_comparisons = run_chunk(
        model,
        cache,
        chunk_sequences(),
        past_lens=context_lens,
        attention_backend=args.attention_backend,
        flash_attention=flash_attention,
    )
    print(f"Attention backend: {args.attention_backend}")
    if flash_attention is not None:
        print(
            f"Direct kernel inputs: device={device}, dtype={dtype}, "
            f"block_size={BLOCK_SIZE}, head_dim={model.config.head_dim}"
        )
    print()
    print_request_examples()
    print()
    print_block_tables()
    print()
    print("Block-table columns needed by each batch")
    print("-----------------------------------------")
    print(f"context: {context_debug['block_tables'].shape[1]}")
    print(f"chunk:   {prefill_debug['block_tables'].shape[1]}")
    print()
    print("Ragged prefill metadata")
    print("-----------------------")
    print(f"query_lens:          {chunk_lens.tolist()}")
    print(f"seqused_k:           {prefill_debug['key_lens'].tolist()}")
    print(f"cu_seqlens_q:        {prefill_debug['cu_seqlens_q'].tolist()}")
    print(f"q_flat shape:        {tuple(prefill_debug['q_flat'].shape)}")
    print(
        "block_tables shape:   "
        f"{tuple(prefill_debug['block_tables'].shape)}"
    )
    print("block_tables passed to attention:")
    print(prefill_debug["block_tables"])
    print("prefill_slot_mapping:")
    print(prefill_debug["slot_mapping"])

    if args.attention_backend == "compare":
        print()
        print("Reference versus FlashAttention kernel")
        print("----------------------------------------")
        for stage, differences in (
            ("context", context_comparisons),
            ("chunk", chunk_comparisons),
        ):
            for layer_idx, difference in enumerate(differences):
                print(
                    f"{stage}, layer {layer_idx}: PASS "
                    f"(max |difference|={difference:.3g})"
                )

    print()
    print("Chunk validation against full dense recomputation")
    print("-------------------------------------------------")
    expected_chunk = reference_suffixes(
        model,
        through_chunk_sequences(),
        chunk_lens,
    )
    validation_atol = (
        1e-5
        if args.attention_backend == "reference"
        else FLASH_ATTN_ATOL
    )
    validation_rtol = (
        1e-5
        if args.attention_backend == "reference"
        else FLASH_ATTN_RTOL
    )
    assert_outputs_match(
        valid_rows(chunk_output, chunk_lens),
        expected_chunk,
        atol=validation_atol,
        rtol=validation_rtol,
    )

    print()
    print("Modification introduced in tutorial 5")
    print("  - Valid queries are packed and described by cu_seqlens_q.")
    print("  - seqused_k reports each request's valid cached length.")
    print("  - FlashAttention-style causal alignment replaces a dense mask.")
    print("  - Slot mappings write K/V before paged attention reads the cache.")
    print("  - Decode is the same operation with query_len=1, not a separate path.")
    print("  - --backend compare checks the readable reference against the v5 kernel.")


if __name__ == "__main__":
    main()
