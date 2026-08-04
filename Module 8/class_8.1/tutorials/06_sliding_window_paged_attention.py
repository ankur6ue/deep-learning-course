#!/usr/bin/env python3
"""Tutorial 6: full versus sliding-window paged attention.

Tutorial 5 introduced the paged FlashAttention metadata contract. This tutorial
keeps that contract and adds a per-layer visibility policy:

    layer 0: sliding window of 3 tokens
    layer 1: full causal attention

The paged cache stores every key and value, and `seqused_k` still reports the
complete valid sequence length. A separate `window_size` decides which valid
keys a layer may read. The default path is a readable CPU reference. Optional
FlashAttention modes pass the same paged metadata and the per-layer window to
vLLM's `flash_attn_varlen_func` binding.
"""

from __future__ import annotations

import argparse
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
    decode_sequences,
    embed_padded,
    physical_slot,
    print_block_tables,
    print_request_examples,
    reference_suffixes,
    through_chunk_sequences,
    through_decode_sequences,
    valid_rows,
)


ATTENTION_BACKENDS = ("reference", "flash-attn", "compare")
FLASH_ATTN_ATOL = 5e-3
FLASH_ATTN_RTOL = 5e-3
LAYER_WINDOWS: tuple[int | None, ...] = (3, None)


class FlashAttentionKernel:
    """Call paged FlashAttention with full or local causal visibility."""

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
                "used by simple_vllm_engine."
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
        window_size: int | None,
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
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive or None")

        # FlashAttention counts the current query among the visible keys.
        # A three-token causal window is therefore two keys to the left and
        # the current key: (2, 0).
        flash_window = (
            (-1, -1)
            if window_size is None
            else (window_size - 1, 0)
        )
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
                window_size=flash_window,
                block_table=block_tables,
                fa_version=self._version(q_flat.shape[-1]),
            )
        except Exception as exc:
            raise RuntimeError(
                "The vLLM paged-varlen FlashAttention call failed. Verify that "
                "the installed vLLM wheel supports this GPU, PyTorch build, and "
                "the window_size argument."
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
    cu_seqlens_q = torch.zeros(
        query_lens.numel() + 1,
        device=q.device,
        dtype=torch.int32,
    )
    cu_seqlens_q[1:] = torch.cumsum(query_lens.to(torch.int32), dim=0)
    return q_flat, cu_seqlens_q, valid_query_rows


def read_logical_range(
    flat_cache: torch.Tensor,
    block_table: torch.Tensor,
    first_position: int,
    last_position: int,
) -> torch.Tensor:
    """Resolve a logical inclusive key range through one block-table row."""
    slots = [
        physical_slot(block_table, token_position)
        for token_position in range(first_position, last_position + 1)
    ]
    indices = torch.tensor(slots, device=flat_cache.device, dtype=torch.long)
    return flat_cache[indices]


def paged_attention_reference(
    q_flat: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    cu_seqlens_q: torch.Tensor,
    query_lens: torch.Tensor,
    seqused_k: torch.Tensor,
    block_tables: torch.Tensor,
    window_size: int | None,
) -> torch.Tensor:
    """Readable full or local causal attention over the paged cache."""
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
            query_position = first_query_position + flat_idx - query_start
            first_visible_key = 0
            if window_size is not None:
                first_visible_key = max(
                    0,
                    query_position - window_size + 1,
                )
            k_visible = read_logical_range(
                flat_k,
                block_tables[request_idx],
                first_visible_key,
                query_position,
            )
            v_visible = read_logical_range(
                flat_v,
                block_tables[request_idx],
                first_visible_key,
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
    # The padded Q width is the largest request query length, which
    # FlashAttention needs as a host-side launch parameter.
    max_seqlen_q = hidden.shape[1]
    block_tables = build_batch_block_tables(key_lens)
    slot_mapping = build_slot_mapping(
        past_lens,
        query_lens,
        max_query_len=max_seqlen_q,
        block_tables=block_tables,
    )
    first_layer_debug = {}
    comparison_differences = []

    for layer_idx, (layer, window_size) in enumerate(
        zip(model.layers, LAYER_WINDOWS, strict=True)
    ):
        q, k_new, v_new = layer.project(hidden)

        # Request layout and visibility metadata stay constant across layers.
        # Only the projected K/V values and this layer's window differ.
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
            reference_output = paged_attention_reference(
                q_flat,
                cache,
                layer_idx,
                cu_seqlens_q,
                query_lens,
                seqused_k=key_lens,
                block_tables=block_tables,
                window_size=window_size,
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
                window_size=window_size,
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


def visible_decode_keys(key_len: int, window_size: int | None) -> list[int]:
    """Return one decode query's visible logical key positions."""
    query_position = key_len - 1
    first_key = 0
    if window_size is not None:
        first_key = max(0, query_position - window_size + 1)
    return list(range(first_key, query_position + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare readable full/local paged attention with the vLLM "
            "FlashAttention kernel."
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

    chunk_output, chunk_lens, chunk_debug, chunk_comparisons = run_chunk(
        model,
        cache,
        chunk_sequences(),
        past_lens=context_lens,
        attention_backend=args.attention_backend,
        flash_attention=flash_attention,
    )
    lengths_through_chunk = context_lens + chunk_lens

    decode_output, decode_lens, decode_debug, decode_comparisons = run_chunk(
        model,
        cache,
        decode_sequences(),
        past_lens=lengths_through_chunk,
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
    print(f"chunk:   {chunk_debug['block_tables'].shape[1]}")
    print(f"decode:  {decode_debug['block_tables'].shape[1]}")
    print()
    print("Per-layer attention policy")
    print("--------------------------")
    print("layer 0: window_size=3 -> FlashAttention window_size=(2, 0)")
    print("layer 1: window_size=None -> FlashAttention window_size=(-1, -1)")
    print()
    print("Ragged prefill metadata shared by both layers")
    print("------------------------------------------------")
    print(f"query_lens:          {chunk_lens.tolist()}")
    print(f"seqused_k:           {chunk_debug['key_lens'].tolist()}")
    print(f"cu_seqlens_q:        {chunk_debug['cu_seqlens_q'].tolist()}")
    print(f"q_flat shape:        {tuple(chunk_debug['q_flat'].shape)}")
    print(
        "block_tables shape:   "
        f"{tuple(chunk_debug['block_tables'].shape)}"
    )
    print("block_tables passed to attention:")
    print(chunk_debug["block_tables"])
    print("prefill_slot_mapping passed to the K/V writer:")
    print(chunk_debug["slot_mapping"])

    print()
    print("Decode visibility: same seqused_k, different visible ranges")
    print("-----------------------------------------------------------")
    for request, key_len in zip(REQUESTS, decode_debug["key_lens"].tolist()):
        local_keys = visible_decode_keys(key_len, LAYER_WINDOWS[0])
        full_keys = visible_decode_keys(key_len, LAYER_WINDOWS[1])
        print(
            f"{request.name}: seqused_k={key_len}, "
            f"layer 0 keys={local_keys}, layer 1 keys={full_keys}"
        )

    if args.attention_backend == "compare":
        print()
        print("Reference versus FlashAttention kernel")
        print("----------------------------------------")
        for stage, differences in (
            ("context", context_comparisons),
            ("chunk", chunk_comparisons),
            ("decode", decode_comparisons),
        ):
            for layer_idx, difference in enumerate(differences):
                print(
                    f"{stage}, layer {layer_idx}: PASS "
                    f"(max |difference|={difference:.3g})"
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
    print()
    print("Chunk validation against mixed-window dense recomputation")
    print("---------------------------------------------------------")
    expected_chunk = reference_suffixes(
        model,
        through_chunk_sequences(),
        chunk_lens,
        layer_windows=LAYER_WINDOWS,
    )
    assert_outputs_match(
        valid_rows(chunk_output, chunk_lens),
        expected_chunk,
        atol=validation_atol,
        rtol=validation_rtol,
    )

    print()
    print("Decode validation against mixed-window dense recomputation")
    print("----------------------------------------------------------")
    expected_decode = reference_suffixes(
        model,
        through_decode_sequences(),
        decode_lens,
        layer_windows=LAYER_WINDOWS,
    )
    assert_outputs_match(
        valid_rows(decode_output, decode_lens),
        expected_decode,
        atol=validation_atol,
        rtol=validation_rtol,
    )

    print()
    print("Modification introduced in tutorial 6")
    print("  - Every layer uses tutorial 5's paged metadata contract.")
    print("  - Each layer supplies its own full or local visibility window.")
    print("  - seqused_k remains the complete number of valid cached keys.")
    print("  - window_size changes reads, not cache contents or write slots.")
    print("  - --backend compare checks each window against FlashAttention.")
    print("  - v10 uses a Triton paged kernel for Gemma's mixed layer shapes.")


if __name__ == "__main__":
    main()
