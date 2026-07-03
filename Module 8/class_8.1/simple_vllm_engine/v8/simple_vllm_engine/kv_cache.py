from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import EngineConfig, ModelConfig
from .kernels import triton_reshape_and_cache_flash


@dataclass
class PrefixLookup:
    """Result of a prefix-cache lookup."""

    block_ids: list[int]
    cached_tokens: int


class FullBlockPrefixCache:
    """Tiny full-block prefix cache.

    The cache maps a token prefix to the physical KV block that stores that
    prefix. It only publishes complete blocks. For example, with block_size=4:

        prompt A: [10, 11, 12, 13, 14]
        published prefix: (10, 11, 12, 13) -> block_id

    A later prompt starting with those four tokens can retain and reuse that
    block instead of recomputing its K/V.
    """

    def __init__(self, block_size: int) -> None:
        """Index reusable full prompt blocks by prefix tokens."""
        self.block_size = block_size
        self._full_prefix_to_block_id: dict[tuple[int, ...], int] = {}

    def lookup(self, prompt_ids: list[int]) -> PrefixLookup:
        """Find the longest reusable cached prefix for a prompt.

        Args:
            prompt_ids: Full prompt token ids for a new request. Lookup proceeds
                block by block, so only full blocks are reusable.
        """
        cached_blocks: list[int] = []
        full_blocks = len(prompt_ids) // self.block_size
        # Leave the final full prompt block uncached when the prompt is exactly
        # block-aligned. This avoids needing a special-case "all-prefix-cached
        # but still need next-token logits" path in the teaching engine.
        if len(prompt_ids) % self.block_size == 0 and full_blocks > 0:
            full_blocks -= 1
        for block_idx in range(full_blocks):
            prefix_len = (block_idx + 1) * self.block_size
            key = tuple(prompt_ids[:prefix_len])
            block_id = self._full_prefix_to_block_id.get(key)
            if block_id is None:
                break
            cached_blocks.append(block_id)
        return PrefixLookup(
            block_ids=cached_blocks,
            cached_tokens=len(cached_blocks) * self.block_size,
        )

    def insert_full_blocks(
        self,
        prompt_ids: list[int],
        block_ids: list[int],
        published_until_block: int,
        full_blocks_available: int,
    ) -> tuple[int, list[int]]:
        """Publish newly completed full prompt blocks into the prefix cache.

        Args:
            prompt_ids: Full prompt token ids.
            block_ids: Physical block ids holding this request's KV cache.
            published_until_block: Number of full blocks already published for
                this request.
            full_blocks_available: Number of full blocks currently available in
                the request's cached prefix.
        """
        retained_block_ids: list[int] = []
        for block_idx in range(published_until_block, full_blocks_available):
            prefix_len = (block_idx + 1) * self.block_size
            key = tuple(prompt_ids[:prefix_len])
            if key not in self._full_prefix_to_block_id:
                self._full_prefix_to_block_id[key] = block_ids[block_idx]
                retained_block_ids.append(block_ids[block_idx])
        return full_blocks_available, retained_block_ids


class PagedKVCache:
    """Paged K/V memory shared by all active requests.

    Each request sees a logical sequence of tokens. The cache stores that
    sequence in fixed-size physical pages:

        logical token 0..15   -> request.block_ids[0]
        logical token 16..31  -> request.block_ids[1]

    The physical blocks can be anywhere in the global pool. This is the core
    vLLM idea: grow and free request memory one page at a time instead of
    allocating a single contiguous `[max_seq_len, heads, dim]` buffer per
    request.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        engine_config: EngineConfig,
    ) -> None:
        """Allocate the paged KV cache for every attention layer.

        Args:
            model_config: Determines heads, head dimension, and layer count.
            engine_config: Determines page size, page count, device, and dtype.
        """
        self.model_config = model_config
        self.engine_config = engine_config
        device = torch.device(engine_config.device)
        dtype = engine_config.dtype
        shape = (
            engine_config.num_blocks,
            engine_config.block_size,
            model_config.num_key_value_heads,
            model_config.head_dim,
        )
        self.k_layers = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(model_config.num_layers)]
        self.v_layers = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(model_config.num_layers)]
        self.k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.refcounts = torch.zeros(engine_config.num_blocks, dtype=torch.int32)
        self.free_block_ids = list(range(engine_config.num_blocks - 1, -1, -1))
        self.prefix_cache = FullBlockPrefixCache(engine_config.block_size)

    @property
    def block_size(self) -> int:
        """Return the number of token slots in one physical KV page."""
        return self.engine_config.block_size

    def blocks_needed(self, total_tokens: int) -> int:
        """Return how many pages are needed for `total_tokens` logical tokens."""
        return (total_tokens + self.block_size - 1) // self.block_size

    def physical_slot(self, block_ids: list[int], token_index: int) -> int:
        """Map one logical token index to one flattened physical cache slot.

        Example with `block_size = 16`:

            token_index = 37
            logical_block = 37 // 16 = 2
            block_offset = 37 % 16 = 5
            physical_block = block_ids[2]
            physical_slot = physical_block * 16 + 5

        Attention/KV-write kernels use this flattened slot to write one token's
        K/V vector directly into the cache.
        """
        logical_block, block_offset = divmod(token_index, self.block_size)
        return block_ids[logical_block] * self.block_size + block_offset

    def retain_blocks(self, block_ids: list[int]) -> None:
        """Increase refcounts for shared pages."""
        for block_id in block_ids:
            self.refcounts[block_id] += 1

    def allocate_block(self) -> int:
        """Pop one free physical page from the allocator."""
        if not self.free_block_ids:
            raise RuntimeError("KV cache exhausted: no free blocks remaining")
        block_id = self.free_block_ids.pop()
        self.refcounts[block_id] = 1
        return block_id

    def ensure_capacity(self, block_ids: list[int], total_tokens: int) -> list[int]:
        """Extend a request's block table until it can store `total_tokens`.

        Args:
            block_ids: Current logical-to-physical page mapping for a request.
            total_tokens: Target logical sequence length after appending more
                prompt or decode tokens.
        """
        needed = self.blocks_needed(total_tokens)
        out = list(block_ids)
        while len(out) < needed:
            out.append(self.allocate_block())
        return out

    def release_blocks(self, block_ids: list[int]) -> None:
        """Decrease refcounts and return fully unused pages to the free list."""
        for block_id in block_ids:
            self.refcounts[block_id] -= 1
            if self.refcounts[block_id] < 0:
                raise RuntimeError(f"Negative refcount for block {block_id}")
            if self.refcounts[block_id] == 0:
                self.free_block_ids.append(block_id)

    def write_kv_to_mapped_slots(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        *,
        assume_all_valid: bool = False,
    ) -> None:
        """Write K/V vectors to cache locations described by physical slot ids.

        `slot_mapping` is an input routing table, not something this method
        mutates. Each valid entry is a flattened physical KV-cache slot for the
        corresponding K/V row; padded query/decode positions must contain `-1`.

        Args:
            layer_idx: Transformer layer being written.
            slot_mapping: Physical cache slots shaped like `k_tokens[..., 0, 0]`.
                Decode may pass a one-dimensional `[B]` mapping for `[B, 1, H, D]`.
            k_tokens: New keys shaped `[B, T, Hkv, D]`.
            v_tokens: New values shaped `[B, T, Hkv, D]`.
        """
        flat_slots = slot_mapping.reshape(-1)
        k_flat_all = k_tokens.reshape(-1, k_tokens.shape[-2], k_tokens.shape[-1])
        v_flat_all = v_tokens.reshape(-1, v_tokens.shape[-2], v_tokens.shape[-1])
        # The Triton cache-write kernel masks slot == -1 itself. Trying the full
        # shape first keeps decode CUDA graph writes shape-stable; only the
        # PyTorch fallback compacts invalid padded rows.
        if triton_reshape_and_cache_flash(
            k_flat_all,
            v_flat_all,
            self.k_layers[layer_idx],
            self.v_layers[layer_idx],
            flat_slots,
            self.k_scale,
            self.v_scale,
        ):
            return
        if assume_all_valid:
            slots = flat_slots.to(dtype=torch.long)
            k_flat = k_flat_all
            v_flat = v_flat_all
        else:
            valid = flat_slots >= 0
            slots = flat_slots[valid].to(dtype=torch.long)
            k_flat = k_flat_all[valid]
            v_flat = v_flat_all[valid]
        if slots.numel() == 0:
            return
        k_cache = self.k_layers[layer_idx].view(-1, k_tokens.shape[-2], k_tokens.shape[-1])
        v_cache = self.v_layers[layer_idx].view(-1, v_tokens.shape[-2], v_tokens.shape[-1])
        k_cache.index_copy_(0, slots, k_flat)
        v_cache.index_copy_(0, slots, v_flat)

    def block_tables_tensor(
        self,
        block_id_lists: list[list[int]],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Materialize the padded block table consumed by paged attention.

        This is metadata for the fast paged attention path, not a dense K/V
        fallback. Each row maps a request's logical blocks to physical cache
        blocks.
        """
        device = self.k_layers[0].device
        if not block_id_lists:
            return torch.empty((0, 0), device=device, dtype=torch.long)
        max_blocks = max((self.blocks_needed(seq_len) for seq_len in seq_lens), default=0)
        block_tables = torch.full(
            (len(block_id_lists), max_blocks),
            -1,
            device=device,
            dtype=torch.long,
        )
        for req_idx, (block_ids, seq_len) in enumerate(zip(block_id_lists, seq_lens, strict=True)):
            blocks_needed = self.blocks_needed(seq_len)
            if blocks_needed == 0:
                continue
            block_tables[req_idx, :blocks_needed] = torch.tensor(
                block_ids[:blocks_needed],
                device=device,
                dtype=torch.long,
            )
        return block_tables
