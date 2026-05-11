from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import EngineConfig, ModelConfig


@dataclass
class PrefixLookup:
    block_ids: list[int]
    cached_tokens: int


class FullBlockPrefixCache:
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

    def write_tokens(
        self,
        layer_idx: int,
        block_ids: list[int],
        start_token: int,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
    ) -> None:
        """Append a logical token slice into paged KV storage.

        Args:
            layer_idx: Transformer layer being written.
            block_ids: Request-local block table. `block_ids[0]` stores logical
                tokens `0..block_size-1`, `block_ids[1]` stores the next block,
                and so on.
            start_token: Logical token index where this write begins. For
                decode, this is usually the current cached sequence length.
            k_tokens: New keys shaped `[Tnew, Hkv, D]`.
            v_tokens: New values shaped `[Tnew, Hkv, D]`.
        """
        for offset in range(k_tokens.shape[0]):
            token_index = start_token + offset
            logical_block = token_index // self.block_size
            block_offset = token_index % self.block_size
            block_id = block_ids[logical_block]
            self.k_layers[layer_idx][block_id, block_offset].copy_(k_tokens[offset])
            self.v_layers[layer_idx][block_id, block_offset].copy_(v_tokens[offset])

    def gather_tokens(
        self,
        layer_idx: int,
        block_ids: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one request's paged KV cache into a dense token-major view.

        Args:
            layer_idx: Transformer layer to read.
            block_ids: Request-local block table.
            seq_len: Number of valid logical tokens to gather. `block_ids` may
                have spare capacity beyond this length.
        """
        if seq_len == 0:
            device = self.k_layers[layer_idx].device
            h = self.model_config.num_key_value_heads
            d = self.model_config.head_dim
            empty = torch.empty((0, h, d), device=device, dtype=self.k_layers[layer_idx].dtype)
            return empty, empty

        blocks_needed = self.blocks_needed(seq_len)
        block_tensor = torch.tensor(block_ids[:blocks_needed], device=self.k_layers[layer_idx].device, dtype=torch.long)
        k_pages = self.k_layers[layer_idx].index_select(0, block_tensor)
        v_pages = self.v_layers[layer_idx].index_select(0, block_tensor)
        k = k_pages.reshape(blocks_needed * self.block_size, *k_pages.shape[2:])[:seq_len]
        v = v_pages.reshape(blocks_needed * self.block_size, *v_pages.shape[2:])[:seq_len]
        return k, v
