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


@dataclass
class PagedBatchView:
    """Dense teaching view built from a paged KV cache.

    Optimized attention backends consume `block_tables` directly. The reference
    dense paths use this object to show the equivalent padded K/V tensors.
    """

    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


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
        remaining = k_tokens.shape[0]
        src_offset = 0
        token_index = start_token

        # The source chunk is contiguous in logical token order, so we can copy
        # one contiguous slice per touched page rather than writing token by
        # token. This keeps the paged-cache mapping explicit while making the
        # hot path closer to how a real serving engine would batch KV writes.
        while remaining > 0:
            logical_block, block_offset = divmod(token_index, self.block_size)
            tokens_in_block = min(remaining, self.block_size - block_offset)
            block_id = block_ids[logical_block]

            src_slice = slice(src_offset, src_offset + tokens_in_block)
            dst_slice = slice(block_offset, block_offset + tokens_in_block)
            self.k_layers[layer_idx][block_id, dst_slice].copy_(k_tokens[src_slice])
            self.v_layers[layer_idx][block_id, dst_slice].copy_(v_tokens[src_slice])

            remaining -= tokens_in_block
            src_offset += tokens_in_block
            token_index += tokens_in_block

    def write_slot_mapping(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        *,
        assume_all_valid: bool = False,
    ) -> None:
        """Write a padded batch of K/V tokens using physical slot ids.

        Args:
            layer_idx: Transformer layer being written.
            slot_mapping: Physical cache slots shaped like `k_tokens[..., 0, 0]`.
                Invalid padded query positions must contain `-1`.
            k_tokens: New keys shaped `[B, T, Hkv, D]`.
            v_tokens: New values shaped `[B, T, Hkv, D]`.
        """
        flat_slots = slot_mapping.reshape(-1)
        k_flat_all = k_tokens.reshape(-1, k_tokens.shape[-2], k_tokens.shape[-1])
        v_flat_all = v_tokens.reshape(-1, v_tokens.shape[-2], v_tokens.shape[-1])
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
        if triton_reshape_and_cache_flash(
            k_flat,
            v_flat,
            self.k_layers[layer_idx],
            self.v_layers[layer_idx],
            slots,
            self.k_scale,
            self.v_scale,
        ):
            return
        k_cache = self.k_layers[layer_idx].view(-1, k_tokens.shape[-2], k_tokens.shape[-1])
        v_cache = self.v_layers[layer_idx].view(-1, v_tokens.shape[-2], v_tokens.shape[-1])
        k_cache.index_copy_(0, slots, k_flat)
        v_cache.index_copy_(0, slots, v_flat)

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

    def block_tables_tensor(
        self,
        block_id_lists: list[list[int]],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Materialize a padded batched block table.

        Args:
            block_id_lists: One logical block table per request.
            seq_lens: Valid logical token lengths for those requests. This is
                used to decide how many entries in each block table row are
                meaningful.
        """
        device = self.k_layers[0].device
        if not block_id_lists:
            return torch.empty((0, 0), device=device, dtype=torch.long)
        # Pad to the longest logical sequence in the batch. Each row is one
        # request's logical block table.
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

    def gather_batch(
        self,
        layer_idx: int,
        block_id_lists: list[list[int]],
        seq_lens: list[int],
    ) -> PagedBatchView:
        """Gather multiple requests' paged KV into one padded dense batch.

        Args:
            layer_idx: Transformer layer to read.
            block_id_lists: One block table per request.
            seq_lens: Valid cached-token length per request. Example: if three
                requests have cached prefix lengths `[5, 12, 0]`, the output is
                padded to length 12 and only those prefix lengths are valid.
        """
        device = self.k_layers[layer_idx].device
        dtype = self.k_layers[layer_idx].dtype
        seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.long)
        if not block_id_lists:
            h = self.model_config.num_key_value_heads
            d = self.model_config.head_dim
            empty = torch.empty((0, 0, h, d), device=device, dtype=dtype)
            return PagedBatchView(
                block_tables=torch.empty((0, 0), device=device, dtype=torch.long),
                seq_lens=seq_lens_tensor,
                k=empty,
                v=empty,
            )

        max_seq_len = max(seq_lens, default=0)
        h = self.model_config.num_key_value_heads
        d = self.model_config.head_dim
        # Gather the paged cache into one padded dense view for the whole batch.
        # A production paged-attention kernel would usually consume the block
        # table directly instead of materializing this dense tensor.
        k_batch = torch.zeros((len(block_id_lists), max_seq_len, h, d), device=device, dtype=dtype)
        v_batch = torch.zeros((len(block_id_lists), max_seq_len, h, d), device=device, dtype=dtype)
        block_tables = self.block_tables_tensor(block_id_lists, seq_lens)

        for req_idx, seq_len in enumerate(seq_lens):
            if seq_len == 0:
                continue
            blocks_needed = self.blocks_needed(seq_len)
            block_ids = block_tables[req_idx, :blocks_needed]
            # `block_ids` may be non-contiguous physical pages. The block table
            # is the indirection that maps this request's logical sequence onto
            # those scattered cache pages.
            k_pages = self.k_layers[layer_idx].index_select(0, block_ids)
            v_pages = self.v_layers[layer_idx].index_select(0, block_ids)
            k_batch[req_idx, :seq_len].copy_(
                k_pages.reshape(blocks_needed * self.block_size, h, d)[:seq_len]
            )
            v_batch[req_idx, :seq_len].copy_(
                v_pages.reshape(blocks_needed * self.block_size, h, d)[:seq_len]
            )

        return PagedBatchView(
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            k=k_batch,
            v=v_batch,
        )

    def build_full_kv_batch(
        self,
        layer_idx: int,
        block_id_lists: list[list[int]],
        past_lens: list[int],
        query_lens: list[int],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize `[past | current chunk | pad]` directly in KV-head space.

        This avoids building a separate padded past tensor and then copying it
        again into a second `[past | current chunk]` tensor.
        """
        device = self.k_layers[layer_idx].device
        dtype = self.k_layers[layer_idx].dtype
        batch_size = len(block_id_lists)
        max_key_len = max((past + query for past, query in zip(past_lens, query_lens, strict=True)), default=0)
        h = self.model_config.num_key_value_heads
        d = self.model_config.head_dim
        k_full = torch.zeros((batch_size, max_key_len, h, d), device=device, dtype=dtype)
        v_full = torch.zeros((batch_size, max_key_len, h, d), device=device, dtype=dtype)

        for req_idx, (block_ids, past_len, query_len) in enumerate(
            zip(block_id_lists, past_lens, query_lens, strict=True)
        ):
            if past_len > 0:
                blocks_needed = self.blocks_needed(past_len)
                block_tensor = torch.tensor(block_ids[:blocks_needed], device=device, dtype=torch.long)
                k_pages = self.k_layers[layer_idx].index_select(0, block_tensor)
                v_pages = self.v_layers[layer_idx].index_select(0, block_tensor)
                k_full[req_idx, :past_len].copy_(
                    k_pages.reshape(blocks_needed * self.block_size, h, d)[:past_len]
                )
                v_full[req_idx, :past_len].copy_(
                    v_pages.reshape(blocks_needed * self.block_size, h, d)[:past_len]
                )
            if query_len > 0:
                k_full[req_idx, past_len : past_len + query_len].copy_(k_new[req_idx, :query_len])
                v_full[req_idx, past_len : past_len + query_len].copy_(v_new[req_idx, :query_len])
        return k_full, v_full

    def write_batch(
        self,
        layer_idx: int,
        block_id_lists: list[list[int]],
        start_tokens: list[int],
        valid_lengths: list[int],
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
    ) -> None:
        """Write a batched chunk into paged KV, request by request.

        Args:
            layer_idx: Transformer layer being updated.
            block_id_lists: One block table per request.
            start_tokens: Logical starting token index for each request's write.
                During decode this is usually the cached sequence length; during
                chunked prefill it is the prompt tokens already computed.
            valid_lengths: Number of valid tokens from each row of `k_tokens`
                and `v_tokens`. Rows may be padded to a common length.
            k_tokens: Batched keys `[B, Tpad, Hkv, D]`.
            v_tokens: Batched values `[B, Tpad, Hkv, D]`.
        """
        for req_idx, (block_ids, start_token, valid_len) in enumerate(
            zip(block_id_lists, start_tokens, valid_lengths, strict=True)
        ):
            if valid_len == 0:
                continue
            # `start_token` says where this chunk begins in the logical
            # sequence, so writes land in the correct page and page offset.
            self.write_tokens(
                layer_idx=layer_idx,
                block_ids=block_ids,
                start_token=start_token,
                k_tokens=k_tokens[req_idx, :valid_len],
                v_tokens=v_tokens[req_idx, :valid_len],
            )

    def write_decode_slots(
        self,
        layer_idx: int,
        slot_mapping: torch.Tensor,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
    ) -> None:
        """Write one decode token per batch row using flattened KV slots.

        Args:
            layer_idx: Transformer layer being updated.
            slot_mapping: Flattened physical slots shaped `[B]`, where each slot
                is `physical_block_id * block_size + block_offset`. Padded CUDA
                graph rows use `-1`, matching vLLM's PAD_SLOT_ID convention, and
                must not write to the cache.
            k_tokens: Decode keys shaped `[B, 1, Hkv, D]`.
            v_tokens: Decode values shaped `[B, 1, Hkv, D]`.
        """
        if triton_reshape_and_cache_flash(
            k_tokens[:, 0],
            v_tokens[:, 0],
            self.k_layers[layer_idx],
            self.v_layers[layer_idx],
            slot_mapping,
            self.k_scale,
            self.v_scale,
        ):
            return
        valid = slot_mapping >= 0
        if not torch.all(valid):
            if not torch.any(valid):
                return
            slot_mapping = slot_mapping[valid]
            k_tokens = k_tokens[valid]
            v_tokens = v_tokens[valid]
        flat_k = self.k_layers[layer_idx].view(-1, self.model_config.num_key_value_heads, self.model_config.head_dim)
        flat_v = self.v_layers[layer_idx].view(-1, self.model_config.num_key_value_heads, self.model_config.head_dim)
        flat_k.index_copy_(0, slot_mapping, k_tokens[:, 0])
        flat_v.index_copy_(0, slot_mapping, v_tokens[:, 0])
