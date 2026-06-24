from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .attention_backends import build_attention_backend
from .config import EngineConfig, ModelConfig
from .kernels import TritonDecodeMetadata, describe_kernel_stack
from .kernels import prepare_decode_inputs, update_decode_state
from .kv_cache import PagedKVCache
from .model import AttentionBatchMetadata, MiniLlamaLM
from common.profiling import SimpleProfiler
from .requests import RequestSpec, RequestState
from .scheduler import ContinuousBatchScheduler, PrefillWorkItem


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax token for each row of logits.

    Args:
        logits: Token scores shaped `[B, vocab_size]`.
    """
    return torch.argmax(logits, dim=-1)


@dataclass
class EngineResult:
    request_id: str
    prompt_tokens: int
    generated_ids: list[int]
    finish_reason: str | None
    prefix_cache_hits: int
    scheduler_steps: int


@dataclass
class DecodeGraphWarmupResult:
    captured: int = 0
    skipped: int = 0


class AsyncTokenCopy:
    """Deferred GPU-to-CPU token copy for CPU-visible request bookkeeping.

    The copy is launched on a CUDA side stream. The CPU can keep scheduling
    more GPU work immediately, and only waits when `tolist()` is called.
    """

    def __init__(
        self,
        cpu_tokens: torch.Tensor,
        event: torch.cuda.Event | None = None,
        gpu_tokens: torch.Tensor | None = None,
    ) -> None:
        self.cpu_tokens = cpu_tokens
        self.event = event
        # Keep the source GPU tensor alive until the side-stream copy finishes.
        # Otherwise PyTorch may free/reuse its storage while the copy stream is
        # still reading from it. This is the same lifetime-management idea vLLM
        # uses for sampled token ids.
        self._gpu_tokens = gpu_tokens
        self._tokens: list[int] | None = None

    def tolist(self) -> list[int]:
        if self._tokens is not None:
            return self._tokens
        if self.event is not None:
            # This is the first point where the CPU must see the copied values.
            # If the D2H copy has not finished yet, this blocks until it has.
            self.event.synchronize()
        self._gpu_tokens = None
        self._tokens = [int(token) for token in self.cpu_tokens.tolist()]
        return self._tokens


class DecodeGpuState:
    """GPU-side decode state, modeled after vLLM's request-state table.

    CPU request objects still own scheduling and final output. This table keeps
    the last sampled token on GPU so the next decode input does not depend on a
    synchronous per-token `.item()` round trip.

    `request_id_to_slot` maps a Python request id to a row in `last_token_ids`.
    `last_token_ids[slot]` is the latest sampled token for that request.
    """

    def __init__(self, *, max_num_reqs: int, device: torch.device, enabled: bool) -> None:
        self.enabled = enabled and device.type == "cuda"
        self.device = device
        self.request_id_to_slot: dict[str, int] = {}
        self.free_slots = list(range(max_num_reqs - 1, -1, -1))
        # One GPU-resident token per active request slot. This is logically 1D:
        #
        #   request_id_to_slot["req-A"] == 0  ->  last_token_ids[0] is req-A's
        #                                        next decode input token.
        #
        # Keeping this on GPU avoids synchronizing each sampled token back to
        # CPU just to feed it into the next decode step.
        self.last_token_ids = torch.zeros(max_num_reqs, device=device, dtype=torch.long)
        self.slot_cpu = torch.empty(
            max_num_reqs,
            dtype=torch.long,
            pin_memory=self.enabled,
        )
        self.slot_workspace = torch.empty(max_num_reqs, device=device, dtype=torch.long)
        self.copy_stream = torch.cuda.Stream(device) if self.enabled else None

    def allocate(self, request: RequestState) -> None:
        if request.request_id in self.request_id_to_slot:
            return
        if not self.free_slots:
            raise RuntimeError("decode GPU state ran out of request slots")
        slot = self.free_slots.pop()
        self.request_id_to_slot[request.request_id] = slot
        if self.enabled:
            self.last_token_ids[slot] = 0

    def release(self, request: RequestState) -> None:
        slot = self.request_id_to_slot.pop(request.request_id, None)
        if slot is not None:
            if self.enabled:
                self.last_token_ids[slot] = 0
            self.free_slots.append(slot)

    def _slot_ids(self, requests: list[RequestState]) -> torch.Tensor:
        for idx, req in enumerate(requests):
            self.slot_cpu[idx] = self.request_id_to_slot[req.request_id]
        out = self.slot_workspace[: len(requests)]
        out.copy_(self.slot_cpu[: len(requests)], non_blocking=self.enabled)
        return out

    def set_last_token_from_cpu(self, request: RequestState, token_id: int) -> None:
        if not self.enabled:
            return
        self.allocate(request)
        slot = self.request_id_to_slot[request.request_id]
        self.last_token_ids[slot] = token_id

    def copy_last_tokens_to(
        self,
        dst: torch.Tensor,
        requests: list[RequestState],
        *,
        pad_token_id: int,
    ) -> bool:
        if not self.enabled:
            return False
        if not requests:
            return True
        slots = self._slot_ids(requests)
        dst[: len(requests)].copy_(self.last_token_ids.index_select(0, slots))
        if dst.shape[0] > len(requests):
            dst[len(requests) :].fill_(pad_token_id)
        return True

    def update_last_tokens(
        self,
        requests: list[RequestState],
        next_tokens: torch.Tensor,
    ) -> None:
        if not self.enabled or not requests:
            return
        slots = self._slot_ids(requests)
        update_decode_state(
            req_slots=slots,
            next_tokens=next_tokens,
            last_token_ids=self.last_token_ids,
            count=len(requests),
        )

    def async_copy_tokens(self, next_tokens: torch.Tensor, count: int) -> AsyncTokenCopy:
        tokens = next_tokens[:count].detach()
        if not self.enabled or self.copy_stream is None:
            return AsyncTokenCopy(tokens.to("cpu"))
        event = torch.cuda.Event()
        main_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(main_stream)
        with torch.cuda.stream(self.copy_stream):
            cpu_tokens = tokens.to("cpu", non_blocking=True)
            event.record(self.copy_stream)
        return AsyncTokenCopy(cpu_tokens, event, gpu_tokens=tokens)



class PrefillWorker:
    def __init__(
        self,
        model: MiniLlamaLM,
        engine_config: EngineConfig,
        kv_cache: PagedKVCache,
        decode_gpu_state: DecodeGpuState,
        profiler: SimpleProfiler,
    ) -> None:
        """Create the worker that processes prompt chunks.

        Args:
            model: The decoder-only model used for both prefill and decode.
            engine_config: Runtime limits and token ids.
            kv_cache: Shared paged KV cache written during prefill.
        """
        self.model = model
        self.engine_config = engine_config
        self.kv_cache = kv_cache
        self.decode_gpu_state = decode_gpu_state
        self.profiler = profiler

    def process(self, work_items: list[PrefillWorkItem]) -> None:
        """Run one prefill batch.

        Args:
            work_items: One entry per request chunk scheduled this step. Each
                item says which request to process and how many prompt tokens
                from that request belong in this chunk. For example, a 100-token
                prompt may arrive here multiple times with chunk lengths 32, 32,
                32, and 4.
        """
        if not work_items:
            return
        with self.profiler.section("prefill.prepare"):
            device = next(self.model.parameters()).device
            max_chunk = max(item.chunk_len for item in work_items)
            # Build a padded `[num_requests, max_chunk]` prompt tensor for this
            # scheduler step. If chunk lengths are `[4, 2]`, row 0 has four real
            # tokens and row 1 has two real tokens plus two pad tokens. The
            # `lengths` list tells attention which cells are valid.
            input_ids = torch.full(
                (len(work_items), max_chunk),
                self.engine_config.pad_token_id,
                device=device,
                dtype=torch.long,
            )
            positions = torch.zeros((len(work_items), max_chunk), device=device, dtype=torch.long)
            lengths: list[int] = []

            for idx, item in enumerate(work_items):
                req = item.request
                start = req.prompt_tokens_computed
                end = start + item.chunk_len
                chunk_ids = req.prompt_ids[start:end]
                # Prefill only writes prompt tokens. Generated-token KV entries
                # do not exist yet for requests in the prefill queue, so express
                # capacity in prompt-token terms instead of the more general
                # cached_seq_len. Example: if 32 prompt tokens are already
                # computed and this chunk has 8 tokens, we need capacity for the
                # first 40 prompt tokens.
                req.block_ids = self.kv_cache.ensure_capacity(req.block_ids, end)
                input_ids[idx, : item.chunk_len] = torch.tensor(chunk_ids, device=device, dtype=torch.long)
                # Positions are absolute within the full prompt, not relative to
                # this chunk. RoPE needs absolute positions so chunked prefill
                # matches a single full-prompt prefill.
                positions[idx, : item.chunk_len] = torch.arange(start, end, device=device, dtype=torch.long)
                lengths.append(item.chunk_len)

        with self.profiler.section("prefill.model"):
            requests = [item.request for item in work_items]
            # Prefill stays eager in this version. Capturing prefill with CUDA
            # graphs was more complex and slower in our experiments because
            # prompt chunk shapes vary and are not graph-friendly in this
            # teaching implementation. Decode graph replay is introduced in v8.
            logits = self.model.prefill_chunk(
                requests=requests,
                input_ids=input_ids,
                positions=positions,
                lengths=lengths,
                kv_cache=self.kv_cache,
            )

        with self.profiler.section("prefill.postprocess"):
            next_tokens = sample_greedy(logits)
            deferred_requests: list[RequestState] = []
            deferred_indices: list[int] = []
            defer_cpu_tokens = (
                self.engine_config.enable_async_output_processing
                and self.decode_gpu_state.enabled
            )
            for idx, item in enumerate(work_items):
                req = item.request
                req.prompt_tokens_computed += item.chunk_len
                # Prefix cache entries are published only for complete prompt
                # blocks. Example with block_size=16: after computing 31 prompt
                # tokens, publish 1 block; after 32 tokens, publish 2 blocks.
                full_blocks_now = req.prompt_tokens_computed // self.engine_config.block_size
                if self.engine_config.enable_prefix_cache:
                    req.prefix_blocks_published, newly_cached_blocks = self.kv_cache.prefix_cache.insert_full_blocks(
                        prompt_ids=req.prompt_ids,
                        block_ids=req.block_ids,
                        published_until_block=req.prefix_blocks_published,
                        full_blocks_available=full_blocks_now,
                    )
                    if newly_cached_blocks:
                        self.kv_cache.retain_blocks(newly_cached_blocks)
                if not req.needs_prefill:
                    # The final prefill logits sample the first generated token.
                    # That token is output, but its K/V is not in the cache yet.
                    # The next decode step feeds this token and writes its K/V.
                    if defer_cpu_tokens:
                        deferred_requests.append(req)
                        deferred_indices.append(idx)
                        continue
                    next_token = int(next_tokens[idx].item())
                    req.add_generated_token(next_token)
                    self.decode_gpu_state.set_last_token_from_cpu(req, next_token)
                    if req.should_stop(
                        self.engine_config.eos_token_id,
                        ignore_eos=self.engine_config.ignore_eos,
                    ):
                        continue
                    req.next_input_token_id = next_token
            if deferred_requests:
                index_tensor = torch.tensor(deferred_indices, device=logits.device, dtype=torch.long)
                deferred_tokens = next_tokens.index_select(0, index_tensor)
                token_copy = self.decode_gpu_state.async_copy_tokens(
                    deferred_tokens,
                    len(deferred_requests),
                )
                self.decode_gpu_state.update_last_tokens(deferred_requests, deferred_tokens)
                for out_idx, req in enumerate(deferred_requests):
                    req.defer_generated_token_copy(token_copy, out_idx)
                    if req.should_stop(
                        self.engine_config.eos_token_id,
                        ignore_eos=True,
                    ):
                        continue
                    # Scheduler only needs a non-None value to know this
                    # request is ready for decode. In async mode the real token
                    # is stored in DecodeGpuState.last_token_ids, so this CPU
                    # field is just a placeholder.
                    req.next_input_token_id = self.engine_config.pad_token_id


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


class DecodeEagerWorkspace:
    """Persistent decode buffers for the non-CUDA-graph eager path.

    vLLM eager mode still avoids rebuilding model-runner tensors from scratch on
    every step. This workspace keeps the hot decode metadata buffers alive and
    mutates them in place before calling the normal model forward.
    """

    def __init__(
        self,
        *,
        model: MiniLlamaLM,
        kv_cache: PagedKVCache,
        engine_config: EngineConfig,
        decode_gpu_state: DecodeGpuState,
        scratch_block_id: int,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.engine_config = engine_config
        self.decode_gpu_state = decode_gpu_state
        self.scratch_block_id = scratch_block_id
        self.max_batch_size = engine_config.max_decode_batch_size
        self.max_blocks = 0
        device = next(model.parameters()).device
        pin_host = device.type == "cuda"

        self.input_ids = torch.empty((self.max_batch_size, 1), device=device, dtype=torch.long)
        self.positions = torch.empty((self.max_batch_size, 1), device=device, dtype=torch.long)
        self.query_lens = torch.ones(self.max_batch_size, device=device, dtype=torch.long)
        self.past_lens = torch.empty(self.max_batch_size, device=device, dtype=torch.long)
        self.key_lens = torch.empty(self.max_batch_size, device=device, dtype=torch.long)
        self.seq_lens_i32 = torch.empty(self.max_batch_size, device=device, dtype=torch.int32)
        self.cu_seqlens_q = torch.arange(self.max_batch_size + 1, device=device, dtype=torch.int32)
        self.slot_mapping = torch.empty(self.max_batch_size, device=device, dtype=torch.long)
        descale_shape = (self.max_batch_size, model.config.num_key_value_heads)
        self.k_descale = torch.ones(descale_shape, device=device, dtype=torch.float32)
        self.v_descale = torch.ones(descale_shape, device=device, dtype=torch.float32)

        self.req_slots_host = torch.empty(self.max_batch_size, dtype=torch.long, pin_memory=pin_host)
        self.cached_seq_lens_host = torch.empty(self.max_batch_size, dtype=torch.long, pin_memory=pin_host)
        self.req_slots = torch.empty(self.max_batch_size, device=device, dtype=torch.long)
        self.cached_seq_lens = torch.empty(self.max_batch_size, device=device, dtype=torch.long)
        self.block_tables_host: torch.Tensor
        self.block_tables: torch.Tensor
        self.block_tables_i32: torch.Tensor
        self._resize_block_table_buffers(1)

    def _resize_block_table_buffers(self, max_blocks: int) -> None:
        """Resize the reusable batched block-table metadata buffers.

        This does not allocate KV cache pages for any request. Each request
        owns its logical-to-physical page map in `req.block_ids`; this workspace
        only packs those per-request lists into rectangular tensors shaped:

            [max_decode_batch_size, max_blocks]

        The attention backend needs that rectangular tensor form so row `r`,
        column `c` can answer: "for request r, which physical KV page stores
        logical page c?"

        Example with block_size=16 and max_blocks=4:

            req A has 20 tokens -> req.block_ids == [7, 12]
            req B has 50 tokens -> req.block_ids == [3, 9, 10, 22]

        The packed batch table is:

            [
              [7, 12, scratch, scratch],
              [3,  9,      10,      22],
            ]

        Resizing here only changes the storage available for that packed table.
        The physical pages 7, 12, 3, 9, 10, and 22 were allocated earlier by
        `kv_cache.ensure_capacity(...)`.
        """
        device = self.input_ids.device
        pin_host = device.type == "cuda"
        self.max_blocks = max_blocks
        self.block_tables_host = torch.empty(
            (self.max_batch_size, max_blocks),
            dtype=torch.long,
            pin_memory=pin_host,
        )
        self.block_tables = torch.empty((self.max_batch_size, max_blocks), device=device, dtype=torch.long)
        self.block_tables_i32 = torch.empty((self.max_batch_size, max_blocks), device=device, dtype=torch.int32)

    def _ensure_block_table_buffers(self, required_blocks: int) -> None:
        """Grow the rectangular block-table buffers if this batch needs more columns."""
        if required_blocks <= self.max_blocks:
            return
        # Grow geometrically so a request crossing block boundaries does not
        # reallocate these metadata buffers every few decode steps.
        self._resize_block_table_buffers(_next_power_of_two(required_blocks))

    def prepare(
        self,
        requests: list[RequestState],
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionBatchMetadata]:
        batch_size = len(requests)
        if batch_size > self.max_batch_size:
            raise ValueError("decode batch is larger than max_decode_batch_size")
        # Decode always appends exactly one new token per request. `past_len` is
        # what is already in KV; `key_len` is what attention sees after adding
        # this step's token.
        #
        # Example: a request with 12 cached tokens decodes token 13:
        #
        #   past_len = 12
        #   key_len  = 13
        past_lengths = [req.cached_seq_len for req in requests]
        key_lengths = [past_len + 1 for past_len in past_lengths]
        max_key_len = max(key_lengths, default=1)
        # This is the width needed for the *batch metadata tensor*, not a KV
        # allocation decision. Requests may have different lengths, but the
        # attention backend wants one rectangular table:
        #
        #   block_tables[request_row, logical_block] -> physical_block
        #
        # Example with block_size=16:
        #
        #   key_lengths = [13, 20, 50]
        #   blocks      = [ 1,  2,  4]
        #
        # The batch table therefore needs 4 columns, even though the first row
        # only uses one of them. Unused columns are filled with scratch_block_id.
        required_blocks = max((self.kv_cache.blocks_needed(key_len) for key_len in key_lengths), default=1)
        self._ensure_block_table_buffers(required_blocks)

        # Pack each request's Python list of physical KV page ids into a pinned
        # host staging tensor. Rows shorter than `required_blocks` are padded with
        # scratch_block_id so attention kernels never see invalid page ids.
        self.block_tables_host[:batch_size].fill_(self.scratch_block_id)
        for row_idx, (req, past_len, key_len) in enumerate(
            zip(requests, past_lengths, key_lengths, strict=True)
        ):
            blocks_needed = self.kv_cache.blocks_needed(key_len)
            self.req_slots_host[row_idx] = self.decode_gpu_state.request_id_to_slot[req.request_id]
            self.cached_seq_lens_host[row_idx] = past_len
            # `blocks_needed` is the number of logical KV pages covered by this
            # request's current attention key length. Because this toy engine does
            # not use sliding-window attention, decode attends to every previous
            # token plus the new token, so these are all logical blocks visible to
            # attention for this request.
            #
            # Example with block_size=16:
            #
            #   key_len        == 20
            #   blocks_needed  == 2
            #   req.block_ids  == [7, 12]
            #   copied row     == [7, 12, scratch, scratch]  # if max_blocks == 4
            #
            # In normal decode, DecodeWorker.process() called ensure_capacity()
            # just before this, so `req.block_ids` should already contain at
            # least these blocks. The slice keeps the copied metadata aligned
            # with `key_len`: attention consumes exactly `blocks_needed` entries,
            # and the remaining rectangular-table columns stay padded.
            for block_idx, block_id in enumerate(req.block_ids[:blocks_needed]):
                self.block_tables_host[row_idx, block_idx] = block_id

        # Move compact metadata to GPU once per decode step. A Triton helper then
        # derives input_ids, positions, slot_mapping, and the int32 block table
        # form used by paged attention metadata.
        self.req_slots[:batch_size].copy_(self.req_slots_host[:batch_size], non_blocking=True)
        self.cached_seq_lens[:batch_size].copy_(self.cached_seq_lens_host[:batch_size], non_blocking=True)
        self.block_tables[:batch_size].copy_(self.block_tables_host[:batch_size], non_blocking=True)

        input_ids = self.input_ids[:batch_size]
        positions = self.positions[:batch_size]
        past_lens = self.past_lens[:batch_size]
        key_lens = self.key_lens[:batch_size]
        seq_lens_i32 = self.seq_lens_i32[:batch_size]
        slot_mapping = self.slot_mapping[:batch_size]
        block_tables = self.block_tables[:batch_size]
        block_tables_i32 = self.block_tables_i32[:batch_size]

        prepare_decode_inputs(
            req_slots=self.req_slots[:batch_size],
            cached_seq_lens=self.cached_seq_lens[:batch_size],
            block_tables=block_tables,
            last_token_ids=self.decode_gpu_state.last_token_ids,
            input_ids=input_ids,
            positions=positions,
            past_lens=past_lens,
            key_lens=key_lens,
            seq_lens_i32=seq_lens_i32,
            slot_mapping=slot_mapping,
            block_tables_i32=block_tables_i32,
            actual_count=batch_size,
            block_size=self.kv_cache.block_size,
            pad_token_id=self.engine_config.pad_token_id,
            scratch_block_id=self.scratch_block_id,
        )

        cu_seqlens_q = self.cu_seqlens_q[: batch_size + 1]
        metadata = AttentionBatchMetadata(
            query_lens=self.query_lens[:batch_size],
            past_lens=past_lens,
            key_lens=key_lens,
            key_lens_i32=seq_lens_i32,
            cu_seqlens_q=cu_seqlens_q,
            block_tables=block_tables,
            block_tables_i32=block_tables_i32,
            total_queries=batch_size,
            max_key_len=max_key_len,
            triton_decode_metadata=TritonDecodeMetadata(
                cu_seqlens_q=cu_seqlens_q,
                seq_lens=seq_lens_i32,
                block_table=block_tables_i32,
                max_seqlen_k=max_key_len,
                k_descale=self.k_descale[:batch_size],
                v_descale=self.v_descale[:batch_size],
            ),
            decode_slot_mapping=slot_mapping,
            query_all_valid=True,
        )
        return input_ids, positions, metadata


class DecodeWorker:
    def __init__(
        self,
        model: MiniLlamaLM,
        engine_config: EngineConfig,
        kv_cache: PagedKVCache,
        decode_gpu_state: DecodeGpuState,
        profiler: SimpleProfiler,
    ) -> None:
        """Create the worker that processes one-token decode steps."""
        self.model = model
        self.engine_config = engine_config
        self.kv_cache = kv_cache
        self.decode_gpu_state = decode_gpu_state
        self.profiler = profiler
        device = next(self.model.parameters()).device
        self.input_ids_workspace = torch.empty(
            (engine_config.max_decode_batch_size, 1),
            device=device,
            dtype=torch.long,
        )
        self.positions_workspace = torch.empty(
            (engine_config.max_decode_batch_size, 1),
            device=device,
            dtype=torch.long,
        )
        # Decode CUDA graph replay is introduced in v8. Earlier versions always
        # run eager decode, so there are no graph buckets or graph scratch pages
        # to reason about in this worker.
        self.eager_decode_workspace: DecodeEagerWorkspace | None = None
        self.eager_decode_workspace_enabled = (
            engine_config.enable_eager_decode_workspace
            and engine_config.enable_triton_decode_metadata_kernel
            and self.decode_gpu_state.enabled
            and engine_config.device.startswith("cuda")
            and engine_config.attention_backend == "flash_attn_paged"
        )
        if self.eager_decode_workspace_enabled:
            self.eager_decode_workspace = DecodeEagerWorkspace(
                model=self.model,
                kv_cache=self.kv_cache,
                engine_config=self.engine_config,
                decode_gpu_state=self.decode_gpu_state,
                scratch_block_id=self.kv_cache.allocate_block(),
            )

    def process(self, requests: list[RequestState]) -> None:
        """Run one decode batch.

        Args:
            requests: Requests that are ready to decode one more token. Each
                request contributes exactly one query token here.
        """
        if not requests:
            return
        with self.profiler.section("decode.prepare"):
            batch_size = len(requests)
            if batch_size > self.engine_config.max_decode_batch_size:
                raise ValueError("decode batch is larger than max_decode_batch_size")
            for req in requests:
                # This is the actual per-request KV page allocation step.
                #
                # `req.block_ids` is the request's logical-to-physical page map.
                # If appending one decode token crosses a block boundary,
                # ensure_capacity allocates another physical KV page and appends
                # it to this list.
                #
                # Example with block_size=16:
                #
                #   cached_seq_len = 15, append token 16 -> still one block
                #   req.block_ids  = [7]
                #
                #   cached_seq_len = 16, append token 17 -> need two blocks
                #   req.block_ids  = [7, 12]  # 12 was newly allocated
                #
                # Later, DecodeEagerWorkspace.prepare() packs these lists from
                # all active requests into a rectangular `block_tables` tensor
                # for the attention kernel. That packing/resizing is metadata
                # management, not KV page allocation.
                req.block_ids = self.kv_cache.ensure_capacity(req.block_ids, req.cached_seq_len + 1)
            eager_metadata = None
            if self.eager_decode_workspace is not None:
                # Fast eager path: reuse persistent metadata tensors and fill
                # them with a Triton prep kernel. This is the normal optimized
                # non-graph decode path.
                input_ids, positions, eager_metadata = self.eager_decode_workspace.prepare(requests)
            else:
                # Simple fallback path: materialize the one-token input ids and
                # positions directly. This is easier to understand but rebuilds
                # more tensors per step.
                input_ids = self.input_ids_workspace[:batch_size]
                positions = self.positions_workspace[:batch_size]
                if not self.decode_gpu_state.copy_last_tokens_to(
                    input_ids[:, 0],
                    requests,
                    pad_token_id=self.engine_config.pad_token_id,
                ):
                    input_ids[:, 0] = torch.tensor(
                        [req.next_input_token_id for req in requests],
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                positions[:, 0] = torch.tensor(
                    [req.cached_seq_len for req in requests],
                    device=positions.device,
                    dtype=torch.long,
                )

        with self.profiler.section("decode.model"):
            if eager_metadata is not None:
                logits = self.model.decode_tokens_prebuilt(
                    input_ids=input_ids,
                    positions=positions,
                    metadata=eager_metadata,
                    kv_cache=self.kv_cache,
                )
            else:
                logits = self.model.decode_tokens(
                    requests=requests,
                    input_ids=input_ids,
                    positions=positions,
                    kv_cache=self.kv_cache,
                )

        with self.profiler.section("decode.postprocess"):
            next_tokens = sample_greedy(logits)
            token_copy = self.decode_gpu_state.async_copy_tokens(next_tokens, len(requests))
            self.decode_gpu_state.update_last_tokens(requests, next_tokens)
            if self.engine_config.enable_async_output_processing and self.decode_gpu_state.enabled:
                # Async path: the next decode input stays on GPU. CPU-visible
                # output tokens are resolved later by SimpleVLLMEngine.run().
                for idx, req in enumerate(requests):
                    req.generated_tokens_in_cache += 1
                    req.defer_generated_token_copy(token_copy, idx)
                    if req.should_stop(
                        self.engine_config.eos_token_id,
                        ignore_eos=True,
                    ):
                        continue
                    # Same placeholder convention as prefill: with async output
                    # processing, the actual next token stays in
                    # DecodeGpuState.last_token_ids on GPU.
                    req.next_input_token_id = self.engine_config.pad_token_id
            else:
                # Blocking path: force sampled ids to CPU now. This is simpler
                # and useful for debugging, but it puts D2H synchronization on
                # the per-token critical path.
                sampled_tokens = token_copy.tolist()
                for idx, req in enumerate(requests):
                    req.generated_tokens_in_cache += 1
                    sampled = sampled_tokens[idx]
                    req.add_generated_token(sampled)
                    if req.should_stop(
                        self.engine_config.eos_token_id,
                        ignore_eos=self.engine_config.ignore_eos,
                    ):
                        continue
                    req.next_input_token_id = sampled


class SimpleVLLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        model: MiniLlamaLM | None = None,
    ) -> None:
        """Construct the teaching engine and all of its subsystems.

        Args:
            model_config: Architecture served by this engine.
            engine_config: Runtime settings including block size, batch limits,
                device, and dtype.
            model: Optional preloaded model instance. `v3` uses this to inject a
                pretrained checkpoint-loaded model instead of a random one.
        """
        engine_config.validate(model_config)
        if engine_config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device='cuda', but torch.cuda.is_available() is False. "
                "No CUDA GPU is visible to this process."
        )
        self.model_config = model_config
        self.engine_config = engine_config
        self.model = model or MiniLlamaLM(model_config)
        self.model = self.model.to(device=engine_config.device, dtype=engine_config.dtype)
        self.model.eval()
        if engine_config.enable_torch_compile_model_body:
            self.model.enable_torch_compile(
                fullgraph=engine_config.torch_compile_fullgraph,
                dynamic=engine_config.torch_compile_dynamic,
                scope=engine_config.torch_compile_scope,
            )
        self.profiler = SimpleProfiler(engine_config.device, enabled=engine_config.enable_timing)
        self.model.profiler = self.profiler
        self.model.attention_backend = build_attention_backend(
            engine_config.attention_backend,
            num_attention_heads=model_config.num_attention_heads,
            profiler=self.profiler,
        )
        self.kv_cache = PagedKVCache(model_config, engine_config)
        # A simple vLLM-style GPU request-state table. We size it generously
        # from cache capacity because the teaching engine has no separate
        # `max_num_reqs` scheduler knob.
        self.decode_gpu_state = DecodeGpuState(
            max_num_reqs=max(1, engine_config.num_blocks),
            device=torch.device(engine_config.device),
            enabled=engine_config.enable_gpu_decode_state,
        )
        self.scheduler = ContinuousBatchScheduler(engine_config)
        self.prefill_worker = PrefillWorker(
            self.model,
            engine_config,
            self.kv_cache,
            self.decode_gpu_state,
            self.profiler,
        )
        self.decode_worker = DecodeWorker(
            self.model,
            engine_config,
            self.kv_cache,
            self.decode_gpu_state,
            self.profiler,
        )

    def warmup_decode_cuda_graphs(self, specs: list[RequestSpec]) -> DecodeGraphWarmupResult:
        """Return an empty graph warmup result; graph replay is introduced in v8."""
        return DecodeGraphWarmupResult()

    def kernel_summary(self) -> str:
        """Describe which high-level kernel stack this engine uses."""
        return describe_kernel_stack(self.engine_config.device)

    def submit(self, spec: RequestSpec) -> RequestState:
        """Turn a request spec into live engine state and enqueue it.

        Args:
            spec: Request description. If prefix caching is enabled, submission
                also performs the prefix-cache lookup and seeds the request with
                any reusable cached blocks before it first enters the scheduler.
        """
        req = RequestState.from_spec(spec)
        self.decode_gpu_state.allocate(req)
        if self.engine_config.enable_prefix_cache:
            hit = self.kv_cache.prefix_cache.lookup(req.prompt_ids)
            if hit.block_ids:
                req.block_ids = list(hit.block_ids)
                req.prompt_tokens_computed = hit.cached_tokens
                req.prefix_cache_hits = hit.cached_tokens
                self.kv_cache.retain_blocks(req.block_ids)
        self.scheduler.add_request(req)
        return req

    def run(self, specs: list[RequestSpec]) -> list[EngineResult]:
        """Process a workload to completion.

        Args:
            specs: Requests to serve. Each request may arrive at a later
                scheduler step through `arrival_step`, which lets the benchmark
                demonstrate continuous batching instead of a single static batch.
        """
        specs = sorted(specs, key=lambda spec: spec.arrival_step)
        # Live requests keyed by id. This is used at the end to flush any
        # deferred GPU->CPU token copies, build EngineResult objects, and release
        # resources. The scheduler itself only owns a queue of runnable work; it
        # is not the right place to ask "which requests ever existed?"
        active: dict[str, RequestState] = {}
        # `spec_idx` points at the next request spec that has not arrived yet.
        # Specs are sorted by arrival_step, so admitting newly arrived requests
        # is just a linear scan from this index.
        spec_idx = 0
        # Logical engine time. One engine step means one scheduler decision plus
        # the model work selected by that decision. It is not wall-clock time.
        engine_step = 0

        # Main continuous-batching loop.
        #
        # Keep running while either:
        #   1. the scheduler already has runnable requests, or
        #   2. there are future request specs that have not arrived yet.
        #
        # This is the core difference from a static batch. New requests can
        # enter while older requests are already decoding, and each request may
        # bounce through the scheduler many times:
        #
        #   submit -> prefill chunk(s) -> decode token -> decode token -> ...
        #
        # Each iteration admits newly arrived requests, asks the scheduler what
        # work should run now, executes that work, and requeues unfinished
        # requests for a later engine step.
        while self.scheduler.has_work() or spec_idx < len(specs):
            # Admit every request whose arrival time is now visible.
            #
            # `submit()` converts the immutable RequestSpec into mutable
            # RequestState, allocates a GPU decode-state slot, optionally applies
            # prefix-cache hits, and enqueues the request into the scheduler.
            #
            # Multiple requests can share the same arrival_step, so this is a
            # `while`, not an `if`.
            while spec_idx < len(specs) and specs[spec_idx].arrival_step <= engine_step:
                req = self.submit(specs[spec_idx])
                active[req.request_id] = req
                spec_idx += 1

            # If no request is runnable after admitting arrivals, the engine is
            # idle. This can happen when the next workload item arrives in the
            # future, e.g. current engine_step=3 and next arrival_step=10.
            #
            # Instead of spinning through empty steps 4..9, jump directly to the
            # next arrival. This preserves the logical arrival ordering while
            # keeping the teaching benchmark fast and easy to debug.
            if not self.scheduler.has_work():
                if spec_idx < len(specs):
                    engine_step = specs[spec_idx].arrival_step
                    continue
                # Defensive exit: no runnable work and no future arrivals. The
                # outer loop condition should normally be false in this state,
                # but keeping the break makes the termination condition explicit.
                break

            # Ask the scheduler for one engine-step worth of work.
            #
            # A step may contain:
            #   - decode_batch: requests that already have their prompt/KV cache
            #     and need one more generated token.
            #   - prefill_batch: prompt chunks that need to be run to populate
            #     the KV cache before those requests can decode.
            #
            # Decode-first scheduling is typical for serving: once requests are
            # decoding, each one needs a small amount of work every step to keep
            # latency low, while large prompts can be chunked around that.
            step = self.scheduler.schedule()
            if step.decode_batch:
                # The optimized decode path keeps the sampled token on GPU so
                # the next decode step can consume it without a blocking
                # `.item()`/D2H synchronization. The CPU-visible output list is
                # updated later from asynchronous GPU->CPU token copies.
                #
                # Before launching this decode, record how many async copies
                # were already pending for each request. Example:
                #
                #   before step N: pending copies are [token_from_step_N-1]
                #   after  step N: pending copies are [token_from_step_N-1,
                #                                      token_from_step_N]
                #
                # We resolve only the copies that existed before step N. That
                # gives the CPU a one-step-delayed view of generated tokens for
                # EOS/output bookkeeping, while the GPU immediately uses
                # token_from_step_N as the next model input.
                deferred_before_decode = {
                    req.request_id: len(req.deferred_generated_token_copies)
                    for req in step.decode_batch
                }
                # Run one-token decode for every request in the batch. This
                # updates the paged KV cache, samples/records the next token, and
                # updates each RequestState's generated-token bookkeeping.
                self.decode_worker.process(step.decode_batch)
                for req in step.decode_batch:
                    # If EOS handling is enabled, resolve any async token copies
                    # that were already pending before this decode launch. This
                    # gives us a vLLM-style one-step-late CPU stop check without
                    # forcing every decode step to synchronize with the GPU.
                    if (
                        self.engine_config.enable_async_output_processing
                        and not self.engine_config.ignore_eos
                    ):
                        req.resolve_deferred_generated_tokens(
                            eos_token_id=self.engine_config.eos_token_id,
                            ignore_eos=False,
                            max_to_resolve=deferred_before_decode[req.request_id],
                        )
                    # Count how many scheduler/model steps this request used.
                    # This is useful for debugging continuous batching because
                    # requests with longer prompts or generations will revisit
                    # the scheduler more often.
                    req.scheduler_steps += 1
                    if req.finished:
                        # Finished requests are not requeued. Their blocks and
                        # GPU decode-state slot are released after the whole run
                        # when results are assembled below.
                        continue
                    # The request still needs more work. It may need another
                    # decode token, or it may have been marked for additional
                    # prefill in less common paths. Put it back into the
                    # scheduler so the next engine step can consider it again.
                    self.scheduler.add_request(req)

            if step.prefill_batch:
                # Run one chunk of prompt tokens for each request in the prefill
                # batch. The worker builds padded input tensors, computes K/V for
                # the chunk, and appends those K/V vectors into the paged cache.
                #
                # Prefill can be chunked because large prompts should not block
                # all decode work for too long. After each chunk, the request is
                # requeued if more prompt tokens remain.
                self.prefill_worker.process(step.prefill_batch)
                for item in step.prefill_batch:
                    req = item.request
                    req.scheduler_steps += 1
                    if req.finished:
                        # Unusual for prefill, but possible if a request is
                        # completed by policy or has no remaining work.
                        continue
                    # If the prompt is not fully prefetched, the scheduler will
                    # later issue another prefill chunk. If prefill is complete,
                    # the next scheduled work for this request will be decode.
                    self.scheduler.add_request(req)
            # Advance logical time after one scheduler decision. New requests
            # whose arrival_step equals this new value will be admitted at the
            # top of the next loop iteration.
            engine_step += 1

        results: list[EngineResult] = []
        for req in active.values():
            req.flush_deferred_generated_tokens()
            results.append(
                EngineResult(
                    request_id=req.request_id,
                    prompt_tokens=req.prompt_len,
                    generated_ids=list(req.generated_ids),
                    finish_reason=req.finish_reason,
                    prefix_cache_hits=req.prefix_cache_hits,
                    scheduler_steps=req.scheduler_steps,
                )
            )
            self.kv_cache.release_blocks(req.block_ids)
            self.decode_gpu_state.release(req)
        return results


class SerialEngine:
    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig) -> None:
        """Wrap the teaching engine in a one-request-at-a-time baseline."""
        self.inner = SimpleVLLMEngine(model_config, engine_config)
        self.engine_config = self.inner.engine_config

    def run(self, specs: list[RequestSpec]) -> list[EngineResult]:
        """Serve requests serially by invoking the inner engine one spec at a time."""
        results: list[EngineResult] = []
        for spec in sorted(specs, key=lambda spec: spec.arrival_step):
            results.extend(self.inner.run([spec]))
        return results


@dataclass
class TimedRun:
    wall_time_s: float
    results: list[EngineResult]


def timed_run(engine: SimpleVLLMEngine | SerialEngine, specs: list[RequestSpec]) -> TimedRun:
    """Measure end-to-end runtime for one workload execution.

    Args:
        engine: Engine implementation to benchmark.
        specs: Workload to execute.
    """
    if engine.engine_config.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        results = engine.run(specs)
    if engine.engine_config.device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return TimedRun(wall_time_s=t1 - t0, results=results)
