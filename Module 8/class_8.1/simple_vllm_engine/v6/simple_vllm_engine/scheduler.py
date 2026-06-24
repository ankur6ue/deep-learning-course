from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import EngineConfig
from .requests import RequestState


@dataclass
class PrefillWorkItem:
    """One prompt chunk selected for this scheduler step."""

    request: RequestState
    chunk_len: int


@dataclass
class ScheduleStep:
    """Work chosen for one engine step.

    A step can contain both decode and prefill work:

    - decode_batch contains whole requests, one token each.
    - prefill_batch contains prompt chunks, which may have different lengths.
    """

    decode_batch: list[RequestState]
    prefill_batch: list[PrefillWorkItem]


class ContinuousBatchScheduler:
    """Small decode-first continuous batching scheduler.

    The scheduler owns two queues:

    - waiting_decode: requests with completed prompt KV and a next input token.
    - waiting_prefill: requests that still need prompt tokens written to KV.

    The policy is intentionally simple and vLLM-like:

    1. Serve as many decode requests as allowed by `max_decode_batch_size`.
    2. Spend the remaining token budget on prompt chunks.

    Example with `max_batch_tokens=8`:

        3 decode requests consume 3 token slots.
        5 token slots remain for prefill chunks.
        A 20-token prompt may therefore be scheduled as a 5-token chunk now and
        requeued for later chunks.
    """

    def __init__(self, config: EngineConfig) -> None:
        """Create the decode-first scheduler.

        Args:
            config: Engine limits such as `max_batch_tokens`,
                `max_prefill_chunk_tokens`, and `max_decode_batch_size`.
        """
        self.config = config
        self.waiting_prefill: deque[RequestState] = deque()
        self.waiting_decode: deque[RequestState] = deque()

    def add_request(self, request: RequestState) -> None:
        """Queue a request onto the appropriate work list.

        Args:
            request: Mutable request state. If it still has prompt tokens left,
                it goes to the prefill queue; if prefill is done and it has a
                next input token, it goes to the decode queue.
        """
        # Prefill has priority when the prompt is incomplete: a request cannot
        # decode until all prompt tokens have corresponding K/V cache entries.
        if request.needs_prefill:
            self.waiting_prefill.append(request)
        elif request.ready_for_decode:
            self.waiting_decode.append(request)

    def has_work(self) -> bool:
        """Return true if either the prefill or decode queue is non-empty."""
        return bool(self.waiting_prefill or self.waiting_decode)

    def schedule(self) -> ScheduleStep:
        """Choose one mixed scheduler step.

        The policy is decode-first. Decode requests consume one token each, and
        any remaining token budget is used for prompt chunks. A long prompt may
        be split over multiple calls through `max_prefill_chunk_tokens`.
        """
        decode_batch: list[RequestState] = []
        # Decode is latency-sensitive: each active request usually needs one
        # token every engine step. Keep this queue first and bounded by the max
        # decode batch size.
        while self.waiting_decode and len(decode_batch) < self.config.max_decode_batch_size:
            req = self.waiting_decode.popleft()
            if req.finished or not req.ready_for_decode:
                continue
            decode_batch.append(req)

        remaining_budget = self.config.max_batch_tokens - len(decode_batch)
        prefill_batch: list[PrefillWorkItem] = []
        # Only rotate through the requests that were already waiting at the
        # start of this schedule call. If a request cannot fit because the token
        # budget is exhausted, it is put back for a later engine step instead of
        # spinning forever in this call.
        prefill_rounds = len(self.waiting_prefill)

        while self.waiting_prefill and remaining_budget > 0 and prefill_rounds > 0:
            req = self.waiting_prefill.popleft()
            prefill_rounds -= 1
            if req.finished or not req.needs_prefill:
                continue
            chunk_len = min(
                req.prompt_len - req.prompt_tokens_computed,
                self.config.max_prefill_chunk_tokens,
                remaining_budget,
            )
            if chunk_len <= 0:
                self.waiting_prefill.appendleft(req)
                break
            # `chunk_len` is the number of prompt tokens the prefill worker will
            # process for this request in this step. The request is requeued
            # after the worker if more prompt tokens remain.
            prefill_batch.append(PrefillWorkItem(request=req, chunk_len=chunk_len))
            remaining_budget -= chunk_len

        return ScheduleStep(decode_batch=decode_batch, prefill_batch=prefill_batch)
