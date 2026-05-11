from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import EngineConfig
from .requests import RequestState


@dataclass
class PrefillWorkItem:
    request: RequestState
    chunk_len: int


@dataclass
class ScheduleStep:
    decode_batch: list[RequestState]
    prefill_batch: list[PrefillWorkItem]


class ContinuousBatchScheduler:
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
        while self.waiting_decode and len(decode_batch) < self.config.max_decode_batch_size:
            req = self.waiting_decode.popleft()
            if req.finished or not req.ready_for_decode:
                continue
            decode_batch.append(req)

        remaining_budget = self.config.max_batch_tokens - len(decode_batch)
        prefill_batch: list[PrefillWorkItem] = []
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
            prefill_batch.append(PrefillWorkItem(request=req, chunk_len=chunk_len))
            remaining_budget -= chunk_len

        return ScheduleStep(decode_batch=decode_batch, prefill_batch=prefill_batch)
