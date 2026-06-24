from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class TokenCopy(Protocol):
    """Tiny protocol for async GPU->CPU token copies.

    The engine stores objects that eventually behave like a one-dimensional
    token list. In practice these are small CPU tensors produced by asynchronous
    copies from GPU sampled-token buffers.
    """

    def tolist(self) -> list[int]: ...


@dataclass
class RequestSpec:
    """Immutable user/workload input.

    `arrival_step` is logical scheduler time. Example:

        arrival_step=0  -> request is available immediately
        arrival_step=5  -> request enters after five engine scheduling steps
    """

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    arrival_step: int = 0


@dataclass
class RequestState:
    """Mutable serving state for one request.

    A request moves through three broad states:

    1. Prefill: `prompt_tokens_computed < len(prompt_ids)`.
    2. Decode: prompt K/V is ready and a next-token marker is available.
    3. Finished: EOS or max_new_tokens reached.

    KV-cache ownership lives in `block_ids`. `block_ids[i]` is the physical KV
    page holding logical tokens:

        i * block_size ... (i + 1) * block_size - 1
    """

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    block_ids: list[int] = field(default_factory=list)
    prompt_tokens_computed: int = 0
    generated_ids: list[int] = field(default_factory=list)
    deferred_generated_token_copies: list[tuple[TokenCopy, int]] = field(default_factory=list)
    # Count generated tokens whose K/V has been written by a decode step. The
    # first token sampled from prefill is user-visible output, but its K/V does
    # not exist yet; it becomes decode input, and only that decode pass writes
    # its K/V into the cache.
    generated_tokens_in_cache: int = 0
    next_input_token_id: int | None = None
    finished: bool = False
    finish_reason: str | None = None
    prefix_blocks_published: int = 0
    prefix_cache_hits: int = 0
    scheduler_steps: int = 0

    @classmethod
    def from_spec(cls, spec: RequestSpec) -> "RequestState":
        """Create mutable request state from an immutable request spec.

        Args:
            spec: User-facing request description. It contains the prompt ids,
                `max_new_tokens`, and the arrival step, but none of the runtime
                scheduling or KV-cache fields yet.
        """
        return cls(
            request_id=spec.request_id,
            prompt_ids=list(spec.prompt_ids),
            max_new_tokens=spec.max_new_tokens,
        )

    @property
    def prompt_len(self) -> int:
        """Return the total number of prompt tokens for this request."""
        return len(self.prompt_ids)

    @property
    def cached_seq_len(self) -> int:
        """Return how many tokens already have K/V entries in the cache.

        This includes prompt tokens that have been prefetched plus generated
        tokens that have already gone through decode.
        """
        return self.prompt_tokens_computed + self.generated_tokens_in_cache

    @property
    def needs_prefill(self) -> bool:
        """Return true while any prompt tokens still need K/V computation."""
        return self.prompt_tokens_computed < self.prompt_len

    @property
    def ready_for_decode(self) -> bool:
        """Return true once the next decode token can be scheduled.

        This becomes true after prefill has produced the first sampled token, or
        after a prior decode step sampled another token to feed back in. In the
        async GPU-state path, `next_input_token_id` may be only a non-None
        readiness marker; the real token is stored in `DecodeGpuState` on GPU.
        """
        return (not self.finished) and (self.next_input_token_id is not None)

    @property
    def generated_len(self) -> int:
        """Return how many output tokens have been sampled so far."""
        return len(self.generated_ids) + len(self.deferred_generated_token_copies)

    def add_generated_token(self, token_id: int) -> None:
        """Append one sampled output token already materialized on CPU."""
        self.generated_ids.append(token_id)

    def defer_generated_token_copy(self, token_copy: TokenCopy, index: int) -> None:
        """Record a sampled token that is being copied to CPU asynchronously.

        The next decode step can consume the sampled token from GPU state. The
        CPU only needs the id later for user-visible output and EOS checks, so
        we avoid putting a GPU sync on the critical path.
        """
        self.deferred_generated_token_copies.append((token_copy, index))

    def flush_deferred_generated_tokens(self) -> None:
        """Materialize deferred sampled tokens into `generated_ids`."""
        if not self.deferred_generated_token_copies:
            return
        for token_copy, index in self.deferred_generated_token_copies:
            self.generated_ids.append(int(token_copy.tolist()[index]))
        self.deferred_generated_token_copies.clear()

    def resolve_deferred_generated_tokens(
        self,
        *,
        eos_token_id: int,
        ignore_eos: bool,
        max_to_resolve: int | None = None,
    ) -> None:
        """Materialize older async token copies and apply CPU-side stop checks.

        `max_to_resolve` lets the scheduler check tokens from previous steps
        while leaving the just-produced token deferred. That mirrors vLLM's
        optimistic async-output path: the next token can already be fed from GPU
        state, and CPU stop decisions catch up one step later.
        """
        if not self.deferred_generated_token_copies:
            return
        count = len(self.deferred_generated_token_copies)
        if max_to_resolve is not None:
            count = min(count, max_to_resolve)
        if count <= 0:
            return

        remaining = self.deferred_generated_token_copies[count:]
        pending = self.deferred_generated_token_copies[:count]
        self.deferred_generated_token_copies = remaining
        for token_copy, index in pending:
            token_id = int(token_copy.tolist()[index])
            self.generated_ids.append(token_id)
            if not ignore_eos and token_id == eos_token_id:
                self.mark_finished("eos")
                # Any still-deferred tokens were generated optimistically after
                # EOS and should not appear in the user-visible output.
                self.deferred_generated_token_copies.clear()
                return

    def should_stop(self, eos_token_id: int, *, ignore_eos: bool = False) -> bool:
        """Check whether generation should end after the latest sample.

        Args:
            eos_token_id: Token id that should terminate generation. The other
                stop condition is reaching `max_new_tokens`.
        """
        if self.generated_len == 0:
            return False
        if self.generated_len >= self.max_new_tokens:
            self.mark_finished("length")
            return True
        if ignore_eos:
            return False
        if self.generated_ids[-1] == eos_token_id:
            self.mark_finished("eos")
            return True
        return False

    def mark_finished(self, reason: str) -> None:
        """Force the request into a finished state with a reason string."""
        self.finished = True
        self.finish_reason = reason
        self.next_input_token_id = None
