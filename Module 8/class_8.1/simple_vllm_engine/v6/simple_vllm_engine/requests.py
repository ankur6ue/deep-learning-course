from __future__ import annotations

from dataclasses import dataclass, field


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
        after a prior decode step sampled another token to feed back in.
        """
        return (not self.finished) and (self.next_input_token_id is not None)

    @property
    def generated_len(self) -> int:
        """Return how many output tokens have been sampled so far."""
        return len(self.generated_ids)

    def add_generated_token(self, token_id: int) -> None:
        """Append one sampled output token already materialized on CPU."""
        self.generated_ids.append(token_id)

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
