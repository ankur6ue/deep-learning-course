from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestSpec:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    arrival_step: int = 0


@dataclass
class RequestState:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    block_ids: list[int] = field(default_factory=list)
    prompt_tokens_computed: int = 0
    generated_ids: list[int] = field(default_factory=list)
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
        tokens that have already been decoded.
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
        """Append one sampled output token to the request history."""
        self.generated_ids.append(token_id)

    def should_stop(self, eos_token_id: int) -> bool:
        """Check whether generation should end after the latest sample.

        Args:
            eos_token_id: Token id that should terminate generation. The other
                stop condition is reaching `max_new_tokens`.
        """
        if not self.generated_ids:
            return False
        if self.generated_len >= self.max_new_tokens:
            self.finished = True
            self.finish_reason = "length"
            self.next_input_token_id = None
            return True
        if self.generated_ids[-1] == eos_token_id:
            self.finished = True
            self.finish_reason = "eos"
            self.next_input_token_id = None
            return True
        return False

    def mark_finished(self, reason: str) -> None:
        """Force the request into a finished state with a reason string."""
        self.finished = True
        self.finish_reason = reason
        self.next_input_token_id = None
