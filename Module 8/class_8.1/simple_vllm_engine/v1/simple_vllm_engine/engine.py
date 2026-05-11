from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import EngineConfig, ModelConfig
from .kernels import describe_kernel_stack
from .kv_cache import PagedKVCache
from .model import MiniLlamaLM
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


class PrefillWorker:
    def __init__(
        self,
        model: MiniLlamaLM,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        kv_cache: PagedKVCache,
    ) -> None:
        """Create the worker that processes prompt chunks.

        Args:
            model: The decoder-only model used for both prefill and decode.
            model_config: Architecture details such as number of heads and
                layers.
            engine_config: Runtime limits and token ids.
            kv_cache: Shared paged KV cache written during prefill.
        """
        self.model = model
        self.model_config = model_config
        self.engine_config = engine_config
        self.kv_cache = kv_cache

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
        device = next(self.model.parameters()).device
        max_chunk = max(item.chunk_len for item in work_items)
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
            req.block_ids = self.kv_cache.ensure_capacity(req.block_ids, req.cached_seq_len + item.chunk_len)
            input_ids[idx, : item.chunk_len] = torch.tensor(chunk_ids, device=device, dtype=torch.long)
            positions[idx, : item.chunk_len] = torch.arange(start, end, device=device, dtype=torch.long)
            lengths.append(item.chunk_len)

        logits = self.model.prefill_chunk(
            requests=[item.request for item in work_items],
            input_ids=input_ids,
            positions=positions,
            lengths=lengths,
            kv_cache=self.kv_cache,
        )

        for idx, item in enumerate(work_items):
            req = item.request
            req.prompt_tokens_computed += item.chunk_len
            full_prompt_blocks = req.prompt_len // self.engine_config.block_size
            full_blocks_now = min(req.prompt_tokens_computed // self.engine_config.block_size, full_prompt_blocks)
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
                next_token = int(sample_greedy(logits[idx : idx + 1])[0].item())
                req.add_generated_token(next_token)
                if req.should_stop(self.engine_config.eos_token_id):
                    continue
                req.next_input_token_id = next_token


class DecodeWorker:
    def __init__(
        self,
        model: MiniLlamaLM,
        engine_config: EngineConfig,
        kv_cache: PagedKVCache,
    ) -> None:
        """Create the worker that processes one-token decode steps."""
        self.model = model
        self.engine_config = engine_config
        self.kv_cache = kv_cache

    def process(self, requests: list[RequestState]) -> None:
        """Run one decode batch.

        Args:
            requests: Requests that are ready to decode one more token. Each
                request contributes exactly one query token here.
        """
        if not requests:
            return
        device = next(self.model.parameters()).device
        input_ids = torch.tensor(
            [[req.next_input_token_id] for req in requests],
            device=device,
            dtype=torch.long,
        )
        positions = torch.tensor(
            [[req.cached_seq_len] for req in requests],
            device=device,
            dtype=torch.long,
        )
        for req in requests:
            req.block_ids = self.kv_cache.ensure_capacity(req.block_ids, req.cached_seq_len + 1)

        logits = self.model.decode_tokens(
            requests=requests,
            input_ids=input_ids,
            positions=positions,
            kv_cache=self.kv_cache,
        )

        next_tokens = sample_greedy(logits)
        for idx, req in enumerate(requests):
            req.generated_tokens_in_cache += 1
            sampled = int(next_tokens[idx].item())
            req.add_generated_token(sampled)
            if req.should_stop(self.engine_config.eos_token_id):
                continue
            req.next_input_token_id = sampled


class SimpleVLLMEngine:
    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig) -> None:
        """Construct the teaching engine and all of its subsystems.

        Args:
            model_config: Architecture served by this engine.
            engine_config: Runtime settings including block size, batch limits,
                device, and dtype.
        """
        engine_config.validate(model_config)
        if engine_config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device='cuda', but torch.cuda.is_available() is False. "
                "No CUDA GPU is visible to this process."
            )
        self.model_config = model_config
        self.engine_config = engine_config
        self.model = MiniLlamaLM(model_config).to(device=engine_config.device, dtype=engine_config.dtype)
        self.model.eval()
        self.kv_cache = PagedKVCache(model_config, engine_config)
        self.scheduler = ContinuousBatchScheduler(engine_config)
        self.prefill_worker = PrefillWorker(self.model, model_config, engine_config, self.kv_cache)
        self.decode_worker = DecodeWorker(self.model, engine_config, self.kv_cache)

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
        requests: list[RequestState] = []
        active: dict[str, RequestState] = {}
        spec_idx = 0
        engine_step = 0

        while self.scheduler.has_work() or spec_idx < len(specs):
            while spec_idx < len(specs) and specs[spec_idx].arrival_step <= engine_step:
                req = self.submit(specs[spec_idx])
                requests.append(req)
                active[req.request_id] = req
                spec_idx += 1

            if not self.scheduler.has_work():
                if spec_idx < len(specs):
                    engine_step = specs[spec_idx].arrival_step
                    continue
                break

            step = self.scheduler.schedule()
            if step.decode_batch:
                self.decode_worker.process(step.decode_batch)
                for req in step.decode_batch:
                    req.scheduler_steps += 1
                    if req.finished:
                        continue
                    self.scheduler.add_request(req)

            if step.prefill_batch:
                self.prefill_worker.process(step.prefill_batch)
                for item in step.prefill_batch:
                    req = item.request
                    req.scheduler_steps += 1
                    if req.finished:
                        continue
                    self.scheduler.add_request(req)
            engine_step += 1

        results: list[EngineResult] = []
        for req in active.values():
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
    with torch.no_grad():
        results = engine.run(specs)
    if engine.engine_config.device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return TimedRun(wall_time_s=t1 - t0, results=results)
