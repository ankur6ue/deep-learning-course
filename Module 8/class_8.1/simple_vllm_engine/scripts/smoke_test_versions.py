#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSIONS = [f"v{idx}" for idx in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run small deterministic correctness smoke tests for v1-v8."
    )
    parser.add_argument(
        "--versions",
        nargs="*",
        default=DEFAULT_VERSIONS,
        help="Version folders to test, e.g. v1 v2 v3. Defaults to v1-v8.",
    )
    parser.add_argument(
        "--child-version",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_parent(versions: list[str]) -> None:
    """Run each version in a fresh Python process.

    Every version folder exposes the same package name, `simple_vllm_engine`.
    A subprocess per version avoids import-cache contamination between folders.
    """
    for version in versions:
        print(f"=== {version} ===", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--child-version", version],
            check=True,
            cwd=ROOT,
        )


def run_child(version: str) -> None:
    version_dir = ROOT / version
    sys.path.insert(0, str(version_dir))
    sys.path.insert(1, str(ROOT))

    from simple_vllm_engine.config import EngineConfig, ModelConfig
    from simple_vllm_engine.engine import SerialEngine, SimpleVLLMEngine, timed_run
    from simple_vllm_engine.requests import RequestSpec
    from common.tokenizer import SimpleTokenizer

    prompts = [
        "alpha beta gamma delta",
        "alpha beta theta",
        "zeta eta theta iota kappa",
        "alpha beta gamma lambda",
    ]
    arrivals = [0, 1, 0, 2]
    tokenizer = SimpleTokenizer.from_texts(prompts)
    specs = [
        RequestSpec(
            request_id=f"req-{idx}",
            prompt_ids=tokenizer.encode(prompt, add_bos=True, add_eos=False),
            max_new_tokens=4,
            arrival_step=arrivals[idx],
        )
        for idx, prompt in enumerate(prompts)
    ]

    # Keep dimensions realistic enough for attention shapes while small enough
    # that every version can run quickly in a subprocess.
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        intermediate_size=1536,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    def engine_config(**overrides) -> EngineConfig:
        kwargs = dict(
            block_size=16,
            num_blocks=128,
            max_batch_tokens=16,
            max_prefill_chunk_tokens=8,
            max_decode_batch_size=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            enable_prefix_cache=False,
            eos_token_id=-999,
            pad_token_id=0,
        )
        if any(field.name == "ignore_eos" for field in dataclasses.fields(EngineConfig)):
            kwargs["ignore_eos"] = True
        kwargs.update(overrides)
        public_fields = {field.name for field in dataclasses.fields(EngineConfig)}
        return EngineConfig(**{key: value for key, value in kwargs.items() if key in public_fields})

    def assert_invariants(results) -> None:
        by_id = {result.request_id: result for result in results}
        assert set(by_id) == {spec.request_id for spec in specs}
        for spec in specs:
            result = by_id[spec.request_id]
            assert len(result.generated_ids) == spec.max_new_tokens, (
                spec.request_id,
                len(result.generated_ids),
                spec.max_new_tokens,
                result.finish_reason,
            )
            assert result.prompt_tokens == len(spec.prompt_ids)
            assert result.scheduler_steps > 0

    def run_engine(config: EngineConfig, seed: int = 1234):
        torch.manual_seed(seed)
        engine = SimpleVLLMEngine(model_config, config)
        run = timed_run(engine, specs)
        assert_invariants(run.results)
        del engine
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return {result.request_id: result.generated_ids for result in run.results}

    def run_serial(config: EngineConfig, seed: int = 1234):
        torch.manual_seed(seed)
        engine = SerialEngine(model_config, config)
        run = timed_run(engine, specs)
        assert_invariants(run.results)
        del engine
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return {result.request_id: result.generated_ids for result in run.results}

    if version in {"v1", "v2"}:
        config = engine_config()
        assert run_serial(config) == run_engine(config)
        print(f"{version}: PASS serial_vs_batched exact_ids")
        return

    if version == "v3":
        ids = run_engine(engine_config())
        print(f"v3: PASS gathered_sdpa_invariants ids={ids}")
        return

    if version in {"v4", "v5", "v6"}:
        ids = run_engine(engine_config())
        print(f"{version}: PASS flash_attn_paged_invariants ids={ids}")
        return

    if version == "v7":
        ids = run_engine(
            engine_config(
                enable_gpu_decode_state=True,
                enable_triton_decode_metadata_kernel=True,
                enable_eager_decode_workspace=True,
                enable_async_output_processing=True,
            )
        )
        print(f"v7: PASS async_gpu_state_invariants ids={ids}")
        return

    if version == "v8":
        ids = run_engine(
            engine_config(
                enable_decode_cuda_graphs=True,
                unsafe_decode_cuda_graphs=False,
                enable_gpu_decode_state=True,
                enable_triton_decode_metadata_kernel=True,
                enable_eager_decode_workspace=True,
                enable_async_output_processing=True,
            )
        )
        print(f"v8: PASS safe_graph_config_invariants ids={ids}")
        return

    raise AssertionError(f"Unhandled version: {version}")


def main() -> None:
    args = parse_args()
    if args.child_version is not None:
        run_child(args.child_version)
    else:
        run_parent(args.versions)


if __name__ == "__main__":
    main()
