from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import EngineConfig, ModelConfig
from .engine import SimpleVLLMEngine
from .model import MiniLlamaLM
from .tokenizer import HFTokenizer


def load_model_config_from_pretrained(model_path: str) -> ModelConfig:
    """Load a Hugging Face config and convert it into the teaching model config.

    Args:
        model_path: Local model directory containing `config.json`.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path)
    architectures = tuple(getattr(hf_config, "architectures", []) or ())
    if architectures and "LlamaForCausalLM" not in architectures:
        raise ValueError(f"v3 currently supports LlamaForCausalLM checkpoints, got {architectures}")

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None and isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta")

    return ModelConfig(
        vocab_size=int(hf_config.vocab_size),
        hidden_size=int(hf_config.hidden_size),
        intermediate_size=int(hf_config.intermediate_size),
        num_layers=int(hf_config.num_hidden_layers),
        num_attention_heads=int(hf_config.num_attention_heads),
        num_key_value_heads=int(getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)),
        max_position_embeddings=int(hf_config.max_position_embeddings),
        rope_theta=float(rope_theta if rope_theta is not None else 10000.0),
        rope_scaling=rope_scaling,
        rms_norm_eps=float(hf_config.rms_norm_eps),
    )


def _hf_to_engine_key_map(num_layers: int) -> dict[str, str]:
    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        dst = f"layers.{layer_idx}"
        mapping.update(
            {
                f"{prefix}.input_layernorm.weight": f"{dst}.input_norm.weight",
                f"{prefix}.post_attention_layernorm.weight": f"{dst}.post_norm.weight",
                f"{prefix}.self_attn.q_proj.weight": f"{dst}.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight": f"{dst}.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight": f"{dst}.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight": f"{dst}.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight": f"{dst}.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight": f"{dst}.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight": f"{dst}.down_proj.weight",
            }
        )
    return mapping


def load_pretrained_weights(model: MiniLlamaLM, model_path: str, dtype: torch.dtype | None = None) -> None:
    """Load a local HF Llama checkpoint into the teaching model.

    Args:
        model: Teaching-model instance whose parameter names mirror the Llama
            weight structure.
        model_path: Local model directory containing a sharded safetensors
            checkpoint and `model.safetensors.index.json`.
        dtype: Optional dtype to cast loaded tensors to before loading them.
    """
    from safetensors.torch import load_file

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    index_data = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index_data["weight_map"]
    hf_to_engine = _hf_to_engine_key_map(model.config.num_layers)
    needed_hf_keys = set(hf_to_engine)
    shard_to_keys: dict[str, list[str]] = {}
    for hf_key in needed_hf_keys:
        shard_name = weight_map.get(hf_key)
        if shard_name is None:
            raise KeyError(f"Missing HF checkpoint tensor: {hf_key}")
        shard_to_keys.setdefault(shard_name, []).append(hf_key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, hf_keys in sorted(shard_to_keys.items()):
        shard_tensors = load_file(str(root / shard_name), device="cpu")
        for hf_key in hf_keys:
            tensor = shard_tensors[hf_key]
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            state_dict[hf_to_engine[hf_key]] = tensor
        del shard_tensors

    missing = set(model.state_dict()) - set(state_dict)
    if missing:
        raise KeyError(f"Missing engine tensors after load: {sorted(missing)}")
    model.load_state_dict(state_dict, strict=True)


def build_engine_from_pretrained(
    model_path: str,
    engine_config: EngineConfig,
) -> tuple[SimpleVLLMEngine, HFTokenizer]:
    """Construct a `SimpleVLLMEngine` loaded from a local HF checkpoint.

    Args:
        model_path: Local model directory.
        engine_config: Runtime settings. EOS and PAD ids are overwritten from
            the tokenizer to keep batching/generation consistent with the model.
    """
    tokenizer = HFTokenizer.from_pretrained(model_path)
    model_config = load_model_config_from_pretrained(model_path)
    model = MiniLlamaLM(model_config)
    load_pretrained_weights(model, model_path, dtype=engine_config.dtype)

    engine_config = replace(
        engine_config,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    engine = SimpleVLLMEngine(model_config, engine_config, model=model)
    return engine, tokenizer
