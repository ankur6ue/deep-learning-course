from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from common.tokenizer import HFTokenizer


@contextmanager
def _torch_default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _version_symbol(module_name: str, symbol_name: str) -> Any:
    """Return a class/function from the currently imported version package.

    Each teaching folder exposes the same package name, `simple_vllm_engine`.
    The caller controls which version is active by putting `vN/` first on
    `sys.path`. This shared loader then imports the active version's config,
    model, and engine classes without knowing whether it is serving v2 or v8.
    """
    module = import_module(f"simple_vllm_engine.{module_name}")
    return getattr(module, symbol_name)


def _filtered_dataclass_init(cls: type, values: dict[str, Any]) -> Any:
    """Construct a dataclass using only fields that version defines.

    Example: v3+ `ModelConfig` accepts `attention_head_dim`, while v2 does not.
    Filtering lets the common loader support both without adding compatibility
    fields to the simpler early version.
    """
    if not is_dataclass(cls):
        return cls(**values)
    public_fields = {field.name for field in fields(cls)}
    return cls(**{name: value for name, value in values.items() if name in public_fields})


def _unwrap_text_config(hf_config: Any) -> Any:
    """Return the decoder text config for multimodal HF wrapper configs.

    Plain Llama/Mistral checkpoints put decoder fields directly on the top-level
    config. Some wrapper checkpoints store the decoder under `text_config`; the
    teaching engine only needs that decoder sub-config.
    """
    text_config = getattr(hf_config, "text_config", None)
    return text_config if text_config is not None else hf_config


def load_model_config_from_pretrained(
    model_path: str,
    max_model_len: int | None = None,
) -> Any:
    """Load a Hugging Face config into the active version's `ModelConfig`.

    The common path is intentionally shape-only. It reads the checkpoint's
    architecture dimensions and returns the current version's dataclass. Later
    versions have extra fields such as `attention_head_dim`; early versions do
    not, so construction is filtered by dataclass fields.
    """
    from transformers import AutoConfig

    try:
        top_config = AutoConfig.from_pretrained(model_path)
    except ValueError:
        # Newer architectures can appear before the local Transformers release
        # knows their `model_type`. The teaching engine only needs decoder shape
        # fields, so fall back to a plain namespace built from config.json.
        config_path = Path(model_path) / "config.json"
        top_config = SimpleNamespace(**json.loads(config_path.read_text()))

    architectures = tuple(getattr(top_config, "architectures", []) or ())
    supported_architectures = {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "Mistral3ForConditionalGeneration",
    }
    if architectures and not (set(architectures) & supported_architectures):
        raise ValueError(
            "simple_vllm_engine supports Llama/Mistral-style decoder checkpoints, "
            f"got {architectures}"
        )

    hf_config = _unwrap_text_config(top_config)
    model_type = getattr(hf_config, "model_type", None)
    if model_type not in {"llama", "mistral", "ministral3"}:
        raise ValueError(f"Unsupported decoder model_type: {model_type!r}")

    rope_scaling = (
        getattr(hf_config, "rope_scaling", None)
        or getattr(hf_config, "rope_parameters", None)
    )
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None and isinstance(rope_scaling, dict):
        rope_theta = rope_scaling.get("rope_theta")

    max_position_embeddings = int(hf_config.max_position_embeddings)
    if max_model_len is not None:
        max_position_embeddings = min(max_position_embeddings, max_model_len)

    attention_head_dim = getattr(hf_config, "head_dim", None)
    ModelConfig = _version_symbol("config", "ModelConfig")
    return _filtered_dataclass_init(
        ModelConfig,
        {
            "vocab_size": int(hf_config.vocab_size),
            "hidden_size": int(hf_config.hidden_size),
            "intermediate_size": int(hf_config.intermediate_size),
            "num_layers": int(hf_config.num_hidden_layers),
            "num_attention_heads": int(hf_config.num_attention_heads),
            "num_key_value_heads": int(getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)),
            "attention_head_dim": int(attention_head_dim) if attention_head_dim is not None else None,
            "max_position_embeddings": max_position_embeddings,
            "rope_theta": float(rope_theta if rope_theta is not None else 10000.0),
            "rope_scaling": rope_scaling,
            "rms_norm_eps": float(hf_config.rms_norm_eps),
        },
    )


def _detect_weight_prefixes(weight_map: dict[str, str]) -> tuple[str, str]:
    """Detect raw decoder weights vs wrapped `language_model.*` weights."""
    if "model.embed_tokens.weight" in weight_map:
        model_prefix = "model"
    elif "language_model.model.embed_tokens.weight" in weight_map:
        model_prefix = "language_model.model"
    else:
        raise KeyError("Could not find model embed_tokens tensor in checkpoint index")

    if "lm_head.weight" in weight_map:
        lm_head_prefix = "lm_head"
    elif "language_model.lm_head.weight" in weight_map:
        lm_head_prefix = "language_model.lm_head"
    else:
        raise KeyError("Could not find lm_head tensor in checkpoint index")
    return model_prefix, lm_head_prefix


def _uses_packed_projections(model: torch.nn.Module) -> bool:
    """Return true when this version stores QKV and gate/up as packed weights.

    Example:
    - v2 has `layers.0.q_proj.weight`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`.
    - v3+ has `layers.0.qkv_proj.weight` and `layers.0.gate_up_proj.weight`.
    """
    state_keys = set(model.state_dict())
    return "layers.0.qkv_proj.weight" in state_keys


def _hf_to_engine_key_map(
    num_layers: int,
    model_prefix: str,
    lm_head_prefix: str,
    *,
    packed: bool,
) -> dict[str, str]:
    """Map one-to-one HF tensors to active-version parameter names."""
    mapping = {
        f"{model_prefix}.embed_tokens.weight": "embed_tokens.weight",
        f"{model_prefix}.norm.weight": "norm.weight",
        f"{lm_head_prefix}.weight": "lm_head.weight",
    }
    for layer_idx in range(num_layers):
        prefix = f"{model_prefix}.layers.{layer_idx}"
        dst = f"layers.{layer_idx}"
        mapping.update(
            {
                f"{prefix}.input_layernorm.weight": f"{dst}.input_norm.weight",
                f"{prefix}.post_attention_layernorm.weight": f"{dst}.post_norm.weight",
                f"{prefix}.self_attn.o_proj.weight": f"{dst}.o_proj.weight",
                f"{prefix}.mlp.down_proj.weight": f"{dst}.down_proj.weight",
            }
        )
        if not packed:
            mapping.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": f"{dst}.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight": f"{dst}.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight": f"{dst}.v_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight": f"{dst}.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight": f"{dst}.up_proj.weight",
                }
            )
    return mapping


def _packed_hf_groups(num_layers: int, model_prefix: str) -> dict[str, list[str]]:
    """Return HF tensors concatenated into packed engine parameters.

    v3+ stores two projections per layer as packed matrices:

        qkv_proj.weight     = cat([q_proj, k_proj, v_proj], dim=0)
        gate_up_proj.weight = cat([gate_proj, up_proj], dim=0)

    The loader keeps that many-to-one mapping explicit so a reader can compare
    the checkpoint names with the optimized model-body layout.
    """
    groups: dict[str, list[str]] = {}
    for layer_idx in range(num_layers):
        prefix = f"{model_prefix}.layers.{layer_idx}"
        dst = f"layers.{layer_idx}"
        groups[f"{dst}.qkv_proj.weight"] = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ]
        groups[f"{dst}.gate_up_proj.weight"] = [
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
        ]
    return groups


def load_pretrained_weights(
    model: torch.nn.Module,
    model_path: str,
    dtype: torch.dtype | None = None,
) -> None:
    """Load a local HF Llama/Mistral-style checkpoint into the active model.

    The code supports both educational layouts used in this sequence:
    separate projection matrices in early versions and packed projection
    matrices in optimized versions.
    """
    from safetensors.torch import load_file

    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    index_data = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index_data["weight_map"]
    model_prefix, lm_head_prefix = _detect_weight_prefixes(weight_map)
    packed = _uses_packed_projections(model)

    num_layers = model.config.num_layers
    hf_to_engine = _hf_to_engine_key_map(
        num_layers,
        model_prefix,
        lm_head_prefix,
        packed=packed,
    )
    packed_groups = _packed_hf_groups(num_layers, model_prefix) if packed else {}
    packed_sources = {
        hf_key
        for hf_keys in packed_groups.values()
        for hf_key in hf_keys
    }
    packed_source_to_group = {
        hf_key: engine_key
        for engine_key, hf_keys in packed_groups.items()
        for hf_key in hf_keys
    }

    needed_hf_keys = set(hf_to_engine) | packed_sources
    shard_to_keys: dict[str, list[str]] = {}
    for hf_key in needed_hf_keys:
        shard_name = weight_map.get(hf_key)
        if shard_name is None:
            raise KeyError(f"Missing HF checkpoint tensor: {hf_key}")
        shard_to_keys.setdefault(shard_name, []).append(hf_key)

    target_state = model.state_dict()
    loaded_engine_keys: set[str] = set()
    loaded_packed_sources: dict[str, torch.Tensor] = {}

    def copy_engine_tensor(engine_key: str, tensor: torch.Tensor) -> None:
        target = target_state[engine_key]
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape mismatch for {engine_key}: checkpoint {tuple(tensor.shape)} "
                f"!= model {tuple(target.shape)}"
            )
        target_dtype = dtype if dtype is not None else target.dtype
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        target.copy_(tensor)
        loaded_engine_keys.add(engine_key)

    for shard_name, hf_keys in sorted(shard_to_keys.items()):
        shard_tensors = load_file(str(root / shard_name), device="cpu")
        for hf_key in hf_keys:
            tensor = shard_tensors[hf_key]
            if hf_key in hf_to_engine:
                copy_engine_tensor(hf_to_engine[hf_key], tensor)
                continue

            # Packed tensors may span multiple source names and shards. Hold each
            # source until the whole group is available, then concatenate in the
            # same order the optimized projection expects.
            loaded_packed_sources[hf_key] = tensor
            engine_key = packed_source_to_group[hf_key]
            group_hf_keys = packed_groups[engine_key]
            if all(group_hf_key in loaded_packed_sources for group_hf_key in group_hf_keys):
                tensors = [loaded_packed_sources.pop(group_hf_key) for group_hf_key in group_hf_keys]
                packed_tensor = torch.cat(tensors, dim=0)
                copy_engine_tensor(engine_key, packed_tensor)
                del tensors
                del packed_tensor
        del shard_tensors

    if loaded_packed_sources:
        raise RuntimeError(
            "Unexpected unpacked HF tensors left after packing: "
            f"{sorted(loaded_packed_sources)}"
        )

    missing = set(target_state) - loaded_engine_keys
    if missing:
        raise KeyError(f"Missing engine tensors after load: {sorted(missing)}")


def build_engine_from_pretrained(
    model_path: str,
    engine_config: Any,
) -> tuple[Any, HFTokenizer]:
    """Construct the active version's engine from a local HF checkpoint."""
    MiniLlamaLM = _version_symbol("model", "MiniLlamaLM")
    SimpleVLLMEngine = _version_symbol("engine", "SimpleVLLMEngine")

    tokenizer = HFTokenizer.from_pretrained(model_path)
    model_config = load_model_config_from_pretrained(
        model_path,
        max_model_len=getattr(engine_config, "max_model_len", None),
    )
    with _torch_default_dtype(engine_config.dtype):
        model = MiniLlamaLM(model_config)
    load_pretrained_weights(model, model_path, dtype=engine_config.dtype)

    engine_config = replace(
        engine_config,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    engine = SimpleVLLMEngine(model_config, engine_config, model=model)
    return engine, tokenizer
