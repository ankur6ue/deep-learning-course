#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from transformers.models.ministral3.modeling_ministral3 import Ministral3ForCausalLM

SRC_MODEL = Path("/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16")
DST_MODEL = Path("/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours")

# This must be run from an environment with the transformer 5 build

def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[INFO] Copied {src.name}")


def patch_config_file(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text())

    # Make it a plain text-only HF causal LM config.
    cfg["model_type"] = "mistral"
    cfg["architectures"] = ["MistralForCausalLM"]
    cfg["tie_word_embeddings"] = False

    # Remove leftovers from newer / multimodal configs if present.
    for key in [
        "transformers_version",
        "text_config",
        "vision_config",
        "image_token_index",
        "multimodal_projector_bias",
        "projector_hidden_act",
        "spatial_merge_size",
        "quantization_config",
    ]:
        cfg.pop(key, None)

    config_path.write_text(json.dumps(cfg, indent=2))
    print(f"[INFO] Patched {config_path.name}")


def copy_working_tokenizer_assets(src_model_dir: Path, dst_model_dir: Path) -> None:
    # These are the assets that ended up working reliably in your vLLM path.
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tekken.json",
        "tokenizer.model",
    ]
    for name in tokenizer_files:
        copy_if_exists(src_model_dir / name, dst_model_dir / name)


def main() -> None:
    DST_MODEL.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading full model from: {SRC_MODEL}")
    full_model = Mistral3ForConditionalGeneration.from_pretrained(
        SRC_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("[INFO] Building text-only causal LM config...")
    text_config = full_model.model.language_model.config
    text_config.tie_word_embeddings = False

    print("[INFO] Instantiating text-only causal LM...")
    text_only_model = Ministral3ForCausalLM(text_config)

    print("[INFO] Copying language backbone...")
    text_only_model.model.load_state_dict(
        full_model.model.language_model.state_dict(),
        strict=True,
    )

    print("[INFO] Copying lm_head...")
    text_only_model.lm_head.load_state_dict(
        full_model.lm_head.state_dict(),
        strict=True,
    )

    print("[INFO] Freeing full multimodal model before save...")
    del full_model
    gc.collect()

    print("[INFO] Saving extracted text-only model...")
    text_only_model.save_pretrained(
        DST_MODEL,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    print("[INFO] Saving tokenizer once via AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(DST_MODEL)

    print("[INFO] Patching extracted config...")
    patch_config_file(DST_MODEL / "config.json")

    print("[INFO] Replacing tokenizer assets with the known-good originals...")
    copy_working_tokenizer_assets(SRC_MODEL, DST_MODEL)

    metadata = {
        "source_model": str(SRC_MODEL),
        "extracted_from": "Mistral3ForConditionalGeneration",
        "backbone_submodule": "model.language_model",
        "lm_head_source": "lm_head",
        "saved_class": "Ministral3ForCausalLM",
        "tokenizer_source": str(SRC_MODEL),
    }
    (DST_MODEL / "extraction_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[INFO] Wrote extraction_metadata.json")

    print(f"[INFO] Done. Extracted model saved to: {DST_MODEL}")


if __name__ == "__main__":
    main()