#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC_MODEL = Path("/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16")
MODEL_DIR = Path("/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[INFO] Copied {src.name}")


def patch_config(path: Path) -> None:
    cfg = json.loads(path.read_text())
    cfg["model_type"] = "mistral"
    cfg["architectures"] = ["MistralForCausalLM"]
    cfg["tie_word_embeddings"] = False

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

    path.write_text(json.dumps(cfg, indent=2))
    print(f"[INFO] Patched {path.name}")


def main() -> None:
    patch_config(MODEL_DIR / "config.json")

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tekken.json",
        "tokenizer.model",
    ]
    for name in tokenizer_files:
        copy_if_exists(SRC_MODEL / name, MODEL_DIR / name)

    print("[INFO] Patch complete.")


if __name__ == "__main__":
    main()