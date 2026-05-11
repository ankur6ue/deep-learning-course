#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


SRC = Path("/home/ankur/dev/models/Ministral-3-8B-Instruct-2512-BF16-Full-NVFP4")
DST = Path("/home/ankur/dev/models/Ministral-3-8B-Instruct-2512-BF16-Full-NVFP4-vllm")


def remap_key(k: str) -> str:
    if k.startswith("lm_head."):
        return k
    if k.startswith("model."):
        return k
    return f"model.{k}"


def patch_single_safetensors(src_path: Path, dst_path: Path) -> None:
    tensors = load_file(str(src_path))
    new_tensors = {}
    seen = set()

    for k, v in tensors.items():
        nk = remap_key(k)
        if nk in seen:
            raise ValueError(f"Duplicate remapped key: {nk}")
        seen.add(nk)
        new_tensors[nk] = v

    save_file(new_tensors, str(dst_path))
    print(f"[INFO] Patched shard: {src_path.name}")


def patch_index_json(src_path: Path, dst_path: Path) -> None:
    obj = json.loads(src_path.read_text())
    weight_map = obj.get("weight_map", {})
    new_weight_map = {}

    for k, v in weight_map.items():
        nk = remap_key(k)
        if nk in new_weight_map:
            raise ValueError(f"Duplicate remapped index key: {nk}")
        new_weight_map[nk] = v

    obj["weight_map"] = new_weight_map
    dst_path.write_text(json.dumps(obj, indent=2))
    print(f"[INFO] Patched index: {src_path.name}")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    for p in SRC.iterdir():
        dst = DST / p.name

        if p.suffix == ".json" and p.name.endswith(".index.json"):
            patch_index_json(p, dst)
        elif p.suffix == ".safetensors":
            patch_single_safetensors(p, dst)
        else:
            shutil.copy2(p, dst)

    print(f"[INFO] Done. Patched checkpoint at: {DST}")


if __name__ == "__main__":
    main()