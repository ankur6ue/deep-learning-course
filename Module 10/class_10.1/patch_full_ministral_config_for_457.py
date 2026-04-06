#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

SRC = Path("/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16")
DST = Path("/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-PATCHED-457")

DST.mkdir(parents=True, exist_ok=True)

# Copy everything except large weight shards first; we'll use the same directory layout later.
for p in SRC.iterdir():
    if p.is_file():
        shutil.copy2(p, DST / p.name)

cfg_path = DST / "config.json"
cfg = json.loads(cfg_path.read_text())

print("Before:", cfg["model_type"], cfg["text_config"]["model_type"])
cfg["text_config"]["model_type"] = "mistral"
cfg_path.write_text(json.dumps(cfg, indent=2))
print("After:", cfg["model_type"], cfg["text_config"]["model_type"])
print("Patched:", cfg_path)