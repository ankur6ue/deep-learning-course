#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer, pipeline

MODEL_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours"
# This doesn't work
TOKENIZER_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours"
# This does..
TOKENIZER_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16"
PROMPT = "Explain suffix decoding in 5 bullet points."


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[INFO] Loading tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
     #   MODEL_DIR,
        TOKENIZER_DIR,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    print(f"[INFO] Loading model from: {MODEL_DIR}")
    pipe = pipeline(
        "text-generation",
        model=MODEL_DIR,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("[INFO] Running generation...")
    out = pipe(
        rendered,
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False,
    )

    print("\n=== OUTPUT ===\n")
    print(out[0]["generated_text"])


if __name__ == "__main__":
    main()