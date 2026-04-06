#!/usr/bin/env python3
from __future__ import annotations

import os

from vllm import LLM, SamplingParams

MODEL_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours"
TOKENIZER_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours"
PROMPT = "Explain suffix decoding in 5 bullet points."


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[INFO] Loading vLLM model from: {MODEL_DIR}")
    print(f"[INFO] Using tokenizer from: {TOKENIZER_DIR}")

    llm = LLM(
        model=MODEL_DIR,
        tokenizer=TOKENIZER_DIR,
        tokenizer_mode="mistral",
        trust_remote_code=False,
        max_model_len=4096,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
    )

    outputs = llm.chat(
        messages=[{"role": "user", "content": PROMPT}],
        sampling_params=sampling,
    )

    print("\n=== OUTPUT ===\n")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()