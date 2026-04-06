#!/usr/bin/env python3
from __future__ import annotations

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours"
TOKENIZER_DIR = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16"

SYSTEM_PROMPT = """You are a financial assistant. Summarize month-over-month spending trends clearly and numerically.

Example 1
Input data:
Jan: Groceries $400, Travel $200
Feb: Groceries $500, Travel $150
Desired summary:
- Groceries increased from Jan to Feb by $100 (+25.0%).
- Travel decreased from Jan to Feb by $50 (-25.0%).

Example 2
Input data:
Mar: Amazon $120, Uber $80
Apr: Amazon $90, Uber $100
Desired summary:
- Amazon decreased from Mar to Apr by $30 (-25.0%).
- Uber increased from Mar to Apr by $20 (+25.0%).

Now analyze the following customer spending data and summarize the month-over-month trend for each item.

Instruction:
Summarize how each item changed month over month, including dollar and percentage changes."""
USER_PROMPT = """Data type: category-level spending

Apr: Groceries $490, Travel $386, Dining $474, Gas $501, Entertainment $578, Shopping $158
May: Groceries $541, Travel $336, Dining $486, Gas $613, Entertainment $629, Shopping $205
Jun: Groceries $579, Travel $306, Dining $506, Gas $760, Entertainment $588, Shopping $285

Please summarize how each item changed month over month, including dollar and percentage changes."""


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(rendered, return_tensors="pt")
    enc.pop("token_type_ids", None)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = out[0, enc["input_ids"].shape[1]:]
    print(tokenizer.decode(gen, skip_special_tokens=True))


if __name__ == "__main__":
    main()