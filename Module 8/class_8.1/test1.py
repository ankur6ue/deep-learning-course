#!/usr/bin/env python3
from transformers import AutoProcessor, Mistral3ForConditionalGeneration
import torch

model_id = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-PATCHED-457"

print("[INFO] Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)
print(type(processor).__name__)

print("[INFO] Loading full model...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(type(model).__name__)
print("[INFO] Loaded successfully.")