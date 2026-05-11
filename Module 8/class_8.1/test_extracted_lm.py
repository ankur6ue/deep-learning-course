import torch
from transformers import AutoTokenizer, pipeline

model_dir = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours"
prompt = "Explain suffix decoding in 5 bullet points."

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

pipe = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

out = pipe(
    rendered,
    max_new_tokens=250,
    do_sample=False,
    return_full_text=False,
)

print(out[0]["generated_text"])