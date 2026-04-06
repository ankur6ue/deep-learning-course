from vllm import LLM, SamplingParams

model_id = "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-TextOnly"

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len=8192,
)

sampling = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=300,
)

prompt = "Explain suffix decoding in 5 bullet points."
outputs = llm.generate([prompt], sampling)

print(outputs[0].outputs[0].text)