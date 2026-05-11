from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8003/v1",
    api_key="dummy",
)

# resp = client.chat.completions.create(
#     model="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-GPTQ-W4A16",
#     messages=[
#         {"role": "user", "content": "Explain suffix decoding in 5 bullet points."}
#     ],
#     temperature=0.0,
#     max_tokens=200,
# )
resp = client.chat.completions.create(
    model="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-NVFP4",
    messages=[
        {"role": "user", "content": "Explain suffix decoding in 5 bullet points."}
    ],
    temperature=0.0,
    max_tokens=200,
)
resp = client.chat.completions.create(
    model="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-NVFP4",
    messages=[
        {"role": "user", "content": "Explain suffix decoding in 5 bullet points."}
    ],
    temperature=0.0,
    max_tokens=200,
)
print(resp.choices[0].message.content)

###
# Here’s a concise explanation of **suffix decoding** in 5 bullet points:
#
# - **Definition**: Suffix decoding is a technique used in **compression algorithms** (e.g., LZ77, LZSS) to identify repeated sequences in data by matching **suffixes** (end portions) of strings with previously seen substrings.
#
# - **How It Works**:
#   - The algorithm scans a **sliding window** of data (e.g., a buffer) to find the **longest matching suffix** of the current input with a substring already processed.
#   - It records the **offset** (distance back in the buffer) and **length** of the match, followed by the **unmatched suffix** (literal bytes).
#
# - **Key Components**:
#   - **Search Buffer**: Stores recently processed data for matching.
#   - **Lookahead Buffer**: Holds the current input being analyzed.
#   - **Output**: Encoded as `(offset, length,
###