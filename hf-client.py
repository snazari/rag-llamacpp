import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

stream = client.chat.completions.create(
    #model="Qwen/Qwen3-Coder-480B-A35B-Instruct:novita",
    #model="moonshotai/Kimi-K2-Instruct:novita",
    model="deepseek-ai/DeepSeek-R1:sambanova",
    messages=[
        {
            "role": "user",
            "content": "Summarize the general theory of relativity."
        }
    ],
    stream=True,
)

print("\n----------")
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
print("\n----------")