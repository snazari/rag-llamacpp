from groq import Groq
client = Groq(api_key="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")
completion = client.chat.completions.create(
    #model="moonshotai/kimi-k2-instruct",
    #model="qwen/qwen3-32b",
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {
            "role": "user",
            "content": "Explain why fast inference is critical for reasoning models"
        }
    ]
)
print(completion.choices[0].message.content)