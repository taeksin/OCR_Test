import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Hello, GPT!"}],
    temperature=1,  # temperature 파라미터 추가 (0.0 ~ 2.0)
    reasoning_effort="minimal",
    max_completion_tokens=2024,
)
print(response)
print(response.choices[0].message.content)
