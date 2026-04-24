import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API key starts with: {api_key[:10] if api_key else 'None'}")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    print("SUCCESS!")
    print(response.choices[0].message.content)
except Exception as e:
    print("ERROR OCCURRED:")
    print(str(e))
