import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    print("Available Models:")
    for m in genai.list_models():
        print(f" - {m.name} (Methods: {m.supported_generation_methods})")
except Exception as e:
    print(f"Error fetching models: {e}")
