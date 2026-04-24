import os
from dotenv import load_dotenv
import google.generativeai as genai
from app import ask_gemini_advanced

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    res = ask_gemini_advanced("test image", "some news", is_image=True, image_warning="warning")
    print(res)
except Exception as e:
    print("FAILED:", e)
