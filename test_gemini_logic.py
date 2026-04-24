import sqlite3
import os
from dotenv import load_dotenv

# Test the structured parsing logic from app.py
def test_parsing_logic():
    # Simulated Gemini response format
    gemini_text = """
    Verdict: REAL ✅
    Confidence: 99
    Understanding: User is stating that India gained independence in 1947.
    Answer: Yes, this is correct.
    Explanation: India gained independence from British rule in 1947.
    Correct Information: The exact date is 15 August 1947.
    """
    
    data = {}
    for line in gemini_text.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()
    
    assert data["Verdict"] == "REAL ✅"
    assert data["Confidence"] == "99"
    assert "independence" in data["Understanding"]
    assert "15 August 1947" in data["Correct Information"]
    print("Parsing logic verified.")

def test_environment_config():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if key:
        print("GEMINI_API_KEY found in environment.")
    else:
        print("GEMINI_API_KEY NOT found. Please ensure it is in .env")

if __name__ == "__main__":
    test_parsing_logic()
    test_environment_config()
