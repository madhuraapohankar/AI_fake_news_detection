from app import ask_gemini
import os
from dotenv import load_dotenv

def test_simplification():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("SKIP: GEMINI_API_KEY not found.")
        return
    
    print("Testing simplification logic...")
    query = "india got independence in 1947"
    response = ask_gemini(query)
    
    print("Response from Gemini:")
    print(response)
    
    assert "Verdict:" in response
    assert "Confidence Score:" in response
    assert "Explanation:" in response
    assert "Correct Information:" in response
    print("Simplification verification PASSED.")

if __name__ == "__main__":
    test_simplification()
