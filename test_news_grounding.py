from app import fetch_real_news, ask_gemini
import os
from dotenv import load_dotenv

def test_news_grounding():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("SKIP: GEMINI_API_KEY not found.")
        return
    
    query = "latest political news india"
    print(f"Testing with query: {query}")
    
    # 1. Test Fetching
    news = fetch_real_news(query)
    print("\n--- Fetched News Headlines ---")
    print(news)
    
    # 2. Test Grounding
    print("\n--- Calling Grounded Gemini ---")
    response = ask_gemini(query, news)
    print("\n--- Full AI Report ---")
    print(response)
    
    assert "Verdict:" in response
    assert "Explanation:" in response
    print("\nReal-time verification PASSED.")

if __name__ == "__main__":
    test_news_grounding()
