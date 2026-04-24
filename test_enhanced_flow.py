from app import fetch_real_news, fetch_youtube_video, parse_gemini_response, ask_gemini
import os
from dotenv import load_dotenv

def test_enhanced_flow():
    load_dotenv()
    query = "Latest update on SpaceX Starship"
    print(f"Testing with query: {query}")
    
    # 1. Fetch News
    news = fetch_real_news(query)
    print(f"News fetched: {len(news)} articles.")
    assert isinstance(news, list)
    
    # 2. Fetch Video
    video = fetch_youtube_video(query)
    print(f"Video URL: {video}")
    assert "youtube.com" in video
    
    # 3. Simulate Gemini & Parse
    if os.getenv("GEMINI_API_KEY"):
        news_text = "\n\n".join([f"Title: {a['title']}" for a in news])
        raw_ai = ask_gemini(query, news_text)
        print("\n--- Raw AI ---")
        print(raw_ai)
        
        parsed = parse_gemini_response(raw_ai)
        print("\n--- Parsed ---")
        print(parsed)
        
        assert "Verdict" in parsed
        assert "confidence_int" in parsed
        print("\nEnhanced Logic Verification PASSED.")
    else:
        print("SKIP: GEMINI_API_KEY missing.")

if __name__ == "__main__":
    test_enhanced_flow()
