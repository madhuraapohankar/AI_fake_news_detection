import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from app import ask_gemini_advanced, parse_ai_json

def run_tests():
    print("Testing Text Verification API with Gemini...")
    queries = [
        "Did India win the T20 World Cup in 2024?", # Should be REAL / True
        "Virat Kohli just signed a $500 Million contract with Manchester United to play football." # Should be FAKE
    ]
    
    for q in queries:
        print(f"\n--- Testing Query: {q} ---")
        try:
            raw_response = ask_gemini_advanced(q, news_context="No live context, use knowledge.")
            print("Raw Response fetched successfully. Parsing...")
            # Test JSON parser
            parsed = parse_ai_json(raw_response)
            
            print(f"VERDICT: {parsed.get('verdict')}")
            print(f"BADGE: {parsed.get('badge')}")
            print(f"CONFIDENCE: {parsed.get('confidence')}%")
            print(f"ANSWER: {parsed.get('answer')}")
            if parsed.get('verdict') == 'FAKE':
                print(f"CORRECT INFO: {parsed.get('correct_info')}")
                
        except Exception as e:
            print(f"Test Failed: {e}")

if __name__ == "__main__":
    run_tests()
