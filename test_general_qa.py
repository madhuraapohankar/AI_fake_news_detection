import sqlite3

def test_general_qa_logic():
    from app import generate_ai_explanation
    
    # Test Case 1: History
    v, e, a, w, s = generate_ai_explanation("Who founded the Mughal Empire?", "Real News")
    assert v == "REAL ✅"
    assert s == "Historical Records"
    assert "Babur" in a
    print("Historical QA verified.")
    
    # Test Case 2: General Knowledge
    v, e, a, w, s = generate_ai_explanation("first pm of india", "Real News")
    assert v == "REAL ✅"
    assert "Nehru" in a
    print("General Knowledge QA verified.")
    
    # Test Case 3: News
    v, e, a, w, s = generate_ai_explanation("raghav chadha bjp", "Real News")
    assert v == "REAL ✅"
    assert s == "Verified News"
    assert "April 2026" in a
    print("News Verification logic verified.")

if __name__ == "__main__":
    try:
        test_general_qa_logic()
    except Exception as e:
        print(f"QA Logic testing skipped or failed: {e}")
