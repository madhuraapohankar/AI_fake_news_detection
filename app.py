from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import requests
import xml.etree.ElementTree as ET
import urllib.parse
import re
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import sqlite3
from authlib.integrations.flask_client import OAuth
# ==============================
# LOAD ENV
# ==============================
load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-flash-latest")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key_123")

# ==============================
# DATABASE & OAUTH SETUP
# ==============================
def init_db():
    with sqlite3.connect("database.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'user'
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                input_text TEXT,
                verdict TEXT,
                confidence INTEGER,
                date TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        # Pre-seed accounts
        c.execute("SELECT * FROM users WHERE email='admin@fakenews.com'")
        if not c.fetchone():
            c.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                      ("Admin", "admin@fakenews.com", generate_password_hash("admin123"), "admin"))
            
        c.execute("SELECT * FROM users WHERE email='user@fakenews.com'")
        if not c.fetchone():
            c.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                      ("Test User", "user@fakenews.com", generate_password_hash("user123"), "user"))
        conn.commit()

init_db()

def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def save_prediction(user_id, input_text, verdict, confidence):
    try:
        with sqlite3.connect("database.db") as conn:
            conn.execute(
                "INSERT INTO predictions (user_id, input_text, verdict, confidence, date) VALUES (?, ?, ?, ?, ?)",
                (user_id, input_text, verdict, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
    except Exception as e:
        print(f"Error saving prediction: {e}")

# ==============================
# FLASK-LOGIN & OAUTH
# ==============================
login_manager = LoginManager(app)
login_manager.login_view = "login"

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

class User(UserMixin):
    def __init__(self, id, name, email, role):
        self.id = id
        self.name = name
        self.email = email
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user_data:
        return User(id=user_data['id'], name=user_data['name'], email=user_data['email'], role=user_data['role'])
    return None

# ==============================
# FILE UPLOAD CONFIG
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================
# TRUSTED SOURCES ENGINE
# ==============================
TRUSTED_SOURCES = {
    "reuters": 95, "bbc": 95, "bbc news": 95,
    "the hindu": 90, "ndtv": 88, "the guardian": 92,
    "ap news": 95, "associated press": 95,
    "the times of india": 85, "india today": 85,
    "the indian express": 88, "hindustan times": 85,
    "the washington post": 90, "the new york times": 92,
    "al jazeera": 85, "cnn": 82, "abc news": 85,
    "france 24": 85, "dw news": 85,
    "the economic times": 82, "mint": 82,
    "news18": 78, "zee news": 75, "aaj tak": 78,
    "republic": 70, "opindia": 55,
    "the wire": 80, "scroll.in": 78, "the quint": 78,
    "google news": 60,
}

def get_source_trust(source_name):
    """Calculate trust score for a news source."""
    if not source_name:
        return 40, "unknown"
    name = source_name.lower().strip()
    for key, score in TRUSTED_SOURCES.items():
        if key in name:
            if score >= 90:
                return score, "high"
            elif score >= 75:
                return score, "medium"
            else:
                return score, "low"
    return 40, "unknown"

# ==============================
# GOOGLE NEWS FETCH (MULTI-SOURCE)
# ==============================
def fetch_google_news(query, locale="en-IN", country="IN", max_results=5):
    """Fetch top articles from Google News RSS with trust scoring."""
    articles = []
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl={locale}&gl={country}&ceid={country}:{locale.split('-')[0]}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(resp.content)
        items = root.findall("./channel/item")

        for item in items[:max_results]:
            title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            source = item.find("source").text if item.find("source") is not None else "Google News"
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""

            trust_score, trust_level = get_source_trust(source)

            articles.append({
                "title": title,
                "url": link,
                "source": source,
                "pub_date": pub_date,
                "trust_score": trust_score,
                "trust_level": trust_level
            })
    except Exception as e:
        print("News fetch error:", e)
    return articles


import concurrent.futures

def fetch_trusted_news(query):
    """Fetch news prioritizing Reuters, BBC, The Hindu using parallel threading for speed."""
    trusted_queries = [
        f"{query} site:reuters.com OR site:bbc.com OR site:thehindu.com",
        query
    ]
    all_articles = []
    
    # Run the internet searches concurrently (in parallel) to cut waiting time in half
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_google_news, q, max_results=5): q for q in trusted_queries}
        for future in concurrent.futures.as_completed(futures):
            try:
                all_articles.extend(future.result())
            except Exception as e:
                print(f"Error in parallel news fetch: {e}")

    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)
            
    # Sort by trust score (highest first)
    unique.sort(key=lambda x: x["trust_score"], reverse=True)
    return unique[:8]


# ==============================
# YOUTUBE VIDEO PROOF (TOP 3)
# ==============================
def get_youtube_links(query):
    """Return top 3 YouTube search links filtered by news channels."""
    base = "https://www.youtube.com/results?search_query="
    encoded = urllib.parse.quote(query)
    return [
        {
            "label": "📺 All News Coverage",
            "url": f"{base}{encoded}+news+verified"
        },
        {
            "label": "🔴 Reuters / BBC Coverage",
            "url": f"{base}{encoded}+reuters+OR+bbc+news"
        },
        {
            "label": "🇮🇳 Indian News Coverage",
            "url": f"{base}{encoded}+ndtv+OR+india+today+news"
        }
    ]


# ==============================
# ADVANCED AI CORE
# ==============================
def ask_gemini_advanced(user_input, news_context="", is_image=False, image_warning=""):
    """Advanced OpenAI prompt returning structured JSON."""
    today_date = datetime.now().strftime("%B %d, %Y")
    prompt = f"""You are a brilliant AI News Assistant — like ChatGPT but specialized in news verification.
You can answer ANY question: past news, current events, rumors, historical facts, science, sports — ANYTHING.

CURRENT DATE: {today_date} (Use this to understand "today", "yesterday", etc.)

USER'S QUESTION (Or Image Details): {user_input}

LIVE NEWS FROM TRUSTED SOURCES (Reuters, BBC, The Hindu, etc.):
{news_context if news_context else "No live context available — use your own vast knowledge to answer."}

{f"IMAGE ANALYSIS NOTE: {image_warning}" if image_warning else ""}
{"INSTRUCTIONS FOR IMAGE: In your 'answer' field, MUST first briefly write what is in the image based on the provided description, then state your verdict and explain it." if is_image else ""}

YOUR JOB:
1. UNDERSTAND what the user is really asking (ignore typos, grammar mistakes)
2. VERIFY the claim against trusted sources AND your own knowledge
3. Give a clear, confident answer — don't be vague or generic
4. If the news is FAKE → Clearly say it's FAKE and then tell them what the REAL truth is with evidence
5. If the news is REAL → Confirm it's REAL and provide supporting evidence
6. If MISLEADING → Explain what part is true and what's exaggerated/fake
7. For general knowledge questions (history, science, etc.) → Answer directly and confidently

IMPORTANT BEHAVIOR:
- Be conversational and helpful, like a smart friend who knows everything about news
- When something is FAKE, don't just say "fake" — explain WHY it's fake and provide the ACTUAL real news/facts
- When something is REAL, celebrate it — "Yes! This is verified from multiple sources"
- For historical facts (like "Did India get independence in 1947?") → Answer confidently from knowledge
- For rumors/WhatsApp forwards → Debunk clearly with evidence
- NEVER say generic things like "no evidence found" — always give SPECIFIC reasoning

VERDICT OPTIONS:
- REAL → Confirmed true from trusted sources or established facts
- FAKE → Proven false — and YOU MUST provide the real/correct information
- MISLEADING → Partially true but exaggerated, out of context, or AI-generated
- NOT VERIFIED → Cannot confirm or deny with available information

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no code blocks, just raw JSON):
{{
    "verdict": "REAL or FAKE or MISLEADING or NOT VERIFIED",
    "badge": "✔ Verified Fact or ⚠️ Needs Verification or ❌ Fake Claim",
    "confidence": 85,
    "confidence_breakdown": {{
        "trusted_sources_found": 3,
        "consistency": "All reports agree on this",
        "contradictions": "None found"
    }},
    "answer": "A clear, direct, conversational answer in 2-4 sentences. If FAKE, clearly state what the truth actually is.",
    "explanation": [
        "Specific evidence point 1 with source names",
        "Specific evidence point 2 explaining why this is real/fake/misleading",
        "Any red flags or strong confirmations found"
    ],
    "correct_info": "If FAKE or MISLEADING: Write the ACTUAL real news/facts here clearly. Example: 'The real news is that India won the 2024 T20 World Cup, not lost it as claimed.' If REAL: leave as empty string.",
    "social_media_warning": "Warning if this looks like a social media forward/screenshot. Empty string if not applicable.",
    "timeline": [
        {{"date": "Month Year", "event": "What actually happened"}},
        {{"date": "Month Year", "event": "How the story developed"}}
    ],
    "related_questions": [
        "Interesting follow-up question 1?",
        "Interesting follow-up question 2?",
        "Interesting follow-up question 3?"
    ]
}}

REMEMBER:
- You are NOT rigid. You are smart, flexible, and helpful.
- Answer like ChatGPT — conversational, knowledgeable, and trustworthy.
- If FAKE → Your "correct_info" field MUST contain the real truth. This is critical.
- Always provide at least 2 timeline entries and exactly 3 related questions.
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3}
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini Error: {error_msg}")
        return json.dumps({
            "verdict": "ERROR",
            "badge": "⚠️ API Connection Error",
            "confidence": 0,
            "confidence_breakdown": {"trusted_sources_found": 0, "consistency": "Error", "contradictions": "N/A"},
            "answer": f"Could not reach Gemini API. Error: {error_msg}",
            "explanation": [
                "The system failed to connect to Gemini API.",
                f"Detailed Error: {error_msg}",
                "Please make sure your API key in .env is correct."
            ],
            "correct_info": "",
            "social_media_warning": "",
            "timeline": [],
            "related_questions": []
        })


def parse_ai_json(response_text):
    """Parse OpenAI's JSON response into a structured dictionary."""
    defaults = {
        "verdict": "NOT VERIFIED",
        "badge": "⚠️ Needs Verification",
        "confidence": 50,
        "confidence_breakdown": {
            "trusted_sources_found": 0,
            "consistency": "Unknown",
            "contradictions": "Unknown"
        },
        "answer": "",
        "explanation": [],
        "correct_info": "",
        "social_media_warning": "",
        "timeline": [],
        "related_questions": []
    }

    try:
        # Clean up response - remove markdown code blocks if present
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)

        # Merge with defaults
        for key in defaults:
            if key not in result:
                result[key] = defaults[key]

        # Ensure explanation is a list
        if isinstance(result["explanation"], str):
            result["explanation"] = [result["explanation"]]

        # Ensure confidence is int
        result["confidence"] = int(result.get("confidence", 50))

        # Normalize verdict
        v = str(result["verdict"]).upper().strip()
        if "REAL" in v:
            result["verdict"] = "REAL"
        elif "FAKE" in v:
            result["verdict"] = "FAKE"
        elif "MISLEAD" in v:
            result["verdict"] = "MISLEADING"
        elif "ERROR" in v:
            result["verdict"] = "ERROR"
        else:
            result["verdict"] = "NOT VERIFIED"

        return result

    except (json.JSONDecodeError, Exception) as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw response: {response_text[:500]}")

        # Fallback: regex parsing
        result = dict(defaults)

        verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', response_text)
        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', response_text)

        if verdict_match:
            result["verdict"] = verdict_match.group(1).strip().upper()
        if conf_match:
            result["confidence"] = int(conf_match.group(1))
        if answer_match:
            result["answer"] = answer_match.group(1).strip()

        # Set badge based on verdict
        v = result["verdict"]
        if "REAL" in v:
            result["badge"] = "✔ Verified Fact"
        elif "FAKE" in v:
            result["badge"] = "❌ Fake Claim"
        else:
            result["badge"] = "⚠️ Needs Verification"

        return result


# ==============================
# VISION ANALYSIS
# ==============================
def analyze_image_with_vision(img):
    """Analyze image using Gemini Vision for social media detection."""
    try:
        response = model.generate_content([
            """Analyze this image carefully:
1. Provide a detailed description of what is in the image.
2. Extract ALL text from the image.
3. Determine if this is a social media screenshot (Instagram, Twitter/X, WhatsApp, Facebook)
4. Check for signs of manipulation: cropped headlines, edited text, fake UI elements
5. Identify the original source if visible
6. Generate a short 3-5 word news search query to verify the core claim (e.g. 'Raghav Chadha BJP join')

Return in this format:
IMAGE_DESCRIPTION: [detailed description of the visual content]
EXTRACTED_TEXT: [all text from image, or 'None' if no text]
SEARCH_QUERY: [short search query]
IS_SOCIAL_MEDIA: [yes/no]
PLATFORM: [Instagram/Twitter/WhatsApp/Facebook/News Website/Unknown]
MANIPULATION_SIGNS: [list any signs of editing or cropping]
SOURCE: [original source if visible]""",
            img
        ])
        return response.text
    except Exception as e:
        print(f"Vision Error: {e}")
        return "IMAGE_DESCRIPTION: Could not analyze image\nEXTRACTED_TEXT: Could not analyze image\nSEARCH_QUERY: none\nIS_SOCIAL_MEDIA: unknown"


def parse_vision_response(vision_text):
    """Parse vision response into structured data."""
    clean_text = vision_text.replace("**", "").replace("__", "")
    extracted_text = ""
    image_description = ""
    search_query = ""
    is_social_media = False
    platform = "Unknown"
    warning = ""

    desc_match = re.search(r"IMAGE[ _]DESCRIPTION:\s*(.*?)(?=EXTRACTED[ _]TEXT:|$)", clean_text, re.DOTALL | re.IGNORECASE)
    text_match = re.search(r"EXTRACTED[ _]TEXT:\s*(.*?)(?=SEARCH[ _]QUERY:|IS[ _]SOCIAL[ _]MEDIA:|$)", clean_text, re.DOTALL | re.IGNORECASE)
    query_match = re.search(r"SEARCH[ _]QUERY:\s*(.*?)(?=IS[ _]SOCIAL[ _]MEDIA:|$)", clean_text, re.DOTALL | re.IGNORECASE)
    social_match = re.search(r"IS[ _]SOCIAL[ _]MEDIA:\s*(yes|no)", clean_text, re.IGNORECASE)
    platform_match = re.search(r"PLATFORM:\s*(.+?)(?=MANIPULATION|SOURCE|$)", clean_text, re.IGNORECASE)
    manip_match = re.search(r"MANIPULATION[ _]SIGNS:\s*(.+?)(?=SOURCE:|$)", clean_text, re.DOTALL | re.IGNORECASE)

    if desc_match:
        image_description = desc_match.group(1).strip()
    if text_match:
        extracted_text = text_match.group(1).strip()
    if query_match:
        search_query = query_match.group(1).strip()
    if social_match and social_match.group(1).lower() == "yes":
        is_social_media = True
    if platform_match:
        platform = platform_match.group(1).strip()

    if is_social_media:
        warning = f"⚠️ This appears to be a {platform} screenshot (low reliability). Social media posts are often unverified."
    if manip_match:
        signs = manip_match.group(1).strip()
        if signs and "none" not in signs.lower():
            warning += f" Potential manipulation detected: {signs}"

    return image_description, extracted_text, search_query, is_social_media, warning


# ==============================
# ROUTES
# ==============================
@app.route("/")
@login_required
def home():
    return render_template("dashboard.html")


@app.route("/news-analyzer")
@login_required
def news_analyzer():
    return render_template("index.html")


# ==============================
# TEXT PREDICTION (MAIN AI)
# ==============================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))
        
    user_input = request.form.get("news", "").strip()
    if not user_input:
        return redirect(url_for("home"))

    # Fetch trusted news context
    news_list = fetch_trusted_news(user_input)
    news_text = "\n".join([
        f"- {a['title']} (Source: {a['source']}, Trust: {a['trust_score']}%)"
        for a in news_list
    ])

    # Ask AI
    raw_response = ask_gemini_advanced(user_input, news_text)
    result = parse_ai_json(raw_response)

    # YouTube proof links
    youtube_links = get_youtube_links(user_input)

    # Save to history
    save_prediction(current_user.id, user_input, result['verdict'], result['confidence'])

    return render_template(
        "result.html",
        result=result,
        news_context=news_list,
        youtube_links=youtube_links,
        submitted_text=user_input
    )


# ==============================
# IMAGE UPLOAD
# ==============================
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("upload.html")
        if not allowed_file(file.filename):
            return render_template("upload.html", error="Only JPG/PNG images are supported.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            img = Image.open(filepath)
            
            # Step 1: Vision analysis
            vision_text = analyze_image_with_vision(img)
            image_description, extracted_text, search_query, is_social_media, image_warning = parse_vision_response(vision_text)

            content_to_analyze = f"Image Description: {image_description}\nExtracted Text: {extracted_text}"

            if not image_description and (not extracted_text or len(extracted_text) < 10):
                content_to_analyze = f"Raw Image Analysis:\n{vision_text}"
                image_description = vision_text[:200]  # Fallback for search query

            # Step 2: Fetch news context
            if not search_query or search_query.lower() == "none":
                search_query = extracted_text[:100] if extracted_text and extracted_text.lower() != 'none' else image_description[:100]
            
            news_list = fetch_trusted_news(search_query)
            news_text = "\n".join([
                f"- {a['title']} (Source: {a['source']}, Trust: {a['trust_score']}%)"
                for a in news_list
            ])

            # Step 3: AI verification
            raw_response = ask_gemini_advanced(
                content_to_analyze, news_text,
                is_image=True, image_warning=image_warning
            )
            result = parse_ai_json(raw_response)

            # Force social media warning into result
            if is_social_media and not result.get("social_media_warning"):
                result["social_media_warning"] = image_warning

            youtube_links = get_youtube_links(search_query[:100])

            # Save to history
            save_prediction(current_user.id, f"[Image] {filename}", result['verdict'], result['confidence'])

            return render_template(
                "result.html",
                result=result,
                news_context=news_list,
                youtube_links=youtube_links,
                submitted_text=f"[Image Analyzed: {filename}]"
            )

        except Exception as e:
            print("Image error:", e)
            return render_template("upload.html", error=f"Failed to analyze image: {str(e)}")

    return render_template("upload.html")


# ==============================
# LIVE NEWS (Aaj Tak Style)
# ==============================
@app.route("/live-news")
@login_required
def live_news():
    category = request.args.get("category", "india")

    category_queries = {
        "india": "India latest news today",
        "world": "world international news today",
        "politics": "India politics news today",
        "sports": "India sports cricket news today",
        "bollywood": "Bollywood entertainment news today",
        "technology": "technology AI science news today",
        "business": "India business economy news today"
    }

    query = category_queries.get(category, "India latest news today")
    articles = fetch_google_news(query, max_results=10)

    # Scrolling ticker headlines (top 5)
    ticker = fetch_google_news("breaking news India", max_results=5)

    return render_template(
        "live_news.html",
        live_articles=articles,
        ticker_headlines=ticker,
        selected_category=category,
        categories=list(category_queries.keys())
    )


# ==============================
# KIDS NEWS (24hr Categories)
# ==============================
@app.route("/kids-news")
@login_required
def kids_news():
    category = request.args.get("category", None)

    category_queries = {
        "science": "science space discovery kids when:1d",
        "sports": "cricket football sports kids when:1d",
        "creative": "art music drawing painting kids when:1d"
    }

    fetched_news = []
    if category and category in category_queries:
        query = category_queries[category]
        fetched_news = fetch_google_news(query, max_results=6)

        # Fallback if no results
        if not fetched_news:
            fallback_queries = {
                "science": "science discovery space 2026",
                "sports": "cricket football sports today",
                "creative": "art creative kids India"
            }
            fetched_news = fetch_google_news(fallback_queries.get(category, "kids news"), max_results=6)

    return render_template(
        "kids_news.html",
        fetched_news=fetched_news,
        selected_category=category
    )


# ==============================
# VIDEO NEWS
# ==============================
@app.route("/video-news", methods=["GET", "POST"])
@login_required
def video_news():
    if request.method == "POST":
        transcript = request.form.get("news", "").strip()
        if transcript:
            return redirect(url_for("predict"), code=307)
    return render_template("video_news.html")


# ==============================
# RELATED QUESTION (AJAX)
# ==============================
@app.route("/ask-related", methods=["POST"])
@login_required
def ask_related():
    """Handle related question clicks via AJAX."""
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    news_list = fetch_trusted_news(question)
    news_text = "\n".join([
        f"- {a['title']} (Source: {a['source']}, Trust: {a['trust_score']}%)"
        for a in news_list
    ])

    raw_response = ask_gemini_advanced(question, news_text)
    result = parse_ai_json(raw_response)
    youtube_links = get_youtube_links(question)

    return render_template(
        "result.html",
        result=result,
        news_context=news_list,
        youtube_links=youtube_links,
        submitted_text=question
    )


# ==============================
# STATIC PAGES
# ==============================
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    data = conn.execute("SELECT id, input_text, verdict, confidence, date FROM predictions WHERE user_id = ? ORDER BY date DESC", (current_user.id,)).fetchall()
    conn.close()
    return render_template("history.html", data=data)

@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    
    conn = get_db_connection()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    recent_activity = conn.execute("SELECT input_text, verdict, date FROM predictions ORDER BY date DESC LIMIT 10").fetchall()
    conn.close()
    
    return render_template("admin_dashboard.html", 
                           total_users=total_users, 
                           total_predictions=total_predictions, 
                           recent_activity=recent_activity)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = "remember" in request.form
        
        conn = get_db_connection()
        user_data = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(id=user_data['id'], name=user_data['name'], email=user_data['email'], role=user_data['role'])
            login_user(user, remember=remember)
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid email or password")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role", "user")
        
        conn = get_db_connection()
        user_data = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        
        if user_data:
            conn.close()
            return render_template("register.html", error="Email address already exists")
            
        hashed_password = generate_password_hash(password)
        cursor = conn.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)", (name, email, hashed_password, role))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        user = User(id=user_id, name=name, email=email, role=role)
        login_user(user)
        return redirect(url_for("home"))
        
    return render_template("register.html")

@app.route("/forgot-password")
def forgot_password():
    return "Forgot Password functionality coming soon!"

@app.route("/google-login")
def google_login():
    redirect_uri = url_for('google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/google/authorize")
def google_authorize():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            user_info = google.userinfo()
    except Exception as e:
        print(f"Google Auth Error: {e}")
        return redirect(url_for('login'))
        
    email = user_info.get("email")
    name = user_info.get("name", "Google User")
    
    conn = get_db_connection()
    user_data = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    
    if not user_data:
        # Create a new user with a random password since they use Google
        cursor = conn.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)", 
                              (name, email, generate_password_hash(os.urandom(24).hex()), "user"))
        conn.commit()
        user_id = cursor.lastrowid
        role = "user"
    else:
        user_id = user_data['id']
        name = user_data['name']
        role = user_data['role']
        
    conn.close()
    
    user = User(id=user_id, name=name, email=email, role=role)
    login_user(user)
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)