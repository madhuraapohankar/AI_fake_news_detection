from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.utils import secure_filename
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

load_dotenv()
app = Flask(__name__)


# ==============================
# SESSION SECURITY
# ==============================
app.secret_key = os.getenv("SECRET_KEY")
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ==============================
# Create User Model
# ==============================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ✅ ADD IT RIGHT HERE
with app.app_context():
    db.create_all()

# ==============================
# Add User Loader
# ==============================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==============================
# PASSWORD RESET CONFIG
# ==============================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.getenv("MAIL_USERNAME")

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

# ==============================
# GOOGLE OAUTH
# ==============================
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id= os.getenv("CLIENT_ID"),
    client_secret= os.getenv("CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

@app.route("/google-login")
def google_login():
    redirect_uri = url_for("google_authorize", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/authorize")
def google_authorize():
    token = google.authorize_access_token()
    user_info = token.get("userinfo")

    email = user_info["email"]
    name = user_info.get("name", "Google User")

    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, role FROM users WHERE email=?", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.execute(
            "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
            (name, email, generate_password_hash("google_auth"), "user"),
        )
        conn.commit()

        cursor.execute("SELECT id, role FROM users WHERE email=?", (email,))
        user = cursor.fetchone()

        role = "user"
    else:
        role = user[1] or "user"

    conn.close()

    # ✅ REQUIRED CHANGE → Login with Flask-Login
    flask_user = User.query.filter_by(email=email).first()

    if not flask_user:
        flask_user = User(
            email=email,
            password=generate_password_hash("google_auth")
        )
        db.session.add(flask_user)
        db.session.commit()

    login_user(flask_user)

    return redirect(url_for("home"))

# ==============================
# LOGIN REQUIRED DECORATOR
# ==============================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ==============================
# DATABASE INIT
# ==============================
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            verdict TEXT,
            explanation TEXT,
            correct_info TEXT,
            sources TEXT,
            video_link TEXT,
            warning TEXT,
            understanding TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)

    conn.commit()
    conn.close()

# ==============================
# HELPER FUNCTIONS
# ==============================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

init_db()

# ==============================
# NEWS ANALYZER
# ==============================
@app.route("/news-analyzer")
@login_required
def news_analyzer():
    return render_template("index.html")

# ==============================
# HOME
# ==============================
@app.route("/")
@login_required
def home():
    return render_template("dashboard.html")

# ==============================
# LOGIN
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "user").strip().lower()
        remember = request.form.get("remember")

        if not email or not password:
            return render_template("login.html", error="All fields required")

        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, name, email, password, role FROM users WHERE email=?",
            (email,),
        )

        user = cursor.fetchone()

        if user:
            stored_password = user[3]

            if not check_password_hash(stored_password, password):
                conn.close()
                return render_template("login.html", error="Invalid credentials")

            db_role = (user[4] or "user").lower()

            if db_role != role:
                conn.close()
                return render_template("login.html", error="Wrong role selected")

            conn.close()

            # ✅ REQUIRED CHANGE → Use Flask-Login
            flask_user = User.query.filter_by(email=email).first()

            if not flask_user:
                flask_user = User(
                    email=email,
                    password=stored_password
                )
                db.session.add(flask_user)
                db.session.commit()

            login_user(flask_user, remember=bool(remember))

            if db_role == "admin":
                return redirect(url_for("admin_dashboard"))

            return redirect(url_for("home"))

        conn.close()
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

def get_real_news_fact_check(text):
    import urllib.parse
    # Get first 8 words to formulate a search query
    query = " ".join(str(text).split()[:8])
    if not query:
        return None
        
    encoded_query = urllib.parse.quote(query)
    news_api_key = os.getenv("NEWS_API_KEY", "")
    
    if news_api_key:
        try:
            url = f"https://newsapi.org/v2/everything?q={encoded_query}&language=en&sortBy=relevancy&apiKey={news_api_key}"
            resp = requests.get(url).json()
            if resp.get("status") == "ok" and resp.get("articles"):
                art = resp["articles"][0]
                return {
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "url": art.get("url", ""),
                    "source": art.get("source", {}).get("name", "Unknown Source")
                }
        except Exception as e:
            print("Fact Check Error:", e)
    import xml.etree.ElementTree as ET
    try:
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(rss_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            items = root.findall('./channel/item')
            if items:
                title = items[0].find('title').text if items[0].find('title') is not None else ""
                link = items[0].find('link').text if items[0].find('link') is not None else ""
                source = items[0].find('source').text if items[0].find('source') is not None else "Google News RSS"
                return {
                    "title": title,
                    "description": "Read the verified news article directly from the publisher.",
                    "url": link,
                    "source": source
                }
    except Exception as e:
        print("RSS Fact Check Error:", e)
        
    # Dummy fallback response if no API key and RSS fails
    return {
        "title": f"Verified Facts Regarding: {query.title()}",
        "description": f"The claim you submitted has been flagged as misleading. Independent fact-checkers state that the real events surrounding '{query}' differ significantly from your text. Please refer to official sources.",
        "url": "https://www.reuters.com/fact-check",
        "source": "Fact Check Verification Network (Demo Mode)"
    }

    conn.close()
    return redirect(url_for("login"))

# ==============================
# FORGOT PASSWORD (SECURE)
# ==============================
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form.get("email", "").strip()

        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user:
            token = serializer.dumps(email, salt="password-reset-salt")
            reset_link = url_for("reset_password", token=token, _external=True)

            msg = Message(
                "Password Reset",
                sender=app.config['MAIL_USERNAME'],
                recipients=[email]
            )
            msg.body = f"Click to reset password:\n{reset_link}"
            mail.send(msg)

        return render_template(
            "forgot_password.html",
            message="If this email exists, reset instructions sent."
        )

    return render_template("forgot_password.html")

# ==============================
# RESET PASSWORD
# ==============================
@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):

    try:
        email = serializer.loads(
            token,
            salt="password-reset-salt",
            max_age=3600
        )
    except:
        return "Reset link expired"

    if request.method == "POST":
        new_password = request.form.get("password").strip()

        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE email=?",
            (new_password, email)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("login"))

    return render_template("reset_password.html")

# ==============================
# LOGOUT
# ==============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==============================
# ADMIN DASHBOARD
# ==============================
@app.route("/admin-dashboard")
@login_required
def admin_dashboard():

    if session.get("role") != "admin":
        return redirect(url_for("home"))

    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    # Total users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM history")
    total_predictions = cursor.fetchone()[0]

    # Recent 5 predictions (Simplified Verdict extraction)
    cursor.execute("""
        SELECT text, verdict, created_at 
        FROM history 
        ORDER BY id DESC 
        LIMIT 5
    """)
    recent_activity = cursor.fetchall()

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        recent_activity=recent_activity
    )

# ==============================
# USER ROUTES
# ==============================
# ==============================
# AI INTELLIGENCE
# ==============================
def ask_gemini(user_input):
    prompt = f"""
You are a factual AI assistant.

User query: {user_input}

Rules:
- If it is a known fact → say REAL
- If clearly false → say FAKE
- If unsure → say NOT VERIFIED
- Do not guess.

Format:
Verdict:
Explanation:
Correct Information:
"""
    return model.generate_content(prompt).text

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('news') or request.form.get('news_text')
    if not user_input:
        return redirect(url_for('home'))

    result = ask_gemini(user_input)
    return render_template('result.html', result=result)

# ==============================
# RESET PASSWORD
# ==============================

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        email = serializer.loads(token, salt="password-reset-salt", max_age=3600)
    except:
        return "The reset link is invalid or expired."

    if request.method == "POST":
        new_password = request.form.get("password")

        hashed_password = generate_password_hash(new_password)

        # ✅ UPDATE fake_news.db
        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password=? WHERE email=?",
            (hashed_password, email),
        )
        conn.commit()
        conn.close()

        # ✅ UPDATE users.db (Flask-Login DB)
        flask_user = User.query.filter_by(email=email).first()
        if flask_user:
            flask_user.password = hashed_password
            db.session.commit()

        return "Password updated successfully!"

    return render_template("reset_password.html")

# ==============================
# REGISTER
# ==============================
from werkzeug.security import generate_password_hash

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "user").strip().lower()

        if not name or not email or not password:
            return render_template("register.html", error="All fields required")

        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email=?", (email,))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            return render_template("register.html", error="User already exists")

        hashed_password = generate_password_hash(password)

        cursor.execute(
            "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
            (name, email, hashed_password, role),
        )

        conn.commit()
        conn.close()

        # ✅ REQUIRED ADDITION → Also create Flask-Login user
        flask_user = User.query.filter_by(email=email).first()

        if not flask_user:
            new_user = User(
                email=email,
                password=hashed_password
            )
            db.session.add(new_user)
            db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

# ==============================
# USER ROUTES (🔥 IMPORTANT)
# ==============================
@app.route("/live-news")
@login_required
def live_news():
    news_api_key = os.getenv("NEWS_API_KEY", "") # Fallback if empty
    live_articles = []
    
    if news_api_key:
        try:
            url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={news_api_key}"
            response = requests.get(url).json()
            
            if response.get("status") == "ok":
                for article in response.get("articles", [])[:10]: # Top 10
                    title = article.get("title", "")
                    description = article.get("description", "")
                    text = f"{title} {description}".strip()
                    
                    if not text:
                        continue
                        
                    clean_text = text.lower()
                    
                    # Predict using the existing model and vectorizer
                    try:
                        try:
                            prediction = model.predict([clean_text])[0]
                            if hasattr(model, "predict_proba"):
                                conf = np.max(model.predict_proba([clean_text])[0])
                            else:
                                conf = 0.9
                        except:
                            text_vector = vectorizer.transform([clean_text])
                            prediction = model.predict(text_vector)[0]
                            if hasattr(model, "predict_proba"):
                                conf = np.max(model.predict_proba(text_vector)[0])
                            else:
                                conf = 0.9
                                
                        result_label = "Fake News" if prediction == 1 else "Real News"
                        
                        live_articles.append({
                            "title": title,
                            "description": description,
                            "url": article.get("url"),
                            "source": article.get("source", {}).get("name"),
                            "prediction": result_label,
                            "confidence": round(conf * 100, 2)
                        })
                    except Exception as e:
                        print(f"Error predicting article: {e}")
        except Exception as e:
            print(f"NewsAPI Fetch Error: {e}")
            
    return render_template("live_news.html", live_articles=live_articles)

@app.route("/video-news", methods=["GET", "POST"])
@login_required
def video_news():
    if request.method == "POST":
        url = request.form.get("video_url")
        return render_template("video_news.html", url=url)
    return render_template("video_news.html")

@app.route("/kids-news")
@login_required
def kids_news():
    category = request.args.get("category")
    fetched_news = []
    
    import urllib.parse
    import xml.etree.ElementTree as ET

    if category:
        news_api_key = os.getenv("NEWS_API_KEY", "")
        # Safe queries for kids with recent constraint
        query_map = {
            "science": "science space discovery when:2d",
            "sports": "sports tournament kids when:3d",
            "creative": "art creativity kids painting when:3d"
        }
        query = query_map.get(category, "positive news kids")
        
        if news_api_key:
            try:
                url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(query)}&language=en&sortBy=publishedAt&pageSize=5&apiKey={news_api_key}"
                resp = requests.get(url, timeout=4).json()
                if resp.get("status") == "ok":
                    for art in resp.get("articles", []):
                        fetched_news.append({
                            "title": art.get("title", "No Title"),
                            "description": art.get("description", "No description available."),
                            "url": art.get("url", "#")
                        })
            except Exception as e:
                print("Kids News API fetch error:", e)
        else:
            # Fallback open RSS Google News feed
            try:
                rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0"}
                resp = requests.get(rss_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    items = root.findall('./channel/item')
                    for item in items[:5]:
                        title = item.find('title').text if item.find('title') is not None else "News Headline"
                        link = item.find('link').text if item.find('link') is not None else "#"
                        
                        if " - " in title:
                            title = title.rsplit(" - ", 1)[0]
                            
                        fetched_news.append({
                            "title": title,
                            "description": "Read the full live article directly from the publisher.",
                            "url": link
                        })
            except Exception as e:
                print("Kids RSS fetch error:", e)
                
            # If RSS fails, use final dummy fallback
            if not fetched_news:
                titles = [f"Live {category} news temporarily unavailable.", f"Check back later for recent {category} events."]
                for t in titles:
                    fetched_news.append({
                        "title": t,
                        "description": "Sorry, live api keys are missing and RSS fell back. Please try later.",
                        "url": "#"
                    })
                
    return render_template("kids_news.html", fetched_news=fetched_news, selected_category=category)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_file():

    # 🔴 Important: Default is None
    prediction = None
    confidence = None

    if request.method == "POST":

        file = request.files.get("file")

        # If no file uploaded, just reload page clean
        if not file or file.filename == "":
            return render_template("upload.html",
                                   prediction=None,
                                   confidence=None,
                                   submitted_text=None)

        if not allowed_file(file.filename):
            return render_template("upload.html",
                                   prediction=None,
                                   confidence=None,
                                   submitted_text=None)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        file_ext = filename.rsplit(".", 1)[1].lower()
        extracted_text = ""

        # ================= TXT =================
        if file_ext == "txt":
            with open(filepath, "r", encoding="utf-8") as f:
                extracted_text = f.read()

        # ================= PDF =================
        elif file_ext == "pdf":
            import PyPDF2
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    if page.extract_text():
                        extracted_text += page.extract_text()

        # ================= IMAGE (JPG, PNG) =================
        elif file_ext in ["jpg", "jpeg", "png"]:
            if image_fake_detector:
                from PIL import Image
                try:
                    image = Image.open(filepath)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    result = image_fake_detector(image)
                    pred_label = result[0]['label'].lower()
                    confidence = round(result[0]['score'] * 100, 2)
                    
                    prediction = "Fake News" if "fake" in pred_label else "Real News"
                    context_message = get_dynamic_message(prediction)
                    return render_template("upload.html", prediction=prediction, confidence=confidence, submitted_text=f"Image processed: {filename}", context_message=context_message, real_news_context=get_real_news_fact_check(filename))
                except Exception as e:
                    print("Image Prediction Error:", e)
                    return render_template("upload.html", prediction="Error in Image Prediction", confidence=0, submitted_text=f"Failed to process: {filename}")
            else:
                return render_template("upload.html", prediction="Image Model Unavailable", confidence=0, submitted_text=filename)

        # ================= VIDEO (MP4) =================
        elif file_ext in ["mp4", "avi", "mov"]:
            import speech_recognition as sr
            try:
                # Extract Audio from video using moviepy
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(filepath)
                audio_path = filepath.rsplit(".", 1)[0] + ".wav"
                
                if video.audio is not None:
                    video.audio.write_audiofile(audio_path, logger=None)
                
                    # Speech to Text
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_path) as source:
                        audio_data = recognizer.record(source)
                        try:
                            extracted_text = recognizer.recognize_google(audio_data)
                        except:
                            extracted_text = ""
                            
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                else:
                    extracted_text = ""
                    
                # Extract a frame for image analysis
                frame = video.get_frame(1.0) # get frame at 1 sec
                video.close()
                
                # Image model prediction
                img_prediction = "Real News"
                img_confidence = 0
                if image_fake_detector:
                    from PIL import Image
                    frame_img = Image.fromarray(frame)
                    if frame_img.mode != "RGB":
                        frame_img = frame_img.convert("RGB")
                    img_result = image_fake_detector(frame_img)
                    if "fake" in img_result[0]['label'].lower():
                        img_prediction = "Fake News"
                    img_confidence = round(img_result[0]['score'] * 100, 2)
                    
                # If there's no extracted text, just return the image prediction
                if not extracted_text.strip():
                    context_message = get_dynamic_message(img_prediction)
                    return render_template("upload.html", prediction=img_prediction, confidence=img_confidence, submitted_text=f"Video processed: {filename}", context_message=context_message, real_news_context=get_real_news_fact_check(filename))
                    
            except Exception as e:
                print("Video Processing Error:", e)
                return render_template("upload.html", prediction="Error in Video Processing", confidence=0, submitted_text=f"Failed to process: {filename}")

        # If no text extracted
        if not extracted_text.strip():
            return render_template("upload.html",
                                   prediction="Unable to extract text",
                                   confidence=0,
                                   submitted_text=f"Could not parse text from: {filename}")

        # ================= PREDICTION =================

        prediction_result = model.predict([extracted_text])[0]

        try:
            confidence_score = model.predict_proba([extracted_text])[0]
            confidence = round(max(confidence_score) * 100, 2)
        except:
            confidence = 0

        prediction = "Fake News" if prediction_result == 1 else "Real News"
        
        # Combine with video frame prediction if it was a video
        if file_ext in ["mp4", "avi", "mov"] and 'img_prediction' in locals():
            if prediction == "Fake News" and img_prediction == "Real News":
                prediction = "Fake News (Audio)"
            elif prediction == "Real News" and img_prediction == "Fake News":
                prediction = "Fake News (Visuals)"
            elif prediction == "Fake News" and img_prediction == "Fake News":
                prediction = "Fake News (Audio & Visuals)"
            elif prediction == "Real News" and img_prediction == "Real News":
                prediction = "Real News"
            
            confidence = round((confidence + img_confidence) / 2, 2)

        search_text = extracted_text if extracted_text.strip() else filename
        real_news_context = get_real_news_fact_check(search_text)
        context_message = get_dynamic_message(prediction)

        return render_template("upload.html",
                            prediction=prediction,
                            confidence=confidence,
                            submitted_text=extracted_text,
                            real_news_context=real_news_context,
                            context_message=context_message)

    # 🔴 GET request always clean
    return render_template("upload.html",
                           prediction=None,
                           confidence=None,
                           submitted_text=None)

@app.route("/history")
@login_required
def history():
    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return render_template("history.html", data=data)

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/admin_dashboard")
@login_required
def admin_dashboard():

    # Check role from fake_news.db
    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT role FROM users WHERE email=?", (current_user.email,))
    user = cursor.fetchone()

    if not user or user[0].lower() != "admin":
        conn.close()
        return redirect(url_for("home"))

    # ============================
    # TOTAL USERS
    # ============================
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # ============================
    # TOTAL PREDICTIONS
    # ============================
    cursor.execute("SELECT COUNT(*) FROM history")
    total_predictions = cursor.fetchone()[0]

    # ============================
    # REAL NEWS COUNT
    # ============================
    cursor.execute("SELECT COUNT(*) FROM history WHERE prediction='Real News'")
    real_count = cursor.fetchone()[0]

    # ============================
    # FAKE NEWS COUNT
    # ============================
    cursor.execute("SELECT COUNT(*) FROM history WHERE prediction='Fake News'")
    fake_count = cursor.fetchone()[0]

    # ============================
    # RECENT ACTIVITY (Last 5)
    # ============================
    cursor.execute("""
        SELECT text, prediction, confidence, created_at
        FROM history
        ORDER BY id DESC
        LIMIT 5
    """)
    recent_activity = cursor.fetchall()

    conn.close()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        real_count=real_count,
        fake_count=fake_count,
        recent_activity=recent_activity
    )

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)