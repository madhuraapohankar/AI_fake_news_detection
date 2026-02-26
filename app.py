# updated by archita (FINAL ULTRA STABLE + AUTH UPGRADE)

from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
from functools import wraps
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)

# ==============================
# SESSION SECURITY
# ==============================
app.secret_key = "supersecretkey"
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# ==============================
# PASSWORD RESET CONFIG
# ==============================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'aidetectionfakenews@gmail.com'
app.config['MAIL_PASSWORD'] = 'bfixrfodsmfzzruk'

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

# ==============================
# GOOGLE OAUTH
# ==============================
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id="your-real-client-id",
    client_secret="your-real-secret",
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
            (name, email, "google_auth", "user"),
        )
        conn.commit()
        role = "user"
    else:
        role = user[1] or "user"

    conn.close()

    session.clear()
    session["user"] = name
    session["email"] = email
    session["role"] = role

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
# LOAD MODEL
# ==============================
try:
    model = pickle.load(open("model/model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
except Exception as e:
    print("Model Load Error:", e)
    model = None
    vectorizer = None

# ==============================
# UPLOAD CONFIG
# ==============================
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "png", "jpg", "jpeg", "mp4"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

init_db()

# ==============================
# HOME
# ==============================
@app.route("/")
@login_required
def home():
    return render_template("index.html")

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
            "SELECT id, name, email, role FROM users WHERE email=? AND password=?",
            (email, password),
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            db_role = (user[3] or "user").lower()

            if db_role != role:
                return render_template("login.html", error="Wrong role selected")

            session.clear()
            session["user"] = user[1]
            session["email"] = user[2]
            session["role"] = db_role
            session["user_id"] = user[0]
            session.permanent = bool(remember)

            if db_role == "admin":
                return redirect(url_for("admin_dashboard"))

            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ==============================
# PREDICT
# ==============================
@app.route("/predict", methods=["POST"])
@login_required
def predict():

    # ✅ ALWAYS DEFINE FIRST (VERY IMPORTANT)
    source_page = request.form.get("source_page", "home")

    text = request.form.get("news") or request.form.get("news_text")

    if not text or not text.strip():
        return redirect(url_for("home"))

    if model is None or vectorizer is None:
        return render_template(
            "index.html",
            prediction="Model not loaded",
            confidence=0
        )

    try:
        clean_text = str(text).strip().lower()

        try:
            # ✅ pipeline case
            prediction = model.predict([clean_text])[0]

            if hasattr(model, "predict_proba"):
                confidence = round(
                    np.max(model.predict_proba([clean_text])[0]) * 100, 2
                )
            else:
                confidence = 90.0

        except Exception:
            # ✅ vectorizer + model case
            text_vector = vectorizer.transform([clean_text])
            prediction = model.predict(text_vector)[0]

            if hasattr(model, "predict_proba"):
                confidence = round(
                    np.max(model.predict_proba(text_vector)[0]) * 100, 2
                )
            else:
                confidence = 90.0

        result = "Real News" if prediction == 1 else "Fake News"

    except Exception as e:
        import traceback
        traceback.print_exc()

        return render_template(
            "index.html",
            prediction="Error in prediction",
            confidence=0
        )

    # ✅ SAVE HISTORY
    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history (text, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        text,
        result,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()
    conn.close()

    # ✅ SMART RETURN TO SAME PAGE
    page_map = {
        "home": "index.html",
        "live_news": "live_news.html",
        "video_news": "video_news.html",
        "kids_news": "kids_news.html",
    }

    template_name = page_map.get(source_page, "index.html")

    return render_template(
        template_name,
        prediction=result,
        confidence=confidence
    )

# ==============================
# USER ROUTES (🔥 IMPORTANT)
# ==============================
@app.route("/live-news")
@login_required
def live_news():
    return render_template("live_news.html")

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
    return render_template("kids_news.html")

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_file():
    message = ""

    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            message = "✅ File uploaded successfully"
        else:
            message = "❌ Invalid file type"

    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template("upload.html", message=message, files=uploaded_files)

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
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)