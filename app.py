# updated by archita (ULTRA STABLE VERSION)

from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
import numpy as np
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)

# 🔐 VERY IMPORTANT FOR SESSION STABILITY
app.secret_key = "supersecretkey"
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"


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
# SESSION DEBUG ROUTE (REMOVE LATER)
# ==============================

@app.route("/check-session")
def check_session():
    return f"""
    User: {session.get('user')} <br>
    Email: {session.get('email')} <br>
    Role: {session.get('role')}
    """


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

        if not email or not password:
            return render_template("login.html", error="All fields required")

        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, email, role FROM users
            WHERE email=? AND password=?
        """, (email, password))

        user = cursor.fetchone()
        conn.close()

        if user:
            session.clear()  # 🔥 Prevent session corruption

            session["user"] = user[1]
            session["email"] = user[2]
            session["role"] = (user[3] or "user").lower()

            print("SESSION SAVED:", dict(session))

            if session["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("home"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ==============================
# REGISTER
# ==============================

@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    role = request.form.get("role", "user").strip().lower()

    if not name or not email or not password:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (name, email, password, role))
        conn.commit()
    except Exception as e:
        print("Registration error:", e)

    conn.close()
    return redirect(url_for("login"))


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

    print("ADMIN SESSION:", dict(session))

    if session.get("role") != "admin":
        return redirect(url_for("home"))

    return render_template("admin_dashboard.html")


# ==============================
# ADMIN HISTORY
# ==============================

@app.route("/admin-history")
@login_required
def admin_history():

    if session.get("role") != "admin":
        return redirect(url_for("home"))

    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    return render_template("admin_history.html", data=data)


# ==============================
# USER ROUTES
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


@app.route("/history")
@login_required
def history():

    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    return render_template("history.html", data=data)


# ==============================
# PREDICT
# ==============================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    text = request.form.get("news") or request.form.get("news_text")

    if not text:
        return redirect(url_for("home"))

    if model is None or vectorizer is None:
        return render_template("index.html",
                               prediction="Model not loaded",
                               confidence=0)

    text_vector = vectorizer.transform([text.lower()])
    prediction = model.predict(text_vector)[0]

    if hasattr(model, "predict_proba"):
        confidence = round(
            np.max(model.predict_proba(text_vector)[0]) * 100, 2
        )
    else:
        confidence = 90.0

    result = "Real News" if prediction == 1 else "Fake News"

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

    return render_template("index.html",
                           prediction=result,
                           confidence=confidence)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)