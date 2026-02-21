# updated by archita (cleaned & upgraded)

from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
import numpy as np
from datetime import datetime
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ==============================
# LOAD TRAINED MODEL
# ==============================

try:
    model = pickle.load(open("model/model.pkl", "rb"))
except Exception as e:
    print("Model load error:", e)
    model = None

# ==============================
# UPLOAD CONFIG (FIXED POSITION)
# ==============================

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "png", "jpg", "jpeg", "mp4"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================
# DATABASE INITIALIZATION
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

    conn.commit()
    conn.close()


init_db()

# ==============================
# HOME
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# LOGIN / REGISTER
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    return redirect(url_for("login"))


# ==============================
# LIVE NEWS
# ==============================

@app.route("/live-news")
def live_news():
    return render_template("live_news.html")


# ==============================
# VIDEO NEWS
# ==============================

@app.route("/video-news", methods=["GET", "POST"])
def video_news():
    if request.method == "POST":
        url = request.form.get("video_url")
        return render_template("video_news.html", url=url)

    return render_template("video_news.html")


# ==============================
# FILE UPLOAD
# ==============================

@app.route("/upload", methods=["GET", "POST"])
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

    return render_template("upload.html", message=message)

# ==============================
# DASHBOARD
# ==============================

@app.route("/admin-dashboard")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")
    
# ==============================
# KIDS NEWS
# ==============================

@app.route("/kids-news")
def kids_news():
    return render_template("kids_news.html")


# ==============================
# PAST NEWS
# ==============================
@app.route("/past-news")
def past_news():
    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    return render_template("past_news.html", data=data)

# ==============================
# PREDICT
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news")

    if not text:
        return redirect(url_for("home"))

    if model is None:
        return render_template(
            "index.html",
            prediction="Model not loaded",
            confidence=0
        )

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = round(np.max(probabilities) * 100, 2)

    result = "Real News" if prediction == 1 else "Fake News"

    # Save to DB
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

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


# ==============================
# HISTORY
# ==============================

@app.route("/history")
def history():
    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    return render_template("history.html", data=data)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)