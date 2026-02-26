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
app.config['MAIL_PASSWORD'] = 'bfix rfod smfz zruk'

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

    # This automatically contains user info when using OpenID
    user_info = token.get("userinfo")

    email = user_info["email"]
    name = user_info.get("name", "Google User")

    conn = sqlite3.connect("database/fake_news.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, role FROM users WHERE email=?", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.execute("""
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (name, email, "google_auth", "user"))
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

        # ✅ Validation
        if not email or not password:
            return render_template("login.html", error="All fields required")

        conn = sqlite3.connect("database/fake_news.db")
        cursor = conn.cursor()

        # ✅ Check user
        cursor.execute("""
            SELECT id, name, email, role FROM users
            WHERE email=? AND password=?
        """, (email, password))

        user = cursor.fetchone()
        conn.close()

        if user:
            db_role = (user[3] or "user").lower()

            # ✅ Role check
            if db_role != role:
                return render_template("login.html", error="Wrong role selected")

            # ✅ Clear old session
            session.clear()

            # ⭐ IMPORTANT FOR NAVBAR
            session["user"] = user[1]   # name (for Welcome, Name)
            session["email"] = user[2]
            session["role"] = db_role
            session["user_id"] = user[0]  # ⭐ professional touch

            # ✅ Remember me
            if remember:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=7)
            else:
                session.permanent = False

            # ✅ Redirect by role
            if db_role == "admin":
                return redirect(url_for("admin_dashboard"))

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

    # Real vs Fake count
    cursor.execute("SELECT COUNT(*) FROM history WHERE prediction='Real News'")
    real_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE prediction='Fake News'")
    fake_count = cursor.fetchone()[0]

    # Recent 5 predictions
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

# ==============================
# UPLOAD WITH PREVIEW
# ==============================
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

    return render_template(
        "upload.html",
        message=message,
        files=uploaded_files
    )

# ==============================
# HISTORY
# ==============================
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
        return render_template(
            "index.html",
            prediction="Model not loaded",
            confidence=0
        )

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

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )

# ==============================
# ABOUT PAGE
# ==============================
@app.route("/about")
@login_required
def about():
    return render_template("about.html")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)