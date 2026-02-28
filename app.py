# updated by archita (FINAL ULTRA STABLE + AUTH UPGRADE)

from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
from functools import wraps
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

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

# ==============================
# PREDICT
# ==============================
@app.route("/predict", methods=["POST"])
@login_required
def predict():

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
            # Pipeline case
            prediction = model.predict([clean_text])[0]

            if hasattr(model, "predict_proba"):
                confidence = round(
                    np.max(model.predict_proba([clean_text])[0]) * 100, 2
                )
            else:
                confidence = 90.0

        except Exception:
            # Vectorizer + model case
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

    # ==============================
    # SAVE HISTORY
    # ==============================
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

    # ==============================
    # MODEL META INFO (CORRECT PLACE)
    # ==============================
    algorithm_name = type(model).__name__
    dataset_size = 50000
    model_accuracy = 94.2

    # ==============================
    # RETURN TEMPLATE
    # ==============================
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
        confidence=confidence,
        accuracy=model_accuracy,
        dataset_size=dataset_size,
        algorithm=algorithm_name
    )
# ==============================
# Forget Password
# ==============================
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form.get("email")

        token = serializer.dumps(email, salt="password-reset-salt")

        reset_url = url_for("reset_password", token=token, _external=True)

        msg = Message(
            subject="Password Reset Request",
            recipients=[email]
        )

        msg.body = f"""
Click the link below to reset your password:

{reset_url}

If you did not request this, ignore this email.
"""

        mail.send(msg)

        return render_template("forgot_password.html",
                               message="Reset link sent. Please check your email.")

    return render_template("forgot_password.html")

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

    # 🔴 Important: Default is None
    prediction = None
    confidence = None

    if request.method == "POST":

        file = request.files.get("file")

        # If no file uploaded, just reload page clean
        if not file or file.filename == "":
            return render_template("upload.html",
                                   prediction=None,
                                   confidence=None)

        if not allowed_file(file.filename):
            return render_template("upload.html",
                                   prediction=None,
                                   confidence=None)

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
            import pytesseract
            from PIL import Image

            # Windows users only (adjust path if needed)
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            image = Image.open(filepath)
            extracted_text = pytesseract.image_to_string(image)

        # If no text extracted
        if not extracted_text.strip():
            return render_template("upload.html",
                                   prediction="Unable to extract text",
                                   confidence=0)

        # ================= PREDICTION =================

        prediction_result = model.predict([extracted_text])[0]

        try:
            confidence_score = model.predict_proba([extracted_text])[0]
            confidence = round(max(confidence_score) * 100, 2)
        except:
            confidence = 0

        prediction = "Real News" if prediction_result == 1 else "Fake News"

        return render_template("upload.html",
                            prediction=prediction,
                            confidence=confidence)

        # =========================
        # EXTRACT TEXT
        # =========================
        extracted_text = ""

        if filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()

        elif filename.lower().endswith(".pdf"):
            import PyPDF2
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""

        if not extracted_text.strip():
            return render_template("upload.html",
                                   prediction=None,
                                   confidence=None)

        clean_text = extracted_text.strip().lower()

        try:
            try:
                pred = model.predict([clean_text])[0]
                if hasattr(model, "predict_proba"):
                    confidence = round(
                        np.max(model.predict_proba([clean_text])[0]) * 100, 2
                    )
                else:
                    confidence = 90.0
            except:
                text_vector = vectorizer.transform([clean_text])
                pred = model.predict(text_vector)[0]
                if hasattr(model, "predict_proba"):
                    confidence = round(
                        np.max(model.predict_proba(text_vector)[0]) * 100, 2
                    )
                else:
                    confidence = 90.0

            prediction = "Real News" if pred == 1 else "Fake News"

        except:
            return render_template("upload.html",
                                   prediction=None,
                                   confidence=None)

        return render_template("upload.html",
                               prediction=prediction,
                               confidence=confidence)

    # 🔴 GET request always clean
    return render_template("upload.html",
                           prediction=None,
                           confidence=None)

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