# updated by archita
from flask import Flask, render_template, request
import sqlite3
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# ==============================
# LOAD TRAINED PIPELINE MODEL
# ==============================

model = pickle.load(open("model/model.pkl", "rb"))

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
# HOME ROUTE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# PREDICT ROUTE
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = round(np.max(probabilities) * 100, 2)

    result = "Real News" if prediction == 1 else "Fake News"

    # Store prediction in database
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
# HISTORY ROUTE
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

if __name__ == "__main__":
    app.run(debug=True)
