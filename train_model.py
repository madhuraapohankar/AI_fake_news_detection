# train_model.py
# Production-Level Training Script for AI Fake News Detection

import pandas as pd
import numpy as np
import pickle
import os
import re
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =====================================
# 1️⃣ LOAD DATASET
# =====================================

print("📂 Loading dataset...")

try:
    df = pd.read_csv("dataset/fake news dataset.csv", encoding="latin1")
except Exception as e:
    print("❌ Error loading dataset:", e)
    sys.exit()

print("Original Shape:", df.shape)


# =====================================
# 2️⃣ VALIDATE REQUIRED COLUMNS
# =====================================

required_columns = ["article_title", "article_content", "labels"]

for col in required_columns:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        sys.exit()

print("✅ Required columns verified.")


# =====================================
# 3️⃣ CLEAN DATA
# =====================================

df = df.dropna()
df = df.drop_duplicates()

print("After Cleaning:", df.shape)

# Combine title + content
df["text"] = df["article_title"].fillna("") + " " + df["article_content"].fillna("")


# =====================================
# 4️⃣ TEXT CLEANING FUNCTION
# =====================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # Remove special characters
    text = re.sub(r"\s+", " ", text)             # Remove extra spaces
    return text.strip()

df["text"] = df["text"].apply(clean_text)


# =====================================
# 5️⃣ LABEL PROCESSING (VERY IMPORTANT)
# =====================================

# Convert labels safely to numeric (0 = Fake, 1 = Real)
if df["labels"].dtype == object:
    df["labels"] = df["labels"].str.lower().map({
        "fake": 0,
        "real": 1
    })

# Ensure labels are numeric
df = df[df["labels"].isin([0, 1])]

print("\nLabel Distribution:")
print(df["labels"].value_counts())


# =====================================
# 6️⃣ FEATURES & TARGET
# =====================================

X = df["text"]
y = df["labels"]


# =====================================
# 7️⃣ TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# 8️⃣ TF-IDF VECTORIZATION (UPGRADED)
# =====================================

print("\n🔤 Vectorizing text...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=8000,
    max_df=0.85,
    min_df=3
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =====================================
# 9️⃣ MODEL COMPARISON
# =====================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Multinomial NB": MultinomialNB()
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\n🧠 Training models...\n")

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, model.predict(X_train_vec))

    print(f"Model: {name}")
    print(f"Train Accuracy: {round(train_accuracy*100, 2)}%")
    print(f"Test Accuracy: {round(accuracy*100, 2)}%")
    print("-" * 40)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name


# =====================================
# 🔟 FINAL EVALUATION
# =====================================

print("\n🏆 Best Model Selected:", best_model_name)
print("Test Accuracy:", round(best_accuracy * 100, 2), "%")

y_pred_final = best_model.predict(X_test_vec)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred_final))

print("\n📌 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_final))


# =====================================
# 1️⃣1️⃣ SAVE MODEL & VECTORIZER
# =====================================

os.makedirs("model", exist_ok=True)

pickle.dump(best_model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\n💾 Model & Vectorizer saved successfully!")