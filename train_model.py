import pandas as pd
import pickle
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("dataset/fake news dataset.csv", encoding="latin1")

# Combine title + content
df["text"] = df["article_title"].fillna('') + " " + df["article_content"].fillna('')

# Rename label column
df = df.rename(columns={"labels": "label"})

# ==============================
# 2. TEXT CLEANING FUNCTION
# ==============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # Remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)    # Remove special characters
    text = re.sub(r"\s+", " ", text)          # Remove extra spaces
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# ==============================
# 3. FEATURES & TARGET
# ==============================

X = df["text"]
y = df["label"]

# ==============================
# 4. TRAIN-TEST SPLIT (STRATIFIED)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 5. BUILD ML PIPELINE
# ==============================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1,2),
        sublinear_tf=True
    )),
    ("model", MultinomialNB(alpha=0.5))
])

# ==============================
# 6. TRAIN MODEL
# ==============================

pipeline.fit(X_train, y_train)

# ==============================
# 7. EVALUATION
# ==============================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION RESULTS")
print("==============================\n")

print("Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 8. SAVE MODEL
# ==============================

os.makedirs("model", exist_ok=True)

pickle.dump(pipeline, open("model/model.pkl", "wb"))

print("\nModel saved successfully in /model folder!")
