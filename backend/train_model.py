"""
Trains the TF-IDF + Multinomial Naive Bayes spam classifier and saves it to disk.
This is the same model/preprocessing as spam_detector.py — just separated out
so a backend (FastAPI) can load it without needing Streamlit or retraining
on every request.

Run: python train_model.py
Outputs: model.joblib, vectorizer.joblib, metrics.json
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix,
)
import joblib

# --- Load data (same format as the Streamlit app) ---
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])
df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
df = df.dropna(subset=["text", "label_num"])

# --- Same split, same vectorizer, same model as the original ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_num"],
    test_size=0.2, random_state=42, stratify=df["label_num"],
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

# --- Evaluate, same metrics as the original ---
y_pred = model.predict(X_test_tfidf)
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": float(auc(fpr, tpr)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "dataset_size": len(df),
    "spam_count": int((df["label"] == "spam").sum()),
    "ham_count": int((df["label"] == "ham").sum()),
}

# --- Save everything the API needs ---
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
print(f"Precision: {metrics['precision']*100:.2f}%")
print(f"Recall:    {metrics['recall']*100:.2f}%")
print(f"F1:        {metrics['f1']*100:.2f}%")
print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
print("\nSaved model.joblib, vectorizer.joblib, metrics.json")
