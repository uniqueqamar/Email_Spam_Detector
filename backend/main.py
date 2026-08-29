"""
FastAPI backend for the Email Spam Detector.
Loads the pre-trained TF-IDF + Naive Bayes model and exposes it over REST
so the React frontend can call it directly, instead of everything living
inside a single Streamlit script.

Run: uvicorn main:app --reload --port 8000
Requires: model.joblib, vectorizer.joblib, metrics.json (from train_model.py)
"""

import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Email Spam Detector API")

# Allow the React dev server (and later, your deployed frontend) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the trained model once, at startup ---
try:
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    with open("metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    model = None
    vectorizer = None
    METRICS = None


class Message(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    """Return the saved evaluation metrics — accuracy, precision, recall, etc."""
    if METRICS is None:
        raise HTTPException(status_code=503, detail="Model not trained yet — run train_model.py")
    return METRICS


@app.post("/predict")
def predict(message: Message):
    """Classify a single message as spam or ham, with a confidence score."""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not trained yet — run train_model.py")

    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be empty")

    vec = vectorizer.transform([message.text])
    prediction = int(model.predict(vec)[0])
    spam_probability = float(model.predict_proba(vec)[0][1])

    is_spam = prediction == 1
    return {
        "label": "spam" if is_spam else "ham",
        "confidence": round(spam_probability * 100, 1) if is_spam else round((1 - spam_probability) * 100, 1),
        "spam_probability": round(spam_probability * 100, 1),
    }
