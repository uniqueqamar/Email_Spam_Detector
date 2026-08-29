# Email Spam Detector

An NLP-based machine learning application that classifies messages as **spam** or **ham** (legitimate) using TF-IDF vectorization and a Multinomial Naive Bayes classifier — achieving ~98% accuracy on the SMS Spam Collection dataset.

This project is available in **two forms**:

1. **[Streamlit app](#option-1--streamlit-app)** — a single-file, all-in-one interface with built-in model evaluation charts (confusion matrix, ROC curve, feature importance)
2. **[React + FastAPI stack](#option-2--react--fastapi-stack)** — the same trained model served through a REST API, consumed by a standalone React frontend, demonstrating a decoupled, production-style architecture

Both share the exact same model, preprocessing, and dataset — the difference is purely in how the model is served and presented.

**Live demo (Streamlit):** https://emailspamdetector-x2l8ed75k8qb7bscgntfxf.streamlit.app/

---

## Model Details

- **Algorithm:** Multinomial Naive Bayes (alpha = 0.1)
- **Feature extraction:** TF-IDF, 5,000 features, unigrams + bigrams
- **Train/test split:** 80/20, stratified

| Metric | Score |
|---|---|
| Accuracy | ~98.4% |
| Precision | ~97.8% |
| Recall | ~90–91% |
| F1 Score | ~94% |
| ROC AUC | ~0.99 |

**Observation:** High precision keeps false positives low (legitimate messages rarely get flagged as spam), while slightly lower recall means a small fraction of spam messages are missed — a reasonable trade-off for a spam filter, where wrongly blocking a real message is usually costlier than letting one spam message through.

---

## Option 1 — Streamlit App

The original, self-contained version: data loading, training, evaluation, and the interactive UI all in one script.

**Features:**
- Real-time spam detection with confidence score
- Live model evaluation charts: metrics bar, confusion matrix, ROC curve
- Feature importance — top spam/ham words by log-probability
- Dataset class distribution

**Run it:**
```bash
pip install -r requirements.txt
streamlit run spam_detector.py
```

---

## Option 2 — React + FastAPI Stack

The same model, restructured into a decoupled architecture: training is separated from serving, and the UI is a standalone single-page app that talks to the model over a REST API.

```
React Frontend  →  FastAPI Backend  →  Trained Model  →  Prediction Returned
```

**Why this version exists:** it demonstrates a more production-realistic pattern — the model is trained once and saved to disk, the API loads it once at startup rather than retraining on every request, and the frontend is fully decoupled from the model logic. This is closer to how a real deployed ML service is typically structured.

### Backend (FastAPI)

- `POST /predict` — classify a message, returns label + confidence
- `GET /metrics` — returns the saved evaluation metrics
- `GET /health` — basic liveness check

**Run it:**
```bash
cd backend
pip install -r requirements.txt
python train_model.py      # trains the model, saves model.joblib + vectorizer.joblib + metrics.json
uvicorn main:app --reload --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`.

### Frontend (React + TypeScript + Vite)

A single-page interface: live model metrics, a message input, and a classification result with confidence score.

**Run it:**
```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`. The frontend expects the backend to be running on port 8000.

---

## Tech Stack

**Shared (model + data):** Python · Pandas · NumPy · Scikit-learn

**Streamlit version:** Streamlit · Matplotlib · Seaborn

**Full-stack version:** FastAPI · Uvicorn · Pydantic · Joblib (backend) · React · TypeScript · Vite (frontend)

---

## Project Structure

```
Email_Spam_Detector/
├── spam_detector.py         # Streamlit app (Option 1)
├── SMSSpamCollection        # Dataset (tab-separated: label, text)
├── requirements.txt         # Streamlit app dependencies
├── spam_app1.png / spam_app2.png / spam_detector.png
│
├── backend/                 # FastAPI backend (Option 2)
│   ├── main.py
│   ├── train_model.py
│   └── requirements.txt
│
└── frontend/                 # React frontend (Option 2)
    ├── src/
    │   ├── App.tsx
    │   ├── App.css
    │   └── main.tsx
    └── package.json
```

---

## Future Improvements

- Batch message classification (upload a CSV, get predictions for all rows)
- Try additional models (Logistic Regression, SVM) for comparison
- Dockerize the backend for easier deployment
- Deploy the FastAPI + React stack (e.g. Render/Railway for the API, Vercel for the frontend)

---

## Author

**Qamareen Fatima**
