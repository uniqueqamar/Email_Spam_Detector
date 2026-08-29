import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

type PredictResult = {
  label: "spam" | "ham";
  confidence: number;
};

type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
};

function MailIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="2.5" y="4.5" width="19" height="15" rx="2.5" />
      <path d="M3 6.5l9 6 9-6" />
    </svg>
  );
}

function CheckCircleIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="12" cy="12" r="9.5" />
      <path d="M8 12.5l2.5 2.5L16 9.5" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 3.5l9.5 16.5H2.5L12 3.5z" />
      <path d="M12 10v4.2" strokeLinecap="round" />
      <circle cx="12" cy="17.3" r="0.9" fill="currentColor" stroke="none" />
    </svg>
  );
}

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/metrics`)
      .then((res) => res.json())
      .then(setMetrics)
      .catch(() => setError("Could not reach the backend. Is it running on port 8000?"));
  }, []);

  async function handleCheck() {
    if (!message.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message }),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Something went wrong — check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <div className="shell">
        <header className="header">
          <div className="brand">
            <span className="brand-icon"><MailIcon /></span>
            <div>
              <p className="kicker">Machine Learning · Demo</p>
              <h1>Email Spam Detector</h1>
            </div>
          </div>
          <p className="subtitle">Multinomial Naive Bayes · TF-IDF (5,000 features) · React + FastAPI</p>
        </header>

        {metrics && (
          <section className="metrics-row">
            <div className="metric-card">
              <span className="metric-value">{(metrics.accuracy * 100).toFixed(1)}%</span>
              <span className="metric-label">Accuracy</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{(metrics.precision * 100).toFixed(1)}%</span>
              <span className="metric-label">Precision</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{(metrics.recall * 100).toFixed(1)}%</span>
              <span className="metric-label">Recall</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{(metrics.f1 * 100).toFixed(1)}%</span>
              <span className="metric-label">F1 Score</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{metrics.roc_auc.toFixed(3)}</span>
              <span className="metric-label">ROC AUC</span>
            </div>
          </section>
        )}

        <section className="panel">
          <h2>Check a message</h2>
          <p className="panel-hint">Paste any message below to classify it as spam or legitimate.</p>

          <textarea
            placeholder="e.g. Congratulations, you've been selected for a free prize. Click the link to claim now."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={4}
          />

          <div className="panel-footer">
            <button onClick={handleCheck} disabled={loading || !message.trim()}>
              {loading ? "Analyzing…" : "Check Message"}
            </button>
          </div>

          {error && <p className="error">{error}</p>}

          {result && (
            <div className={`result ${result.label === "spam" ? "result-spam" : "result-ham"}`}>
              <span className="result-icon">
                {result.label === "spam" ? <AlertIcon /> : <CheckCircleIcon />}
              </span>
              <div>
                <p className="result-label">{result.label === "spam" ? "Spam" : "Legitimate (Ham)"}</p>
                <p className="result-detail">
                  {result.confidence}% confidence this message is {result.label === "spam" ? "spam" : "safe"}
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
