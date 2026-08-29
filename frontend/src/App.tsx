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
  dataset_size: number;
  spam_count: number;
  ham_count: number;
};

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Fetch the model's evaluation metrics once, on load.
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
      <div className="header-box">
        <h1>📧 Email Spam Detector</h1>
        <p>Multinomial Naive Bayes · TF-IDF (5000 features) · React + FastAPI</p>
      </div>

      {metrics && (
        <div className="metrics-row">
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
        </div>
      )}

      <div className="checker">
        <h2>Try it — check any message</h2>
        <textarea
          placeholder="e.g. Congratulations! You've won a free iPhone. Click here now!"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={4}
        />
        <button onClick={handleCheck} disabled={loading}>
          {loading ? "Checking..." : "Check Message"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className={result.label === "spam" ? "result-spam" : "result-ham"}>
            <h2>{result.label === "spam" ? "🚨 SPAM" : "✅ HAM (Safe)"}</h2>
            <p>This message is {result.confidence}% likely to be {result.label === "spam" ? "spam" : "safe"}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
