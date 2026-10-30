import { useEffect, useState } from "react";

type Pred = { text: string; label: string; positive_probability: number; confidence: number };

const EXAMPLES = [
  "An absolute masterpiece. The performances were moving and the score was stunning.",
  "Dull, predictable and far too long. I nearly walked out.",
  "The plot was thin, but the visuals carried it.",
];

export default function App() {
  const [text, setText] = useState(EXAMPLES.join("\n"));
  const [preds, setPreds] = useState<Pred[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetch("/api/health").then((r) => r.json()).then((h) => { setOnline(true); setModel(h.model); }).catch(() => setOnline(false));
  }, []);

  async function classify() {
    const texts = text.split("\n").map((t) => t.trim()).filter(Boolean);
    if (!texts.length) { setErr("Enter at least one review."); return; }
    setBusy(true); setErr("");
    try {
      const r = await fetch("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ texts }) });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      setPreds((await r.json()).predictions);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  const pos = preds.filter((p) => p.label === "positive").length;

  return (
    <div className="page">
      <header>
        <h1>🎬 BERT Sentiment Classifier</h1>
        <p>Paste movie reviews, one per line, and a BERT model scores each one as positive or negative.</p>
        <span className={`pill ${online === false ? "bad" : ""}`}>
          {online === null ? "connecting…" : online ? `API online · ${model}` : "API offline"}
        </span>
      </header>
      <section className="card">
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} />
        <div className="row">
          <button onClick={classify} disabled={busy}>{busy ? "Scoring…" : "Classify"}</button>
          {preds.length > 0 && <span className="summary">{pos} positive · {preds.length - pos} negative</span>}
          <a href="http://localhost:8000/docs" target="_blank">API docs ↗</a>
        </div>
        {err && <p className="err">{err}</p>}
      </section>
      <section className="results">
        {preds.map((p, i) => (
          <div key={i} className={`result ${p.label}`}>
            <div className="head">
              <b>{p.label === "positive" ? "👍 Positive" : "👎 Negative"}</b>
              <span>confidence {(p.confidence * 100).toFixed(1)}%</span>
            </div>
            <p>{p.text}</p>
            <div className="meter"><div style={{ width: `${p.positive_probability * 100}%` }} /></div>
            <small>P(positive) = {p.positive_probability.toFixed(4)}</small>
          </div>
        ))}
      </section>
    </div>
  );
}
