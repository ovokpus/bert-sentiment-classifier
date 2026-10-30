import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from app import main, model  # noqa: E402


def fake_classifier(texts, truncation=True):
    return [
        {"label": "POSITIVE", "score": 0.98} if "great" in t.lower() else {"label": "NEGATIVE", "score": 0.9}
        for t in texts
    ]


def client(monkeypatch):
    monkeypatch.setattr(model, "get_classifier", lambda: fake_classifier)
    return TestClient(main.app)


def test_health(monkeypatch):
    r = client(monkeypatch).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_labels(monkeypatch):
    r = client(monkeypatch).post("/predict", json={"texts": ["A great film", "Awful"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["label"] for p in preds] == ["positive", "negative"]
    assert abs(preds[1]["positive_probability"] - 0.1) < 1e-6


def test_predict_rejects_empty(monkeypatch):
    c = client(monkeypatch)
    assert c.post("/predict", json={"texts": []}).status_code == 422
    assert c.post("/predict", json={"texts": ["  "]}).status_code == 422
