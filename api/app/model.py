"""Sentiment model wrapper.

The original notebook fine-tunes a TF-Hub BERT model on Vertex AI; those
weights were never published. For local serving we use a small public
BERT-family checkpoint fine-tuned for the same binary task (SST-2), pulled
from the Hugging Face Hub at image build time. Override with MODEL_NAME.
"""
import os
from functools import lru_cache

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=model_name(), device=-1)


def predict(texts: list[str], classifier=None) -> list[dict]:
    clf = classifier or get_classifier()
    results = clf(texts, truncation=True)
    out = []
    for text, r in zip(texts, results):
        label = r["label"].lower()
        score = float(r["score"])
        pos = score if label == "positive" else 1.0 - score
        out.append(
            {
                "text": text,
                "label": "positive" if pos >= 0.5 else "negative",
                "positive_probability": round(pos, 4),
                "confidence": round(max(pos, 1 - pos), 4),
            }
        )
    return out
