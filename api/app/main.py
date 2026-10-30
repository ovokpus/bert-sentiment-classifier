from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import model

app = FastAPI(
    title="BERT Sentiment Classifier API",
    description="Classify movie-review style text as positive or negative.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64,
                             examples=[["A wonderful, moving film.", "Two hours I will never get back."]])


class Prediction(BaseModel):
    text: str
    label: str
    positive_probability: float
    confidence: float


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok", "model": model.model_name()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    texts = [t.strip() for t in req.texts]
    if any(not t for t in texts):
        raise HTTPException(status_code=422, detail="texts must be non-empty strings")
    return {"model": model.model_name(), "predictions": model.predict(texts)}
