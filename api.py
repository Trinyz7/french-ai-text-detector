"""API REST locale : uvicorn api:app --reload"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from detector.config import MAX_TEXT_CHARS
from detector.inference import analyze, load_detector

app = FastAPI(
    title="French AI Text Detector API",
    version="0.1.0",
    description="API expérimentale de classification de texte français (résultats non probants).",
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


@lru_cache(maxsize=1)
def get_components():
    return load_detector()


def infer(text: str):
    try:
        model, meta = get_components()
        return analyze(text, model, meta)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health():
    try:
        get_components()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "model_not_trained", "model_loaded": False}


@app.post("/predict")
def predict(payload: TextRequest):
    return infer(payload.text)


@app.post("/predict/batch")
def batch_predict(payload: BatchRequest):
    return {"results": [infer(text) for text in payload.texts]}
