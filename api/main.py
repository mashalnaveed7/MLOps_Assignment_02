from pathlib import Path
from threading import Lock
import os

import joblib
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
DATA_PATH = ROOT / "data" / "raw" / "breast_cancer.csv"

app = FastAPI(
    title="Breast Cancer Triage API",
    description="Educational screening-support model, not a diagnostic tool.",
    version="1.0.0",
)

_model = None
_feature_names = None
_stats_lock = Lock()
_stats = {"total_predictions": 0, "malignant_predictions": 0, "benign_predictions": 0}

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
    socket_connect_timeout=1,
)


def update_stats(prediction):
    key = "malignant_predictions" if prediction == 0 else "benign_predictions"
    try:
        with redis_client.pipeline() as pipe:
            pipe.incr("total_predictions")
            pipe.incr(key)
            pipe.execute()
    except redis.RedisError:
        with _stats_lock:
            _stats["total_predictions"] += 1
            _stats[key] += 1


def load_model():
    global _model, _feature_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model file not found.")
        if not DATA_PATH.exists():
            raise HTTPException(status_code=503, detail="Dataset file not found.")

        try:
            _model = joblib.load(MODEL_PATH)
            _feature_names = list(
                pd.read_csv(DATA_PATH, nrows=1).drop(columns=["target"]).columns
            )
        except Exception as exc:
            _model = None
            raise HTTPException(status_code=503, detail=f"Could not load model: {exc}")

    return _model


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly 30 numeric features using the dataset's column names.",
    )


@app.get("/health")
def health():
    try:
        redis_client.ping()
        redis_status = "connected"
    except redis.RedisError:
        redis_status = "unavailable"

    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "redis": redis_status,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    model = load_model()

    if set(request.features.keys()) != set(_feature_names):
        missing = sorted(set(_feature_names) - set(request.features.keys()))
        extra = sorted(set(request.features.keys()) - set(_feature_names))
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Provide exactly the 30 dataset features.",
                "missing": missing,
                "extra": extra,
            },
        )

    row = pd.DataFrame(
        [[request.features[name] for name in _feature_names]],
        columns=_feature_names,
    )

    prediction = int(model.predict(row)[0])
    label = "malignant" if prediction == 0 else "benign"

    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        malignant_index = classes.index(0)
        probability = float(model.predict_proba(row)[0][malignant_index])

    update_stats(prediction)

    return {
        "prediction": prediction,
        "label": label,
        "malignant_probability": probability,
        "model_version": "1.0.0",
        "notice": "Educational screening support only; not a medical diagnosis.",
    }


@app.get("/model/info")
def model_info():
    model = load_model()
    return {
        "model_type": type(model.named_steps.get("model", model)).__name__,
        "model_version": "1.0.0",
        "feature_count": len(_feature_names),
        "positive_class": "malignant",
        "positive_class_target": 0,
    }


@app.get("/stats")
def stats():
    try:
        values = redis_client.mget(
            "total_predictions",
            "malignant_predictions",
            "benign_predictions",
        )
        return {
            "total_predictions": int(values[0] or 0),
            "malignant_predictions": int(values[1] or 0),
            "benign_predictions": int(values[2] or 0),
            "storage": "redis",
        }
    except redis.RedisError:
        with _stats_lock:
            return {**_stats, "storage": "in-memory fallback"}
