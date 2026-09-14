"""
api/main.py

FastAPI service exposing:

    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"predicted_score": 62.31}

The model is loaded once at startup from models/svs_model.pkl and is never
retrained here. Training and prediction are fully separate concerns.

NOTE ON THE OUTPUT FIELD NAME: this endpoint returns `predicted_score`, not
`energy_level`. `predicted_score` is a pure ML/backend concept. The
Supabase column the backend eventually writes this value into
(`wellness_scores.final_energy_level`) is a different name on purpose —
see api/supabase_client.py and api/webhook.py for where that mapping
happens explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess import ALLOWED_ACTIVITY_TYPES, FEATURE_COLUMNS  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "svs_model.pkl"

app = FastAPI(
    title="SVS ML API",
    description="Predicts a user's post-activity energy score (predicted_score).",
    version="2.0.0",
)

_model_pipeline = None


@app.on_event("startup")
def load_model() -> None:
    """Load the trained pipeline into memory once, when the process starts."""
    global _model_pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Run 'python training/train.py' before starting the API."
        )
    _model_pipeline = joblib.load(MODEL_PATH)
    print(f"SVS ML API: loaded model from {MODEL_PATH}")


ActivityType = Literal["mood", "meditation", "journal", "community", "music"]


class PredictionRequest(BaseModel):
    activity_type: ActivityType = Field(
        ..., description=f"Must be one of: {ALLOWED_ACTIVITY_TYPES}"
    )
    activity_energy_level: float = Field(
        ..., ge=0, le=100, description="Energy/SVS value of the activity, 0-100."
    )
    current_energy_level: float = Field(
        ..., ge=0, le=100, description="User's energy level BEFORE the activity, 0-100."
    )


class PredictionResponse(BaseModel):
    predicted_score: float = Field(
        ..., description="Predicted post-activity energy score, 0-100."
    )


class HealthResponse(BaseModel):
    status: str


def run_model_prediction(
    activity_type: str, activity_energy_level: float, current_energy_level: float
) -> float:
    """
    Run the trained pipeline on one row of input and return a clipped,
    rounded prediction. This is the single place prediction logic lives —
    /predict calls it directly; the webhook (api/webhook.py) reaches it
    indirectly over HTTP via ML_MODEL_URL, never by importing this function.
    """
    if _model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame(
        [
            {
                "activity_type": activity_type,
                "activity_energy_level": activity_energy_level,
                "current_energy_level": current_energy_level,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    try:
        raw_prediction = _model_pipeline.predict(input_df)[0]
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    clipped_prediction = max(0.0, min(100.0, float(raw_prediction)))
    return round(clipped_prediction, 2)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness check for load balancers / uptime monitors."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict the user's post-activity energy score.

    Pydantic has already validated:
        - activity_type is one of the five allowed values
        - activity_energy_level is within [0, 100]
        - current_energy_level is within [0, 100]
    """
    predicted_score = run_model_prediction(
        activity_type=request.activity_type,
        activity_energy_level=request.activity_energy_level,
        current_energy_level=request.current_energy_level,
    )
    return PredictionResponse(predicted_score=predicted_score)


# ---------------------------------------------------------------------------
# Mount the Supabase webhook route.
#
# The webhook is implemented in api/webhook.py as its own APIRouter, and it
# talks to this /predict endpoint over HTTP (via ML_MODEL_URL), not by
# importing run_model_prediction directly. That keeps the ML API and the
# backend/webhook integration decoupled: you can run them as one process
# (as done here) today, and split them into two separately deployed
# services later without changing either module's internals.
# ---------------------------------------------------------------------------
from api.webhook import router as webhook_router  # noqa: E402

app.include_router(webhook_router)
