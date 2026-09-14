"""
api/ml_client.py

A thin HTTP client for calling the ML prediction API's POST /predict
endpoint, driven entirely by the ML_MODEL_URL environment variable.

WHY THIS IS A SEPARATE HTTP CALL AND NOT AN IN-PROCESS FUNCTION CALL:

The webhook handler (api/webhook.py) does not import api/main.py's model
pipeline directly. Instead it makes an HTTP request to ML_MODEL_URL. This
keeps the ML service and the Supabase/webhook backend decoupled:

    - Today, you can run everything as ONE process (ML_MODEL_URL pointing
      at http://127.0.0.1:8000/predict, i.e. the same app calling itself).
    - Later, you can deploy the ML API as its own service and point
      ML_MODEL_URL at that service's public URL, with zero code changes.

This also makes testing easier: tests mock this HTTP call instead of
needing a real model loaded in memory to test the webhook logic.
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

ML_MODEL_URL = os.environ.get("ML_MODEL_URL", "http://127.0.0.1:8000/predict")

# Keep timeouts short and explicit: a hanging ML API should not hang the
# webhook (and therefore Supabase's webhook delivery/retry logic) forever.
REQUEST_TIMEOUT_SECONDS = 10


class MLServiceError(RuntimeError):
    """Raised for any failure calling the ML prediction API (network, timeout, bad response)."""


def call_ml_api(
    activity_type: str,
    activity_energy_level: float,
    current_energy_level: float,
    ml_model_url: Optional[str] = None,
) -> float:
    """
    Call the ML API's POST /predict endpoint and return predicted_score.

    Raises MLServiceError with a clear message for:
        - connection failures (ML API unreachable / not running)
        - timeouts
        - a non-200 HTTP response
        - a 200 response missing the expected `predicted_score` field
    """
    url = ml_model_url or ML_MODEL_URL
    payload = {
        "activity_type": activity_type,
        "activity_energy_level": activity_energy_level,
        "current_energy_level": current_energy_level,
    }

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout as exc:
        raise MLServiceError(f"ML API request to '{url}' timed out after {REQUEST_TIMEOUT_SECONDS}s.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise MLServiceError(f"Could not connect to ML API at '{url}'. Is it running?") from exc
    except requests.exceptions.RequestException as exc:
        raise MLServiceError(f"ML API request to '{url}' failed: {exc}") from exc

    if response.status_code != 200:
        raise MLServiceError(
            f"ML API at '{url}' returned status {response.status_code}: {response.text}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise MLServiceError(f"ML API at '{url}' returned a non-JSON response.") from exc

    if "predicted_score" not in body:
        raise MLServiceError(
            f"ML API response is missing 'predicted_score': {body}"
        )

    return float(body["predicted_score"])
