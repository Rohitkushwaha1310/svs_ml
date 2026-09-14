"""
tests/test_api.py

Tests for GET /health and POST /predict. Does not touch Supabase or the
webhook route at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "svs_model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model found. Run 'python training/train.py' first.",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("activity_type", ["mood", "meditation", "journal", "community", "music"])
def test_predict_valid_returns_predicted_score(client: TestClient, activity_type: str) -> None:
    """The response field must be `predicted_score`, not `energy_level`."""
    payload = {
        "activity_type": activity_type,
        "activity_energy_level": 80,
        "current_energy_level": 55,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "predicted_score" in body
    assert "energy_level" not in body
    assert 0.0 <= body["predicted_score"] <= 100.0


def test_predict_invalid_activity_type_is_rejected(client: TestClient) -> None:
    payload = {
        "activity_type": "sleeping",  # not one of the five allowed types
        "activity_energy_level": 50,
        "current_energy_level": 50,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_activity_energy_level_out_of_range_is_rejected(client: TestClient) -> None:
    payload = {"activity_type": "music", "activity_energy_level": 150, "current_energy_level": 40}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    payload["activity_energy_level"] = -10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_current_energy_level_out_of_range_is_rejected(client: TestClient) -> None:
    payload = {"activity_type": "journal", "activity_energy_level": 60, "current_energy_level": 101}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    payload["current_energy_level"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
