"""
tests/test_webhook.py

Tests for POST /webhook/activity. These NEVER touch a real Supabase
project or make a real HTTP call to an ML API — every external call is
monkeypatched at the api.webhook module level (webhook.py imports these
functions by name, so patching api.webhook.<name> is what actually takes
effect at call time).

Covers:
    - webhook authentication (missing / correct X-Webhook-Secret)
    - webhook payload parsing (malformed JSON, missing fields, invalid
      activity_type, non-UUID ids, wrong table/event type)
    - Supabase lookup logic (current_energy_level fetched per-user)
    - ML prediction integration (mocked call_ml_api)
    - final_energy_level update (scoped to the correct user only)
    - duplicate webhook handling (activity_history.process idempotency)

Run with:

    pytest tests/test_webhook.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import api.webhook as webhook_module  # noqa: E402
from api.main import app  # noqa: E402
from api.ml_client import MLServiceError  # noqa: E402
from api.supabase_client import ActivityRecordNotFoundError, WellnessRecordNotFoundError  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "svs_model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model found. Run 'python training/train.py' first.",
)

USER_1 = "00000000-0000-0000-0000-000000000001"
USER_2 = "00000000-0000-0000-0000-000000000002"
USER_UNKNOWN = "00000000-0000-0000-0000-000000000099"

ACTIVITY_HAPPY_PATH = "10000000-0000-0000-0000-000000000001"
ACTIVITY_ISOLATION = "10000000-0000-0000-0000-000000000002"
ACTIVITY_DUPLICATE = "10000000-0000-0000-0000-000000000003"
ACTIVITY_MISSING_USER = "10000000-0000-0000-0000-000000000004"
ACTIVITY_INVALID_TYPE = "10000000-0000-0000-0000-000000000005"
ACTIVITY_UNKNOWN_USER = "10000000-0000-0000-0000-000000000006"
ACTIVITY_SECURED_REJECT = "10000000-0000-0000-0000-000000000007"
ACTIVITY_SECURED_ACCEPT = "10000000-0000-0000-0000-000000000008"
ACTIVITY_ML_DOWN = "10000000-0000-0000-0000-000000000009"
ACTIVITY_NOT_FOUND = "10000000-0000-0000-0000-00000000000a"


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def mock_backend(monkeypatch: pytest.MonkeyPatch) -> Dict[str, float]:
    """
    In-memory stand-ins for wellness_scores (keyed by user_id), the
    activity_history.process flag (keyed by activity_id), and the ML API
    (a fixed, predictable prediction function instead of a real HTTP call).
    """
    wellness_store: Dict[str, float] = {USER_1: 63.0, USER_2: 20.0}
    processed_activity_ids: set = set()

    def fake_get_current_energy_level(user_id: str) -> float:
        if user_id not in wellness_store:
            raise WellnessRecordNotFoundError(f"No wellness_scores record for '{user_id}'.")
        return wellness_store[user_id]

    def fake_update_user_energy_level(user_id: str, new_energy_level: float) -> dict:
        if user_id not in wellness_store:
            raise WellnessRecordNotFoundError(f"No wellness_scores record for '{user_id}'.")
        wellness_store[user_id] = new_energy_level
        return {"user_id": user_id, "final_energy_level": new_energy_level}

    def fake_is_activity_already_processed(activity_id: str) -> bool:
        if activity_id == ACTIVITY_NOT_FOUND:
            raise ActivityRecordNotFoundError(f"No activity_history record for '{activity_id}'.")
        return activity_id in processed_activity_ids

    def fake_mark_activity_processed(activity_id: str) -> None:
        processed_activity_ids.add(activity_id)

    def fake_call_ml_api(activity_type: str, activity_energy_level: float, current_energy_level: float) -> float:
        if activity_type == "__ml_down__":
            raise MLServiceError("Simulated ML API outage.")
        # A simple, deterministic stand-in so tests don't depend on the
        # real trained model's exact output.
        return round(min(100.0, max(0.0, (activity_energy_level + current_energy_level) / 2)), 2)

    monkeypatch.setattr(webhook_module, "get_current_energy_level", fake_get_current_energy_level)
    monkeypatch.setattr(webhook_module, "update_user_energy_level", fake_update_user_energy_level)
    monkeypatch.setattr(webhook_module, "is_activity_already_processed", fake_is_activity_already_processed)
    monkeypatch.setattr(webhook_module, "mark_activity_processed", fake_mark_activity_processed)
    monkeypatch.setattr(webhook_module, "call_ml_api", fake_call_ml_api)

    return wellness_store


def _fake_webhook_payload(
    activity_id: str = ACTIVITY_HAPPY_PATH,
    user_id: str = USER_1,
    activity_type: str = "meditation",
    energy_level: float = 50,
) -> dict:
    return {
        "type": "INSERT",
        "table": "activity_history",
        "schema": "public",
        "record": {
            "id": activity_id,
            "user_id": user_id,
            "activity_type": activity_type,
            "energy_level": energy_level,
            "created_at": "2026-01-01T00:00:00Z",
        },
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def test_webhook_secret_rejects_missing_header(
    client: TestClient, mock_backend: Dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webhook_module, "WEBHOOK_SECRET", "super-secret-value")
    payload = _fake_webhook_payload(activity_id=ACTIVITY_SECURED_REJECT)

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 401


def test_webhook_secret_accepts_correct_header(
    client: TestClient, mock_backend: Dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(webhook_module, "WEBHOOK_SECRET", "super-secret-value")
    payload = _fake_webhook_payload(activity_id=ACTIVITY_SECURED_ACCEPT)

    response = client.post(
        "/webhook/activity", json=payload, headers={"X-Webhook-Secret": "super-secret-value"}
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------
def test_webhook_malformed_json_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    response = client.post(
        "/webhook/activity", content=b"{not valid json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400


def test_webhook_missing_user_id_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload(activity_id=ACTIVITY_MISSING_USER)
    del payload["record"]["user_id"]

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 400


def test_webhook_invalid_activity_type_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload(activity_id=ACTIVITY_INVALID_TYPE, activity_type="sleeping")

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 400


def test_webhook_malformed_uuid_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload(activity_id=ACTIVITY_INVALID_TYPE, user_id="not-a-real-uuid")

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 400


def test_webhook_wrong_table_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload()
    payload["table"] = "some_other_table"

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 400


def test_webhook_non_insert_event_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload()
    payload["type"] = "UPDATE"

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Supabase lookup + ML integration + final_energy_level update (happy path)
# ---------------------------------------------------------------------------
def test_webhook_happy_path_updates_only_correct_user(
    client: TestClient, mock_backend: Dict[str, float]
) -> None:
    payload = _fake_webhook_payload(
        activity_id=ACTIVITY_HAPPY_PATH, user_id=USER_1, activity_type="meditation", energy_level=50
    )

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["user_id"] == USER_1
    assert body["activity_type"] == "meditation"
    assert body["activity_energy_level"] == 50
    assert body["current_energy_level"] == 63.0  # from mock_backend's wellness_store
    assert "predicted_score" in body
    assert 0.0 <= body["predicted_score"] <= 100.0

    # Only USER_1's stored value should have changed.
    assert mock_backend[USER_1] == body["predicted_score"]
    assert mock_backend[USER_2] == 20.0


def test_webhook_user_id_isolation(client: TestClient, mock_backend: Dict[str, float]) -> None:
    original_user_2_value = mock_backend[USER_2]

    payload = _fake_webhook_payload(
        activity_id=ACTIVITY_ISOLATION, user_id=USER_2, activity_type="music", energy_level=90
    )
    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["user_id"] == USER_2
    assert body["current_energy_level"] == original_user_2_value
    assert mock_backend[USER_1] == 63.0  # untouched


def test_webhook_unknown_user_returns_404(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload(activity_id=ACTIVITY_UNKNOWN_USER, user_id=USER_UNKNOWN)

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 404


def test_webhook_ml_api_failure_does_not_update_wellness(
    client: TestClient, mock_backend: Dict[str, float]
) -> None:
    """If the ML API call fails, wellness_scores must be left untouched."""
    payload = _fake_webhook_payload(
        activity_id=ACTIVITY_ML_DOWN, user_id=USER_1, activity_type="__ml_down__", energy_level=50
    )
    # activity_type isn't one of the 5 allowed values, so bypass that check
    # by patching ALLOWED_ACTIVITY_TYPES just for this test, since the
    # point here is to exercise the ML-failure path specifically.
    original_value = mock_backend[USER_1]

    import api.webhook as wh

    old_allowed = wh.ALLOWED_ACTIVITY_TYPES
    wh.ALLOWED_ACTIVITY_TYPES = old_allowed + ["__ml_down__"]
    try:
        response = client.post("/webhook/activity", json=payload)
    finally:
        wh.ALLOWED_ACTIVITY_TYPES = old_allowed

    assert response.status_code == 502
    assert mock_backend[USER_1] == original_value  # unchanged


# ---------------------------------------------------------------------------
# Idempotency / duplicate webhook handling
# ---------------------------------------------------------------------------
def test_webhook_duplicate_activity_is_rejected(client: TestClient, mock_backend: Dict[str, float]) -> None:
    payload = _fake_webhook_payload(activity_id=ACTIVITY_DUPLICATE, user_id=USER_1)

    first_response = client.post("/webhook/activity", json=payload)
    assert first_response.status_code == 200

    second_response = client.post("/webhook/activity", json=payload)
    assert second_response.status_code == 409


def test_webhook_activity_not_found_returns_404(client: TestClient, mock_backend: Dict[str, float]) -> None:
    """If activity_history has no row for this id at all, surface a 404, not a 500."""
    payload = _fake_webhook_payload(activity_id=ACTIVITY_NOT_FOUND, user_id=USER_1)

    response = client.post("/webhook/activity", json=payload)
    assert response.status_code == 404
