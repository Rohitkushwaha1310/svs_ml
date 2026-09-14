"""
api/webhook.py

Receives the Supabase Database Webhook fired when a new row is inserted
into activity_history, calls the ML API over HTTP (see api/ml_client.py),
and writes the prediction into wellness_scores.final_energy_level for the
SAME user only.

This module is an APIRouter mounted onto the main FastAPI app in
api/main.py, so `/health` and `/predict` keep working completely
independently of everything here.

Flow:
    1. Verify the shared webhook secret (if WEBHOOK_SECRET is configured).
    2. Parse + validate the webhook payload.
    3. Only handle INSERT events on activity_history; anything else is a
       400 (nothing to do, and we don't want Supabase to keep retrying a
       request that will never apply cleanly).
    4. Check idempotency via activity_history.process (see
       api/supabase_client.py for why this re-reads from the DB instead of
       trusting the payload).
    5. Read the user's current final_energy_level from wellness_scores.
    6. Map Supabase field names -> ML field names.
    7. Call the ML API over HTTP (ML_MODEL_URL) to get predicted_score.
    8. Write predicted_score into wellness_scores.final_energy_level for
       that SAME user only.
    9. Mark the activity as processed.
   10. Return a JSON summary of every step's values.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from api.ml_client import MLServiceError, call_ml_api
from api.supabase_client import (
    ActivityRecordNotFoundError,
    WellnessRecordNotFoundError,
    get_current_energy_level,
    is_activity_already_processed,
    mark_activity_processed,
    update_user_energy_level,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("svs-ml.webhook")

# Optional shared-secret check. If unset, the webhook runs WITHOUT
# authentication — fine for local testing, never for production.
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

ALLOWED_ACTIVITY_TYPES = ["mood", "meditation", "journal", "community", "music"]

router = APIRouter()


# ---------------------------------------------------------------------------
# Webhook request/response schemas
# ---------------------------------------------------------------------------
class ActivityRecord(BaseModel):
    """
    The activity_history fields we need, as delivered inside a Supabase
    Database Webhook payload's "record" object. Field names match the
    Supabase schema EXACTLY — no renaming happens on this model.
    """

    id: UUID
    user_id: UUID
    activity_type: str
    energy_level: Optional[float] = Field(default=None, ge=0, le=100)
    metadata: Optional[Dict[str, Any]] = None
    process: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        extra = "ignore"  # title/subtitle and any other columns are irrelevant here


class SupabaseWebhookPayload(BaseModel):
    """Shape of a Supabase Database Webhook POST body."""

    type: str  # "INSERT", "UPDATE", "DELETE"
    table: str
    record: ActivityRecord
    schema_: Optional[str] = Field(default=None, alias="schema")

    class Config:
        populate_by_name = True


class WebhookResponse(BaseModel):
    success: bool
    user_id: str
    activity_type: str
    activity_energy_level: float
    current_energy_level: float
    predicted_score: float


# ---------------------------------------------------------------------------
# Field mapping (the one place this happens)
# ---------------------------------------------------------------------------
def resolve_activity_energy_level(activity: ActivityRecord) -> float:
    """
    Determine activity_energy_level from the activity_history record.

    Primary source: activity_history.energy_level (already validated 0-100
    by ActivityRecord above).

    Fallback: if energy_level is NULL for some reason but metadata carries
    an equivalent value (e.g. {"energy_level": 50} or {"svs_value": 50}),
    use that instead of failing outright. This only matters for activity
    types where energy_level might legitimately be populated asynchronously
    — remove this fallback entirely if your application never does that.
    """
    if activity.energy_level is not None:
        return float(activity.energy_level)

    if activity.metadata:
        for key in ("energy_level", "activity_energy_level", "svs_value"):
            value = activity.metadata.get(key)
            if value is not None:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= numeric_value <= 100:
                    return numeric_value

    raise HTTPException(
        status_code=400,
        detail=(
            f"Could not determine activity_energy_level for activity id='{activity.id}': "
            "energy_level is NULL and no usable value was found in metadata."
        ),
    )


def supabase_activity_to_ml_input(
    activity: ActivityRecord, activity_energy_level: float, current_energy_level: float
) -> Dict[str, Any]:
    """
    Build the ML API's request body from Supabase field values.

        activity_history.activity_type        -> ML activity_type (unchanged)
        activity_history.energy_level          -> ML activity_energy_level
        wellness_scores.final_energy_level     -> ML current_energy_level
    """
    return {
        "activity_type": activity.activity_type,
        "activity_energy_level": activity_energy_level,
        "current_energy_level": current_energy_level,
    }


def _safe_log_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of a payload dict safe to log: never logs anything that
    could be a secret or credential, even if such a field were ever added
    to the payload shape in the future.
    """
    redacted_keys = {"secret", "webhook_secret", "service_role_key", "authorization", "token", "password"}
    return {
        key: ("<redacted>" if key.lower() in redacted_keys else value)
        for key, value in payload.items()
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/webhook/activity", response_model=WebhookResponse)
async def handle_activity_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None),
) -> WebhookResponse:
    # --- 1. Verify shared secret, if configured -----------------------
    if WEBHOOK_SECRET:
        if not x_webhook_secret or x_webhook_secret != WEBHOOK_SECRET:
            logger.warning("Webhook rejected: missing or invalid X-Webhook-Secret header.")
            raise HTTPException(status_code=401, detail="Invalid or missing webhook secret.")
    else:
        logger.warning(
            "WEBHOOK_SECRET is not configured — /webhook/activity is running "
            "WITHOUT authentication. Set WEBHOOK_SECRET before deploying."
        )

    # --- 2. Parse + validate payload -----------------------------------
    try:
        raw_body = await request.json()
    except Exception as exc:
        logger.error(f"Webhook rejected: malformed JSON body ({exc}).")
        raise HTTPException(status_code=400, detail="Malformed JSON payload.") from exc

    logger.info(f"Webhook payload received (safe view): {_safe_log_payload(raw_body)}")

    try:
        payload = SupabaseWebhookPayload.model_validate(raw_body)
    except ValidationError as exc:
        logger.error(f"Webhook rejected: payload failed validation ({exc}).")
        raise HTTPException(
            status_code=400, detail=f"Invalid webhook payload: {exc.errors()}"
        ) from exc

    activity = payload.record
    logger.info(
        f"Webhook parsed: type={payload.type} table={payload.table} "
        f"activity_id={activity.id} user_id={activity.user_id} "
        f"activity_type={activity.activity_type}"
    )

    # --- 3. Only handle INSERT events on activity_history ---------------
    if payload.table != "activity_history":
        logger.info(f"Rejecting webhook for unrelated table '{payload.table}'.")
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected table '{payload.table}', expected 'activity_history'.",
        )
    if payload.type != "INSERT":
        logger.info(f"Rejecting non-INSERT event type '{payload.type}' for activity_id={activity.id}.")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported event type '{payload.type}', only INSERT is handled.",
        )
    if activity.activity_type not in ALLOWED_ACTIVITY_TYPES:
        logger.error(f"Rejecting invalid activity_type='{activity.activity_type}' for activity_id={activity.id}.")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid activity_type '{activity.activity_type}'. Allowed: {ALLOWED_ACTIVITY_TYPES}",
        )

    # --- 4. Idempotency check --------------------------------------------
    try:
        already_processed = is_activity_already_processed(str(activity.id))
    except ActivityRecordNotFoundError as exc:
        logger.error(f"Activity id={activity.id} not found in activity_history: {exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if already_processed:
        logger.info(f"Skipping already-processed activity_id={activity.id} (process != NULL).")
        raise HTTPException(
            status_code=409, detail=f"Activity '{activity.id}' has already been processed."
        )

    # --- 5. Determine activity_energy_level ------------------------------
    activity_energy_level = resolve_activity_energy_level(activity)

    # --- 6. Look up the SAME user's current wellness_scores value --------
    try:
        current_energy_level = get_current_energy_level(str(activity.user_id))
    except WellnessRecordNotFoundError as exc:
        logger.error(f"No wellness_scores record for user_id={activity.user_id}: {exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info(f"Fetched current_energy_level={current_energy_level} for user_id={activity.user_id}")

    # --- 7. Map Supabase field names to ML field names -------------------
    ml_input = supabase_activity_to_ml_input(activity, activity_energy_level, current_energy_level)
    logger.info(f"Mapped ML input: {ml_input}")

    # --- 8. Call the ML API over HTTP ------------------------------------
    try:
        predicted_score = call_ml_api(
            activity_type=ml_input["activity_type"],
            activity_energy_level=ml_input["activity_energy_level"],
            current_energy_level=ml_input["current_energy_level"],
        )
    except MLServiceError as exc:
        # Never write anything to wellness_scores if the ML call failed.
        logger.error(f"ML API call failed for activity_id={activity.id}: {exc}")
        raise HTTPException(status_code=502, detail=f"ML API call failed: {exc}") from exc

    logger.info(f"ML API returned predicted_score={predicted_score} for user_id={activity.user_id}")

    # --- 9. Update ONLY this user's wellness_scores.final_energy_level ---
    try:
        update_user_energy_level(str(activity.user_id), predicted_score)
    except WellnessRecordNotFoundError as exc:
        logger.error(f"Failed to update wellness_scores for user_id={activity.user_id}: {exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info(f"Updated wellness_scores.final_energy_level for user_id={activity.user_id} -> {predicted_score}")

    # --- 10. Mark processed for idempotency --------------------------------
    mark_activity_processed(str(activity.id))

    return WebhookResponse(
        success=True,
        user_id=str(activity.user_id),
        activity_type=activity.activity_type,
        activity_energy_level=activity_energy_level,
        current_energy_level=current_energy_level,
        predicted_score=predicted_score,
    )
