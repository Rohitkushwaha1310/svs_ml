"""
api/webhook.py

Handles Supabase INSERT webhooks from:

    public.activity_history

Flow:

    activity_history INSERT
            |
            v
    /webhook/activity
            |
            +--> get activity_type
            |
            +--> get activity_history.energy_level
            |
            +--> find SAME user in wellness_scores
            |
            +--> read wellness_scores.final_energy_level
            |
            +--> run the already-loaded ML model
            |
            +--> update SAME user's
                 wellness_scores.final_energy_level
            |
            +--> mark activity_history.process = "done"

No database schema changes are required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any, Dict, Optional
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from api.supabase_client import (
    ActivityRecordNotFoundError,
    WellnessRecordNotFoundError,
    get_current_energy_level,
    is_activity_already_processed,
    mark_activity_processed,
    update_user_energy_level,
)

load_dotenv()

logger = logging.getLogger("svs-ml.webhook")

router = APIRouter()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ALLOWED_ACTIVITY_TYPES = {
    "mood",
    "meditation",
    "journal",
    "community",
    "music",
}


# ============================================================
# SUPABASE ACTIVITY RECORD
# ============================================================

class ActivityRecord(BaseModel):
    """
    Fields from public.activity_history.

    These names match the Supabase columns exactly.
    """

    model_config = ConfigDict(extra="ignore")

    id: UUID

    user_id: UUID

    activity_type: str

    # public.activity_history.energy_level
    energy_level: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    metadata: Optional[Dict[str, Any]] = None

    # NULL before processing, "done" after processing.
    process: Optional[str] = None

    created_at: Optional[str] = None


# ============================================================
# SUPABASE WEBHOOK PAYLOAD
# ============================================================

class SupabaseWebhookPayload(BaseModel):
    """
    Standard Supabase Database Webhook payload.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    type: str

    table: str

    record: ActivityRecord

    old_record: Optional[Dict[str, Any]] = None

    schema_name: Optional[str] = Field(
        default=None,
        alias="schema",
    )


# ============================================================
# RESPONSE
# ============================================================

class WebhookResponse(BaseModel):
    success: bool
    message: str

    user_id: str
    activity_id: str
    activity_type: str

    activity_energy_level: float
    current_energy_level: float
    predicted_score: float

    process: str


# ============================================================
# VERIFY WEBHOOK SECRET
# ============================================================

def verify_webhook_secret(
    received_secret: Optional[str],
) -> None:
    """
    Verify X-Webhook-Secret sent by Supabase.
    """

    if not WEBHOOK_SECRET:
        logger.error(
            "WEBHOOK_SECRET is not configured in .env"
        )

        raise HTTPException(
            status_code=500,
            detail="WEBHOOK_SECRET is not configured.",
        )

    if not received_secret:
        logger.warning(
            "Webhook rejected: missing X-Webhook-Secret"
        )

        raise HTTPException(
            status_code=401,
            detail="Missing webhook secret.",
        )

    if not secrets.compare_digest(
        received_secret,
        WEBHOOK_SECRET,
    ):
        logger.warning(
            "Webhook rejected: invalid X-Webhook-Secret"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid webhook secret.",
        )


# ============================================================
# GET ACTIVITY ENERGY
# ============================================================

def resolve_activity_energy_level(
    activity: ActivityRecord,
) -> float:
    """
    Get the activity energy level.

    Primary source:

        activity_history.energy_level

    Optional fallback:

        activity_history.metadata.energy_level
        activity_history.metadata.activity_energy_level
        activity_history.metadata.svs_value

    We do NOT invent an energy value.
    """

    # --------------------------------------------------------
    # PRIMARY SOURCE
    # --------------------------------------------------------

    if activity.energy_level is not None:

        value = float(activity.energy_level)

        logger.info(
            "Using activity_history.energy_level=%s",
            value,
        )

        return value

    # --------------------------------------------------------
    # OPTIONAL METADATA FALLBACK
    # --------------------------------------------------------

    metadata = activity.metadata or {}

    possible_keys = (
        "energy_level",
        "activity_energy_level",
        "svs_value",
    )

    for key in possible_keys:

        value = metadata.get(key)

        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if 0 <= numeric_value <= 100:

            logger.info(
                "activity_history.energy_level is NULL. "
                "Using metadata.%s=%s",
                key,
                numeric_value,
            )

            return numeric_value

    # --------------------------------------------------------
    # NO ENERGY FOUND
    # --------------------------------------------------------

    logger.error(
        "Activity %s has no activity energy value.",
        activity.id,
    )

    raise HTTPException(
        status_code=422,
        detail={
            "message": (
                "activity_history.energy_level is NULL "
                "and no usable energy value exists in metadata."
            ),
            "activity_id": str(activity.id),
            "required_column": "activity_history.energy_level",
            "expected_range": "0-100",
        },
    )


# ============================================================
# MAP SUPABASE DATA -> ML DATA
# ============================================================

def build_ml_input(
    activity: ActivityRecord,
    activity_energy_level: float,
    current_energy_level: float,
) -> Dict[str, Any]:
    """
    Explicit mapping:

    activity_history.activity_type
        -> ML activity_type

    activity_history.energy_level
        -> ML activity_energy_level

    wellness_scores.final_energy_level
        -> ML current_energy_level
    """

    return {
        "activity_type": activity.activity_type,
        "activity_energy_level": activity_energy_level,
        "current_energy_level": current_energy_level,
    }


# ============================================================
# WEBHOOK ENDPOINT
# ============================================================

@router.post(
    "/webhook/activity",
    response_model=WebhookResponse,
)
async def handle_activity_webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(
        default=None,
        alias="X-Webhook-Secret",
    ),
) -> WebhookResponse:

    # ========================================================
    # STEP 1 — VERIFY SECRET
    # ========================================================

    verify_webhook_secret(x_webhook_secret)

    # ========================================================
    # STEP 2 — READ JSON
    # ========================================================

    try:
        raw_body = await request.json()

    except Exception as exc:

        logger.exception(
            "Invalid JSON received from Supabase."
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid JSON webhook body.",
        ) from exc

    logger.info(
        "Webhook received: type=%s table=%s",
        raw_body.get("type"),
        raw_body.get("table"),
    )

    # ========================================================
    # STEP 3 — VALIDATE SUPABASE PAYLOAD
    # ========================================================

    try:
        payload = SupabaseWebhookPayload.model_validate(
            raw_body
        )

    except ValidationError as exc:

        logger.error(
            "Supabase webhook payload validation failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid Supabase webhook payload.",
                "errors": exc.errors(),
            },
        ) from exc

    activity = payload.record

    logger.info(
        "Activity received: id=%s user_id=%s "
        "activity_type=%s energy_level=%s process=%s",
        activity.id,
        activity.user_id,
        activity.activity_type,
        activity.energy_level,
        activity.process,
    )

    # ========================================================
    # STEP 4 — CHECK TABLE
    # ========================================================

    if payload.table != "activity_history":

        logger.error(
            "Wrong webhook table: %s",
            payload.table,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected 'activity_history', "
                f"received '{payload.table}'."
            ),
        )

    # ========================================================
    # STEP 5 — ONLY INSERT
    # ========================================================

    if payload.type != "INSERT":

        logger.info(
            "Ignoring non-INSERT event: %s",
            payload.type,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported event type '{payload.type}'. "
                "Only INSERT is supported."
            ),
        )

    # ========================================================
    # STEP 6 — CHECK ACTIVITY TYPE
    # ========================================================

    if activity.activity_type not in ALLOWED_ACTIVITY_TYPES:

        logger.error(
            "Invalid activity_type=%s",
            activity.activity_type,
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid activity_type.",
                "received": activity.activity_type,
                "allowed": sorted(ALLOWED_ACTIVITY_TYPES),
            },
        )

    # ========================================================
    # STEP 7 — CHECK PROCESS
    # ========================================================

    try:

        already_processed = is_activity_already_processed(
            str(activity.id)
        )

    except ActivityRecordNotFoundError as exc:

        logger.error(
            "Activity %s was not found in activity_history.",
            activity.id,
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    if already_processed:

        logger.info(
            "Activity %s already processed. Skipping.",
            activity.id,
        )

        raise HTTPException(
            status_code=409,
            detail=(
                f"Activity '{activity.id}' "
                "has already been processed."
            ),
        )

    # ========================================================
    # STEP 8 — GET ACTIVITY ENERGY
    # ========================================================

    activity_energy_level = resolve_activity_energy_level(
        activity
    )

    logger.info(
        "Activity energy level = %s",
        activity_energy_level,
    )

    # ========================================================
    # STEP 9 — GET CURRENT USER ENERGY
    # ========================================================

    """
    IMPORTANT:

    activity.user_id comes from:

        activity_history.user_id

    We use that SAME user_id to find:

        wellness_scores.user_id

    Then we read:

        wellness_scores.final_energy_level
    """

    try:

        current_energy_level = get_current_energy_level(
            str(activity.user_id)
        )

    except WellnessRecordNotFoundError as exc:

        logger.error(
            "No wellness_scores record found for user_id=%s",
            activity.user_id,
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    logger.info(
        "Current wellness score for user_id=%s = %s",
        activity.user_id,
        current_energy_level,
    )

    # ========================================================
    # STEP 10 — CREATE ML INPUT
    # ========================================================

    ml_input = build_ml_input(
        activity=activity,
        activity_energy_level=activity_energy_level,
        current_energy_level=current_energy_level,
    )

    logger.info(
        "ML input: %s",
        ml_input,
    )

    # ========================================================
    # STEP 11 — RUN ML MODEL DIRECTLY
    # ========================================================

    """
    IMPORTANT:

    /webhook/activity and /predict are inside the SAME
    FastAPI application.

    Therefore we do NOT make an HTTP request to:

        http://127.0.0.1:8000/predict

    That was causing the 10-second timeout.

    Instead, we directly call the already-loaded model
    prediction function from api.main.

    asyncio.to_thread() prevents the synchronous model
    prediction from blocking the FastAPI event loop.
    """

    try:

        # Imported here to avoid an import cycle during startup.
        from api.main import run_model_prediction

        predicted_score = await asyncio.to_thread(
            run_model_prediction,
            activity_type=ml_input["activity_type"],
            activity_energy_level=ml_input[
                "activity_energy_level"
            ],
            current_energy_level=ml_input[
                "current_energy_level"
            ],
        )

    except Exception as exc:

        logger.exception(
            "ML prediction failed for activity_id=%s",
            activity.id,
        )

        # IMPORTANT:
        #
        # Do NOT update wellness_scores.
        # Do NOT mark activity as done.
        #
        # The activity remains unprocessed.

        raise HTTPException(
            status_code=502,
            detail={
                "message": "ML prediction failed.",
                "activity_id": str(activity.id),
                "error": str(exc),
            },
        ) from exc

    logger.info(
        "ML prediction successful: "
        "activity_id=%s user_id=%s predicted_score=%s",
        activity.id,
        activity.user_id,
        predicted_score,
    )

    # ========================================================
    # STEP 12 — VALIDATE PREDICTION
    # ========================================================

    try:

        predicted_score = float(predicted_score)

    except (TypeError, ValueError) as exc:

        logger.error(
            "Invalid prediction returned by ML: %r",
            predicted_score,
        )

        raise HTTPException(
            status_code=502,
            detail="ML returned an invalid prediction.",
        ) from exc

    if not 0 <= predicted_score <= 100:

        logger.error(
            "Prediction outside 0-100: %s",
            predicted_score,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "ML prediction must be between 0 and 100."
            ),
        )

    predicted_score = round(
        predicted_score,
        2,
    )

    logger.info(
        "Predicted score for user_id=%s = %s",
        activity.user_id,
        predicted_score,
    )

    # ========================================================
    # STEP 13 — UPDATE WELLNESS SCORE
    # ========================================================

    """
    IMPORTANT:

    activity_history.user_id
            |
            v
    wellness_scores.user_id
            |
            v
    update ONLY that user's:

    wellness_scores.final_energy_level
    """

    try:

        update_user_energy_level(
            str(activity.user_id),
            predicted_score,
        )

    except WellnessRecordNotFoundError as exc:

        logger.error(
            "Could not update wellness_scores for "
            "user_id=%s: %s",
            activity.user_id,
            exc,
        )

        # Activity is NOT marked done.

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected Supabase wellness update error "
            "for user_id=%s",
            activity.user_id,
        )

        # Activity is NOT marked done.

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to update "
                "wellness_scores.final_energy_level."
            ),
        ) from exc

    logger.info(
        "UPDATED wellness_scores.final_energy_level: "
        "user_id=%s -> %s",
        activity.user_id,
        predicted_score,
    )

    # ========================================================
    # STEP 14 — MARK ACTIVITY AS DONE
    # ========================================================

    try:

        mark_activity_processed(
            str(activity.id)
        )

    except Exception as exc:

        logger.exception(
            "Could not mark activity %s as done.",
            activity.id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Wellness score was updated, "
                "but activity could not be marked as done."
            ),
        ) from exc

    logger.info(
        "UPDATED activity_history.process='done': "
        "activity_id=%s",
        activity.id,
    )

    # ========================================================
    # STEP 15 — SUCCESS RESPONSE
    # ========================================================

    return WebhookResponse(
        success=True,
        message=(
            "Activity processed successfully and "
            "wellness score updated."
        ),
        user_id=str(activity.user_id),
        activity_id=str(activity.id),
        activity_type=activity.activity_type,
        activity_energy_level=activity_energy_level,
        current_energy_level=current_energy_level,
        predicted_score=predicted_score,
        process="done",
    )