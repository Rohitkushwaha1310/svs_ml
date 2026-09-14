"""
api/supabase_client.py

The ONLY module in this project that talks to Supabase. It:

    - reads connection details from environment variables (never hard-coded)
    - exposes narrow, purpose-built functions instead of leaking the raw
      Supabase client everywhere
    - always scopes reads/writes to a single user_id or a single activity
      id — never a global update

Required environment variables (see .env.example):

    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY

The service-role key is used because this is a trusted backend service. It
must NEVER be shipped to any frontend / mobile client.

--------------------------------------------------------------------------
SCHEMA (fixed — source of truth, columns are NOT renamed by this project)
--------------------------------------------------------------------------
activity_history:
    id                uuid
    user_id           uuid
    activity_type     text
    title             text
    subtitle          text nullable
    metadata          jsonb
    created_at        timestamptz
    energy_level      int4 nullable   <- the ACTIVITY's energy/SVS value
    process           text nullable   <- NULL = not yet handled, "done" = processed

wellness_scores:
    id                  uuid
    user_id             uuid
    final_energy_level  int4          <- current energy level (read before
                                          predicting) AND where the new
                                          prediction is written (after)
    breakdown           jsonb
    computed_at         timestamptz
    created_at          timestamptz

Mapping used throughout this project:

    activity_history.energy_level        -> ML activity_energy_level
    wellness_scores.final_energy_level   -> ML current_energy_level (read BEFORE predicting)
    ML predicted_score                   -> wellness_scores.final_energy_level (write AFTER predicting)
    activity_history.process             -> set to "done" after a successful prediction
--------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

import os

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

ACTIVITY_TABLE = "activity_history"
ACTIVITY_PROCESS_COLUMN = "process"
ACTIVITY_PROCESSED_VALUE = "done"

WELLNESS_TABLE = "wellness_scores"
WELLNESS_ENERGY_COLUMN = "final_energy_level"


class SupabaseConfigError(RuntimeError):
    """Raised when required Supabase environment variables are missing."""


class WellnessRecordNotFoundError(RuntimeError):
    """Raised when no wellness_scores row exists for the given user_id."""


class ActivityRecordNotFoundError(RuntimeError):
    """Raised when no activity_history row exists for the given activity id."""


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Build (and cache) a single Supabase client for the process lifetime.
    Uses service-role credentials — backend-only, never bundled into
    frontend code.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
            "(see .env.example). Refusing to start without them."
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ---------------------------------------------------------------------------
# wellness_scores
# ---------------------------------------------------------------------------
def get_current_energy_level(user_id: str) -> float:
    """
    Read the user's CURRENT energy level from
    wellness_scores.final_energy_level, scoped strictly by user_id.

    Raises WellnessRecordNotFoundError if no row exists for that user.
    """
    client = get_supabase_client()

    response = (
        client.table(WELLNESS_TABLE)
        .select(f"user_id, {WELLNESS_ENERGY_COLUMN}")
        .eq("user_id", user_id)
        .order("computed_at", desc=True)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise WellnessRecordNotFoundError(
            f"No wellness_scores record found for user_id='{user_id}'."
        )

    current_value = rows[0].get(WELLNESS_ENERGY_COLUMN)
    if current_value is None:
        raise WellnessRecordNotFoundError(
            f"wellness_scores record for user_id='{user_id}' has no value in "
            f"column '{WELLNESS_ENERGY_COLUMN}'."
        )

    return float(current_value)


def update_user_energy_level(user_id: str, new_energy_level: float) -> Dict[str, Any]:
    """
    Update ONLY the wellness_scores row(s) belonging to `user_id` with the
    new predicted score, writing into final_energy_level (never a new
    column). The `.eq("user_id", user_id)` filter is what guarantees this
    never touches another user's row and never performs a global update.
    """
    client = get_supabase_client()

    response = (
        client.table(WELLNESS_TABLE)
        .update({WELLNESS_ENERGY_COLUMN: new_energy_level})
        .eq("user_id", user_id)
        .execute()
    )

    updated_rows = response.data or []
    if not updated_rows:
        raise WellnessRecordNotFoundError(
            f"Update affected 0 rows for user_id='{user_id}'. "
            "The wellness_scores record may have been deleted concurrently."
        )

    return updated_rows[0]


# ---------------------------------------------------------------------------
# activity_history.process — idempotency using the existing column
# ---------------------------------------------------------------------------
def is_activity_already_processed(activity_id: str) -> bool:
    """
    Check the CURRENT value of activity_history.process for this activity,
    read fresh from the database — never trusted from the webhook payload,
    since a redelivered webhook still carries the row exactly as it looked
    at INSERT time (process=NULL), even after we've since marked it "done".

    Returns True if `process` is anything other than NULL.
    Raises ActivityRecordNotFoundError if the activity id does not exist.
    """
    client = get_supabase_client()

    response = (
        client.table(ACTIVITY_TABLE)
        .select(f"id, {ACTIVITY_PROCESS_COLUMN}")
        .eq("id", activity_id)
        .limit(1)
        .execute()
    )

    rows = response.data or []
    if not rows:
        raise ActivityRecordNotFoundError(
            f"No activity_history record found for id='{activity_id}'."
        )

    return rows[0].get(ACTIVITY_PROCESS_COLUMN) is not None


def mark_activity_processed(activity_id: str) -> None:
    """
    Set activity_history.process = "done" for this activity, scoped
    strictly by activity id.
    """
    client = get_supabase_client()
    (
        client.table(ACTIVITY_TABLE)
        .update({ACTIVITY_PROCESS_COLUMN: ACTIVITY_PROCESSED_VALUE})
        .eq("id", activity_id)
        .execute()
    )


def get_activity_by_id(activity_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the full activity_history row for a given id. Used as a fallback
    if a webhook payload ever arrives without the full row embedded (some
    webhook configurations only send a subset of columns).
    """
    client = get_supabase_client()
    response = (
        client.table(ACTIVITY_TABLE)
        .select("id, user_id, activity_type, energy_level, metadata, process, created_at")
        .eq("id", activity_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None
