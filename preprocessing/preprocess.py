"""
preprocessing/preprocess.py

SINGLE SOURCE OF TRUTH for feature/target names and preprocessing logic.

Every other part of the project (training, evaluation, the FastAPI service,
and the webhook integration) imports its column names and category list
from HERE, and nowhere else. This guarantees the preprocessing pipeline,
the training script, and the API can never drift into incompatible
schemas.

--------------------------------------------------------------------------
NAMING NOTE (important, read this before renaming anything)
--------------------------------------------------------------------------
This project uses two different names for very similar concepts, and that
is INTENTIONAL, not an inconsistency:

    - `predicted_score` is the ML model's output / training target name.
      It is a pure ML concept and does NOT exist as a Supabase column.
    - `wellness_scores.final_energy_level` is the Supabase column the
      backend writes the model's `predicted_score` value into, after the
      HTTP response comes back from the ML API.

Do not rename `predicted_score` to `final_energy_level` anywhere in the ML
code, and do not rename the Supabase column to `predicted_score`. The
mapping between them happens explicitly, once, in the webhook handler
(`api/main.py`) — never implicitly.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Column definitions — the single source of truth
# ---------------------------------------------------------------------------

# The five allowed activity types. Used both for dataset validation and to
# tell OneHotEncoder the exact set of categories to expect, so a category
# that never appears in a given training split still gets a column at
# inference time.
ALLOWED_ACTIVITY_TYPES: List[str] = [
    "mood",
    "meditation",
    "journal",
    "community",
    "music",
]

CATEGORICAL_COLUMNS: List[str] = ["activity_type"]
NUMERIC_COLUMNS: List[str] = ["activity_energy_level", "current_energy_level"]

FEATURE_COLUMNS: List[str] = [
    "activity_type",
    "activity_energy_level",
    "current_energy_level",
]

# The ML training target / model output name. This is a model concept, not
# a Supabase column — see the naming note above.
TARGET_COLUMN: str = "predicted_score"


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer that converts raw feature columns into a
    numeric matrix suitable for scikit-learn regressors.

    - `activity_type` is one-hot encoded with an explicit, fixed category
      list (`categories=[ALLOWED_ACTIVITY_TYPES]`) so the encoder always
      produces the same five columns in the same order, regardless of
      which activity types happen to appear in a given training split.
      `handle_unknown="ignore"` means an unexpected category at inference
      time (which should never happen because the API validates input
      first) results in all-zero columns instead of a crash.

    - Numeric columns pass through unchanged. Every candidate model in this
      project is tree-based (Random Forest, Gradient Boosting, Extra Trees)
      or a simple Linear Regression over only two numeric + one
      one-hot-encoded feature, so scaling adds complexity without real
      benefit.
    """
    categorical_transformer = OneHotEncoder(
        categories=[ALLOWED_ACTIVITY_TYPES],
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("activity_type_encoder", categorical_transformer, CATEGORICAL_COLUMNS),
            ("numeric_passthrough", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    return preprocessor


def build_model_pipeline(regressor: RegressorMixin | BaseEstimator) -> Pipeline:
    """
    Wrap any scikit-learn-compatible regressor together with the shared
    preprocessing step into a single Pipeline.

    Training calls `pipeline.fit(X_train, y_train)`; inference calls
    `pipeline.predict(X_new)`. The exact same preprocessing (fitted on
    training data only) is applied in both cases — this whole Pipeline
    object is what gets saved to `models/svs_model.pkl`.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )
    return pipeline
