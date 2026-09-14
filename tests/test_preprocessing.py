"""
tests/test_preprocessing.py

Tests for preprocessing/preprocess.py — the single source of truth for
feature/target names and the preprocessing pipeline shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess import (  # noqa: E402
    ALLOWED_ACTIVITY_TYPES,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    build_model_pipeline,
    build_preprocessor,
)


def test_allowed_activity_types_are_exactly_five() -> None:
    assert ALLOWED_ACTIVITY_TYPES == ["mood", "meditation", "journal", "community", "music"]


def test_feature_and_target_column_names() -> None:
    assert CATEGORICAL_COLUMNS == ["activity_type"]
    assert NUMERIC_COLUMNS == ["activity_energy_level", "current_energy_level"]
    assert FEATURE_COLUMNS == ["activity_type", "activity_energy_level", "current_energy_level"]
    assert TARGET_COLUMN == "predicted_score"


def test_preprocessor_produces_five_onehot_columns_plus_two_numeric() -> None:
    """
    One-hot encoding 5 fixed categories + 2 passthrough numeric columns
    should produce a 7-column numeric matrix, regardless of which
    categories actually appear in the input.
    """
    preprocessor = build_preprocessor()
    df = pd.DataFrame(
        {
            "activity_type": ["meditation", "mood"],
            "activity_energy_level": [80.0, 40.0],
            "current_energy_level": [55.0, 60.0],
        }
    )
    transformed = preprocessor.fit_transform(df)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    assert dense.shape == (2, 7)  # 5 one-hot columns + 2 numeric


def test_preprocessor_handles_unseen_category_without_crashing() -> None:
    """handle_unknown='ignore' should turn an unexpected category into all-zero one-hot columns."""
    preprocessor = build_preprocessor()
    train_df = pd.DataFrame(
        {
            "activity_type": ["meditation"],
            "activity_energy_level": [80.0],
            "current_energy_level": [55.0],
        }
    )
    preprocessor.fit(train_df)

    unseen_df = pd.DataFrame(
        {
            "activity_type": ["not_a_real_activity"],
            "activity_energy_level": [50.0],
            "current_energy_level": [50.0],
        }
    )
    # Should not raise.
    transformed = preprocessor.transform(unseen_df)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    assert dense.shape == (1, 7)


def test_model_pipeline_fits_and_predicts() -> None:
    """build_model_pipeline should wrap any sklearn regressor with the shared preprocessor."""
    pipeline = build_model_pipeline(LinearRegression())
    X = pd.DataFrame(
        {
            "activity_type": ["mood", "meditation", "journal", "community", "music"],
            "activity_energy_level": [50, 80, 60, 65, 70],
            "current_energy_level": [40, 55, 50, 45, 60],
        }
    )
    y = pd.Series([45, 63, 55, 58, 66])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)
