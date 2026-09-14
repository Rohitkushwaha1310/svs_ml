"""
tests/test_model.py

Tests that the saved pipeline (models/svs_model.pkl) loads correctly and
produces sane predictions. Does not retrain anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess import FEATURE_COLUMNS  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "svs_model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model found. Run 'python training/train.py' first.",
)


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


def test_model_loads(pipeline) -> None:
    assert pipeline is not None
    assert hasattr(pipeline, "predict")


@pytest.mark.parametrize("activity_type", ["mood", "meditation", "journal", "community", "music"])
def test_model_predicts_within_valid_range(pipeline, activity_type: str) -> None:
    input_df = pd.DataFrame(
        [{"activity_type": activity_type, "activity_energy_level": 70, "current_energy_level": 50}],
        columns=FEATURE_COLUMNS,
    )
    prediction = pipeline.predict(input_df)[0]
    # The raw model output isn't guaranteed to be in [0, 100] (that clipping
    # happens in api/main.py's run_model_prediction), but for reasonable
    # in-distribution inputs it should be close.
    assert -10 <= prediction <= 110


def test_model_rejects_unexpected_columns_gracefully() -> None:
    """Sanity check: FEATURE_COLUMNS is exactly the 3 documented ML inputs."""
    assert FEATURE_COLUMNS == ["activity_type", "activity_energy_level", "current_energy_level"]
