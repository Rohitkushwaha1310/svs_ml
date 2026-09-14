"""
training/evaluate.py

Standalone evaluation script. Loads the already-trained pipeline and the
saved held-out test split, computes MAE/RMSE/R2, and prints example rows.
Does NOT retrain anything.

Run with:

    python training/evaluate.py

Or against a different held-out CSV (same columns as the training set):

    python training/evaluate.py path/to/other_test_set.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "svs_model.pkl"
DEFAULT_TEST_SPLIT_PATH = PROJECT_ROOT / "models" / "test_split.csv"
NUM_EXAMPLES_TO_SHOW = 10


def load_pipeline(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. Run 'python training/train.py' first."
        )
    return joblib.load(model_path)


def load_test_data(test_path: Path) -> pd.DataFrame:
    if not test_path.exists():
        raise FileNotFoundError(
            f"No test split found at '{test_path}'. Run 'python training/train.py' first "
            "to generate it, or pass a different CSV path as a command-line argument."
        )
    return pd.read_csv(test_path)


def clip_and_round_predictions(predictions: np.ndarray, decimals: int = 2) -> np.ndarray:
    clipped = np.clip(predictions, 0.0, 100.0)
    return np.round(clipped, decimals)


def main() -> None:
    test_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEST_SPLIT_PATH

    print("=" * 70)
    print("SVS ML - Evaluation")
    print("=" * 70)

    pipeline = load_pipeline(MODEL_PATH)
    print(f"Loaded trained pipeline from: {MODEL_PATH}")

    test_df = load_test_data(test_path)
    print(f"Loaded {len(test_df)} held-out test rows from: {test_path}")

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    predictions = clip_and_round_predictions(pipeline.predict(X_test))

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nTest set performance:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2:   {r2:.3f}")

    print(f"\nExample predictions (up to {NUM_EXAMPLES_TO_SHOW} rows):")
    print(f"{'Actual':>10} | {'Predicted':>10} | {'Error':>10}")
    print("-" * 36)

    sample_size = min(NUM_EXAMPLES_TO_SHOW, len(test_df))
    sample_indices = np.random.RandomState(42).choice(len(test_df), size=sample_size, replace=False)
    for i in sample_indices:
        actual = y_test[i]
        predicted = predictions[i]
        error = predicted - actual
        print(f"{actual:>10.2f} | {predicted:>10.2f} | {error:>10.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
