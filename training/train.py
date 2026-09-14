"""
training/train.py

Full training pipeline for the SVS ML energy-score predictor.

Run with:

    python training/train.py

Steps:

    1. Load the raw CSV dataset (data/training_dataset.csv).
    2. Validate it (columns, ranges, categories, duplicates, missing values).
    3. Split USERS (not rows) into train / validation / test groups, so the
       same user never appears in more than one split.
    4. Build X (features) / y (target) for each split.
    5. Train several candidate regression models on the training split.
    6. Evaluate every candidate on the validation split and pick the best
       one (lowest validation MAE).
    7. Retrain the winning model type on train + validation combined.
    8. Evaluate that final model exactly once on the untouched test split.
    9. Save the fitted pipeline (preprocessing + model) to
       models/svs_model.pkl.
   10. Save a JSON performance report to models/model_metrics.json.
   11. Save the raw test split rows to models/test_split.csv so that
       training/evaluate.py can evaluate the saved model later without
       retraining anything and without re-doing the random split.

--------------------------------------------------------------------------
IMPORTANT: do not train on self-generated predictions
--------------------------------------------------------------------------
The target column, `predicted_score`, must represent a REAL, OBSERVED
post-activity energy score from historical data — never a value this same
model (or any model) previously predicted. Training on a model's own
predictions creates a feedback loop that has nothing to do with real user
outcomes and will silently degrade in ways that are hard to detect.

If your organization does not yet have enough historical
(pre-activity, activity, post-activity) triples to build a legitimate
training set, do NOT fabricate labels to fill the gap. See
`data/README.md` for how this project's example dataset was built and what
you need to replace it with before training on real user data.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess import (  # noqa: E402
    ALLOWED_ACTIVITY_TYPES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model_pipeline,
)

try:
    from xgboost import XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DATA_PATH = PROJECT_ROOT / "data" / "training_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "svs_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
TEST_SPLIT_PATH = MODELS_DIR / "test_split.csv"

REQUIRED_COLUMNS = [
    "user_id",
    "activity_type",
    "activity_energy_level",
    "current_energy_level",
    TARGET_COLUMN,
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1 & 2. Load + validate
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{csv_path}'. "
            "See data/README.md for the required format."
        )
    return pd.read_csv(csv_path)


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the raw dataset before any training happens. Raises a
    ValueError with a clear message on the first problem found — fail
    loudly rather than silently training on bad data.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        null_counts = df[REQUIRED_COLUMNS].isnull().sum()
        offending = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Dataset contains missing values in columns: {offending}. "
            "Please clean the dataset before training."
        )

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} exact duplicate row(s). "
            "Please remove duplicates before training."
        )

    invalid_activity_types = set(df["activity_type"].unique()) - set(
        ALLOWED_ACTIVITY_TYPES
    )
    if invalid_activity_types:
        raise ValueError(
            f"Dataset contains invalid activity_type value(s): {invalid_activity_types}. "
            f"Allowed values are exactly: {ALLOWED_ACTIVITY_TYPES}"
        )

    numeric_range_columns = ["activity_energy_level", "current_energy_level", TARGET_COLUMN]
    for column in numeric_range_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric but has non-numeric values."
            )
        out_of_range = df[(df[column] < 0) | (df[column] > 100)]
        if len(out_of_range) > 0:
            raise ValueError(
                f"Column '{column}' contains {len(out_of_range)} value(s) "
                "outside the valid range [0, 100]."
            )

    print("Dataset validation passed: columns, ranges, categories all OK.")


# ---------------------------------------------------------------------------
# 3. Grouped user-level train / validation / test split
# ---------------------------------------------------------------------------
def grouped_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by user_id so no user's rows appear in more than one split.

    Why: the same user_id can appear many times (once per activity). A
    random ROW split could put some of a user's rows in training and
    others in test, letting the model partly memorize that user's personal
    energy baseline — inflating evaluation metrics and hiding how well the
    model generalizes to a brand-new user.
    """
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be less than 1.0")

    groups = df["user_id"].values

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    train_val_idx, test_idx = next(splitter_test.split(df, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_frac = val_frac / (train_frac + val_frac)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=random_state)
    train_idx, val_idx = next(
        splitter_val.split(train_val_df, groups=train_val_df["user_id"].values)
    )
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_users = set(train_df["user_id"])
    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])
    assert not (train_users & val_users), "Leakage: users shared between train and val!"
    assert not (train_users & test_users), "Leakage: users shared between train and test!"
    assert not (val_users & test_users), "Leakage: users shared between val and test!"

    print(
        f"User split -> train: {len(train_users)} users / {len(train_df)} rows, "
        f"val: {len(val_users)} users / {len(val_df)} rows, "
        f"test: {len(test_users)} users / {len(test_df)} rows"
    )

    return train_df, val_df, test_df


def make_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def get_candidate_models() -> Dict[str, object]:
    candidates: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        candidates["XGBRegressor"] = XGBRegressor(
            n_estimators=300, random_state=RANDOM_STATE, objective="reg:squarederror"
        )
    else:
        print(
            "Note: XGBoost is not installed, so XGBRegressor will be skipped. "
            "Install the optional 'xgboost' package to include it."
        )
    return candidates


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def clip_and_round_predictions(predictions: np.ndarray, decimals: int = 2) -> np.ndarray:
    clipped = np.clip(predictions, 0.0, 100.0)
    return np.round(clipped, decimals)


def main() -> None:
    print("=" * 70)
    print("SVS ML - Training pipeline")
    print("=" * 70)

    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    validate_dataset(df)

    train_df, val_df, test_df = grouped_train_val_test_split(df)

    X_train, y_train = make_features_and_target(train_df)
    X_val, y_val = make_features_and_target(val_df)
    X_test, y_test = make_features_and_target(test_df)

    candidates = get_candidate_models()
    validation_results: Dict[str, Dict[str, float]] = {}

    print("\nTraining candidate models...")
    for name, regressor in candidates.items():
        pipeline = build_model_pipeline(regressor)
        pipeline.fit(X_train, y_train)

        val_predictions = clip_and_round_predictions(pipeline.predict(X_val))
        metrics = compute_regression_metrics(y_val.values, val_predictions)
        validation_results[name] = metrics

        print(
            f"  {name:<28} val MAE={metrics['MAE']:.3f}  "
            f"RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}"
        )

    best_model_name = min(validation_results, key=lambda name: validation_results[name]["MAE"])
    print(f"\nBest model on validation set: {best_model_name}")

    # Retrain the winning model TYPE on train + validation combined, using a
    # brand-new unfitted instance (never reuse an already-fitted model), so
    # preprocessing is refit correctly on the combined data too.
    best_regressor_class = type(candidates[best_model_name])
    best_regressor_params = candidates[best_model_name].get_params()
    final_regressor = best_regressor_class(**best_regressor_params)
    final_pipeline = build_model_pipeline(final_regressor)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val, y_train_val = make_features_and_target(train_val_df)
    final_pipeline.fit(X_train_val, y_train_val)

    test_predictions = clip_and_round_predictions(final_pipeline.predict(X_test))
    test_metrics = compute_regression_metrics(y_test.values, test_predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

    report = {
        "best_model": best_model_name,
        "validation_results_all_models": validation_results,
        "final_model_validation_metrics": validation_results[best_model_name],
        "final_model_test_metrics": test_metrics,
        "train_users": train_df["user_id"].nunique(),
        "val_users": val_df["user_id"].nunique(),
        "test_users": test_df["user_id"].nunique(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved performance report to: {METRICS_PATH}")

    test_df.to_csv(TEST_SPLIT_PATH, index=False)
    print(f"Saved held-out test split to: {TEST_SPLIT_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Best model: {best_model_name}\n")
    val_metrics = validation_results[best_model_name]
    print("Validation:")
    print(f"  MAE:  {val_metrics['MAE']:.3f}")
    print(f"  RMSE: {val_metrics['RMSE']:.3f}")
    print(f"  R2:   {val_metrics['R2']:.3f}")
    print("\nFinal unseen test:")
    print(f"  MAE:  {test_metrics['MAE']:.3f}")
    print(f"  RMSE: {test_metrics['RMSE']:.3f}")
    print(f"  R2:   {test_metrics['R2']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
