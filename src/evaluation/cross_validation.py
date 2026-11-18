from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import aggregate_metrics, calculate_binary_metrics


DEFAULT_EXCLUDED_COLUMNS = {
    "trip_id",
    "label",
    "candidate_label",
    "is_correct",
    "ground_truth",
}


def resolve_feature_columns(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "label",
    trip_id_column: str = "trip_id",
) -> list[str]:
    """Resolve predictive columns while excluding identifiers and labels."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a Pandas DataFrame.")
    if feature_columns is not None:
        return list(feature_columns)

    excluded = DEFAULT_EXCLUDED_COLUMNS | {
        label_column,
        trip_id_column,
        "gps_id",
        "candidate_id",
        "arc_id",
        "node_id",
        "segment_id",
    }
    excluded.update(
        column
        for column in data.columns
        if "ground_truth" in str(column).lower()
        or "true_label" in str(column).lower()
        or "identifier" in str(column).lower()
    )
    return [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def get_group_splits(
    data: pd.DataFrame,
    trip_id_column: str = "trip_id",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return GroupKFold indices so all observations from a trip stay together."""
    if trip_id_column not in data.columns:
        raise ValueError(f"Missing required grouping column: {trip_id_column}")

    groups = data[trip_id_column]
    if groups.isna().any():
        raise ValueError("trip_id cannot contain missing values.")

    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Group-aware cross-validation requires at least two distinct trip IDs.")

    effective_splits = min(n_splits, unique_groups)
    if effective_splits < 2:
        raise ValueError("At least two folds are required.")

    splitter = GroupKFold(n_splits=effective_splits)
    return list(splitter.split(data, groups=groups))


def group_kfold_evaluate(
    data: pd.DataFrame,
    models: Mapping[str, Any],
    *,
    feature_columns: list[str] | None = None,
    label_column: str = "label",
    trip_id_column: str = "trip_id",
    n_splits: int = 5,
) -> dict[str, Any]:
    """Evaluate models with preprocessing refit inside every GroupKFold training fold."""
    if label_column not in data.columns:
        raise ValueError(f"Missing required binary label column: {label_column}")

    features = resolve_feature_columns(data, feature_columns, label_column, trip_id_column)
    if not features:
        raise ValueError("No predictive feature columns were found.")

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    raw_features = data[features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(raw_features.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Feature columns contain infinite values.")
    X = raw_features.replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(data[label_column], errors="coerce")
    if y.isna().any():
        raise ValueError("Label column contains missing or invalid values.")

    unique_labels = set(y.unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("Label values must be binary 0 and 1.")
    if len(unique_labels) != 2:
        raise ValueError("The complete dataset must contain both binary classes.")

    splits = get_group_splits(data, trip_id_column, n_splits)
    results: dict[str, Any] = {"feature_columns": features, "models": {}, "fold_groups": []}

    for model_name, model_template in models.items():
        fold_results: list[dict[str, Any]] = []

        for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
            train_labels = y.iloc[train_indices]
            validation_labels = y.iloc[validation_indices]

            train_trips = set(data.iloc[train_indices][trip_id_column].tolist())
            validation_trips = set(data.iloc[validation_indices][trip_id_column].tolist())
            overlap = train_trips & validation_trips
            if overlap:
                raise RuntimeError("Trip leakage detected: a trip is present in train and validation folds.")

            fold_record: dict[str, Any] = {
                "fold": fold_number,
                "model": model_name,
                "train_trip_count": len(train_trips),
                "validation_trip_count": len(validation_trips),
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
            }

            if train_labels.nunique() < 2:
                fold_record.update(
                    {
                        "status": "skipped_single_class_training_fold",
                        "training_time_seconds": 0.0,
                        "inference_time_seconds": 0.0,
                    }
                )
                fold_results.append(fold_record)
                continue

            model = clone(model_template)
            fit_start = perf_counter()
            model.fit(X.iloc[train_indices], train_labels.to_numpy())
            training_time = perf_counter() - fit_start

            inference_start = perf_counter()
            predictions = model.predict(X.iloc[validation_indices])
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X.iloc[validation_indices])
            inference_time = perf_counter() - inference_start

            metrics = calculate_binary_metrics(
                validation_labels,
                predictions,
                probabilities,
            )
            metrics.update(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "status": "completed",
                    "training_time_seconds": float(training_time),
                    "inference_time_seconds": float(inference_time),
                }
            )
            fold_results.append(metrics)

        completed = [result for result in fold_results if result.get("status", "completed") == "completed"]
        if not completed:
            raise RuntimeError(f"No usable folds for model '{model_name}'.")

        results["models"][model_name] = {
            "fold_results": fold_results,
            "aggregate": aggregate_metrics(completed),
        }

    if not results["fold_groups"]:
        for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
            results["fold_groups"].append(
                {
                    "fold": fold_number,
                    "train_trip_ids": sorted(data.iloc[train_indices][trip_id_column].unique().tolist()),
                    "validation_trip_ids": sorted(data.iloc[validation_indices][trip_id_column].unique().tolist()),
                }
            )

    return results
