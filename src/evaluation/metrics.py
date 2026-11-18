from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate binary classification metrics for one evaluation fold."""
    y_true_array = np.asarray(list(y_true))
    y_pred_array = np.asarray(list(y_pred))

    if y_true_array.size == 0:
        raise ValueError("Cannot calculate metrics for an empty validation fold.")

    labels = np.union1d(y_true_array, y_pred_array)
    if not set(labels).issubset({0, 1}):
        raise ValueError("Binary metrics require labels containing only 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if y_proba is not None:
        proba_array = np.asarray(y_proba)
        if proba_array.ndim == 2:
            positive_scores = proba_array[:, 1]
        else:
            positive_scores = proba_array

        if np.unique(y_true_array).size == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_array, positive_scores))
            except ValueError:
                metrics["roc_auc"] = None

    return metrics


def calculate_candidate_ranking_metrics(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    trip_id_column: str = "trip_id",
    candidate_id_column: str = "candidate_id",
    label_column: str = "label",
) -> dict[str, Any]:
    """Calculate candidate ranking metrics when trip, candidate, and label columns exist."""
    required = {trip_id_column, candidate_id_column, label_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        return {"top1_candidate_accuracy": None, "per_trip_accuracy": {}, "status": "unavailable_missing_columns"}

    ranking = data[[trip_id_column, candidate_id_column, label_column]].copy()
    ranking["probability"] = np.asarray(probabilities).reshape(-1)

    top1_correct: list[float] = []
    per_trip_accuracy: dict[str, float] = {}
    for trip_id, trip_data in ranking.groupby(trip_id_column, sort=False):
        if trip_data[label_column].nunique() == 0:
            continue
        selected_index = trip_data["probability"].idxmax()
        top1_correct.append(float(trip_data.loc[selected_index, label_column] == 1))
        per_trip_accuracy[str(trip_id)] = float(
            np.mean(trip_data.loc[selected_index, label_column] == 1)
        )

    return {
        "top1_candidate_accuracy": float(np.mean(top1_correct)) if top1_correct else None,
        "per_trip_accuracy": per_trip_accuracy,
        "status": "calculated",
    }


def aggregate_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metrics across folds using mean and standard deviation."""
    if not fold_results:
        raise ValueError("At least one fold result is required.")

    ignored = {"confusion_matrix", "fold", "model"}
    numeric_names = {
        key
        for result in fold_results
        for key, value in result.items()
        if key not in ignored and isinstance(value, (int, float))
    }

    aggregate: dict[str, Any] = {"fold_count": len(fold_results)}
    for name in sorted(numeric_names):
        values = np.asarray(
            [result[name] for result in fold_results if result.get(name) is not None],
            dtype=float,
        )
        if values.size:
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values, ddof=0))

    aggregate["confusion_matrices"] = [
        result["confusion_matrix"]
        for result in fold_results
        if "confusion_matrix" in result
    ]
    return aggregate
