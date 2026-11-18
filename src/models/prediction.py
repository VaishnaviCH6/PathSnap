from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.model_io import load_model, validate_feature_frame


IDENTIFIER_COLUMNS = (
    "trip_id",
    "gps_id",
    "candidate_id",
    "arc_id",
)


def predict_candidates(
    data: pd.DataFrame,
    model_path: str | Path,
) -> pd.DataFrame:
    """Return labels and positive-class probabilities while preserving identifier columns."""
    model, metadata = load_model(model_path)
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Model metadata does not contain a valid feature column list.")

    features = validate_feature_frame(data, feature_columns)
    predictions = np.asarray(model.predict(features))

    probabilities = None
    if hasattr(model, "predict_proba"):
        probability_matrix = np.asarray(model.predict_proba(features))
        classes = np.asarray(model.classes_ if hasattr(model, "classes_") else [])
        if hasattr(model, "classes_"):
            class_values = classes
        elif hasattr(model, "named_steps"):
            final_estimator = model.steps[-1][1]
            class_values = getattr(final_estimator, "classes_", np.array([0, 1]))
        else:
            class_values = np.array([0, 1])

        if 1 in class_values:
            positive_index = int(np.flatnonzero(class_values == 1)[0])
            probabilities = probability_matrix[:, positive_index]
        else:
            probabilities = np.zeros(len(features), dtype=float)

    if probabilities is None:
        probabilities = predictions.astype(float)

    if len(predictions) != len(data):
        raise RuntimeError("Prediction length does not match input row count.")

    identifiers = [column for column in IDENTIFIER_COLUMNS if column in data.columns]
    output = data.loc[:, identifiers].copy() if identifiers else pd.DataFrame(index=data.index)
    output["predicted_label"] = predictions.astype(int)
    output["positive_probability"] = probabilities.astype(float)
    output["model_type"] = metadata.get("model_type")
    return output
