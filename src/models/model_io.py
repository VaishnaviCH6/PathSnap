from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn


TRAINING_METADATA_KEYS = (
    "model_type",
    "hyperparameters",
    "feature_columns",
    "random_seed",
    "training_config",
    "validation_metrics",
    "feature_schema_version",
)


def save_model(
    model: Any,
    metadata: dict[str, Any],
    output_directory: str | Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Save a trained pipeline and JSON metadata to a dedicated artifact directory."""
    if not hasattr(model, "predict"):
        raise ValueError("Model must implement predict().")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_name = "".join(
        character.lower() if character.isalnum() else "_"
        for character in model_name
    ).strip("_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = output_path / f"{normalized_name}_{timestamp}.joblib"
    metadata_path = output_path / f"{normalized_name}_{timestamp}.json"

    complete_metadata = {
        "model_type": model_name,
        "training_timestamp_utc": timestamp,
        "sklearn_version": sklearn.__version__,
        **metadata,
    }

    descriptor, temp_name = tempfile.mkstemp(prefix="pathsnap_model_", suffix=".joblib", dir=output_path)
    os.close(descriptor)
    try:
        joblib.dump(model, temp_name)
        os.replace(temp_name, model_path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise

    metadata_path.write_text(json.dumps(complete_metadata, indent=2, default=str), encoding="utf-8")
    return model_path, metadata_path


def load_model(model_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a locally trusted Joblib artifact and its adjacent JSON metadata."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    model = joblib.load(path)
    metadata_path = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def validate_model_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return missing metadata fields needed for safe integration."""
    return [key for key in TRAINING_METADATA_KEYS if key not in metadata]


def validate_feature_frame(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Validate and reorder feature columns without mutating the source frame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Prediction input must be a Pandas DataFrame.")
    if data.empty:
        raise ValueError("Prediction input cannot be empty.")

    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = data.loc[:, feature_columns].copy()
    numeric = features.apply(pd.to_numeric, errors="coerce")
    return numeric.replace([float("inf"), float("-inf")], float("nan"))
