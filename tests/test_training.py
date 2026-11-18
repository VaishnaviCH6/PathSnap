import numpy as np
import pandas as pd
import pytest

from src.models.training import train_and_evaluate_models


def make_training_data(rows=120, trips=12):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(rows, 4))
    y = ((X[:, 0] + X[:, 1] + rng.normal(scale=0.4, size=rows)) > 0).astype(int)
    data = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    data["label"] = y
    data["trip_id"] = np.repeat([f"trip_{i}" for i in range(trips)], rows // trips)
    return data


def test_training_rejects_empty_and_missing_columns():
    with pytest.raises(ValueError):
        train_and_evaluate_models(pd.DataFrame(), feature_columns=["f1"])

    data = make_training_data()
    with pytest.raises(ValueError):
        train_and_evaluate_models(data, feature_columns=["missing_feature"])


def test_training_rejects_infinite_features():
    data = make_training_data()
    data.loc[0, "f1"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        train_and_evaluate_models(data, feature_columns=["f1", "f2", "f3", "f4"])


def test_training_rejects_invalid_labels():
    data = make_training_data()
    data.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="binary"):
        train_and_evaluate_models(data, feature_columns=["f1", "f2", "f3", "f4"])
