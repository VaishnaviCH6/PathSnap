from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.evaluation.cross_validation import get_group_splits, group_kfold_evaluate
from src.models.model_factory import get_models
from src.models.training import train_and_evaluate_models


@pytest.fixture(scope="module")
def synthetic_trip_data():
    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42,
    )

    data = pd.DataFrame(X, columns=[f"feature_{index}" for index in range(6)])
    data["label"] = y
    data["trip_id"] = np.repeat([f"trip_{index:02d}" for index in range(12)], 20)

    data.loc[data.index % 37 == 0, "feature_0"] = np.nan
    data["ground_truth_status"] = y
    data["candidate_id"] = np.arange(len(data))
    return data


def test_group_kfold_keeps_trips_separate(synthetic_trip_data):
    splits = get_group_splits(synthetic_trip_data, trip_id_column="trip_id", n_splits=5)

    assert len(splits) == 5

    for train_indices, validation_indices in splits:
        train_trips = set(synthetic_trip_data.iloc[train_indices]["trip_id"])
        validation_trips = set(synthetic_trip_data.iloc[validation_indices]["trip_id"])
        assert train_trips.isdisjoint(validation_trips)


def test_group_evaluation_uses_configurable_features(synthetic_trip_data):
    models = get_models(random_state=42)
    feature_columns = ["feature_0", "feature_1", "feature_2"]

    results = group_kfold_evaluate(
        synthetic_trip_data,
        models,
        feature_columns=feature_columns,
        n_splits=5,
    )

    assert set(results["models"]) == set(models)
    assert results["feature_columns"] == feature_columns

    for model_name, model_result in results["models"].items():
        assert model_result["fold_results"]
        assert "accuracy_mean" in model_result["aggregate"]
        assert "accuracy_std" in model_result["aggregate"]

        for fold in model_result["fold_results"]:
            assert fold["status"] in {"completed", "skipped_single_class_training_fold"}
            if fold["status"] == "completed":
                assert 0.0 <= fold["accuracy"] <= 1.0
                assert fold["roc_auc"] is None or 0.0 <= fold["roc_auc"] <= 1.0


def test_training_saves_csv_and_json(tmp_path, synthetic_trip_data):
    results = train_and_evaluate_models(
        synthetic_trip_data,
        feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        n_splits=5,
        random_state=42,
        output_directory=tmp_path,
    )

    json_path = Path(results["json_path"])
    csv_path = Path(results["csv_path"])

    assert json_path.exists()
    assert csv_path.exists()


def test_invalid_or_missing_trip_ids(synthetic_trip_data):
    invalid = synthetic_trip_data.copy()
    invalid.loc[0, "trip_id"] = np.nan

    with pytest.raises(ValueError, match="trip_id"):
        get_group_splits(invalid)
