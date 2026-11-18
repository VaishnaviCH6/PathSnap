import numpy as np
import pandas as pd

from src.models.model_factory import get_models
from src.models.model_io import save_model
from src.models.prediction import predict_candidates


def test_prediction_contract_preserves_identifiers(tmp_path):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    model = get_models(random_state=42)["Logistic Regression"]
    model.fit(pd.DataFrame(X, columns=["f1", "f2"]), y)

    model_path, _ = save_model(
        model,
        {
            "hyperparameters": {"C": 1.0},
            "feature_columns": ["f1", "f2"],
            "random_seed": 42,
            "training_config": {},
            "validation_metrics": {},
            "feature_schema_version": "test-v1",
        },
        tmp_path,
        "Logistic Regression",
    )

    data = pd.DataFrame({
        "trip_id": ["a", "a", "b", "b"],
        "gps_id": [1, 2, 3, 4],
        "candidate_id": [10, 11, 20, 21],
        "arc_id": [100, 101, 102, 103],
        "f1": X[:, 0],
        "f2": X[:, 1],
    })

    output = predict_candidates(data, model_path)
    assert len(output) == len(data)
    assert set(output.columns) >= {"trip_id", "gps_id", "candidate_id", "arc_id", "predicted_label", "positive_probability"}
    assert np.all((output["positive_probability"] >= 0.0) & (output["positive_probability"] <= 1.0))
