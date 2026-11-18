from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_factory import get_models
from src.models.model_io import load_model, save_model, validate_feature_frame


def test_save_load_and_reproduce_predictions(tmp_path):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    model = get_models(random_state=42)["Logistic Regression"]
    model.fit(pd.DataFrame(X, columns=["feature_a", "feature_b"]), y)

    model_path, metadata_path = save_model(
        model,
        {
            "hyperparameters": {"C": 1.0},
            "feature_columns": ["feature_a", "feature_b"],
            "random_seed": 42,
            "training_config": {"cv": "GroupKFold"},
            "validation_metrics": {"f1": 1.0},
            "feature_schema_version": "test-v1",
        },
        tmp_path,
        "Logistic Regression",
    )

    loaded, metadata = load_model(model_path)
    input_frame = pd.DataFrame(X, columns=["feature_a", "feature_b"])
    features = validate_feature_frame(input_frame, ["feature_a", "feature_b"])

    np.testing.assert_array_equal(model.predict(features), loaded.predict(features))
    assert metadata_path.is_file()
    assert metadata["sklearn_version"]
