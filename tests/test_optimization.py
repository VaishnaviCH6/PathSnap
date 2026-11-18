import numpy as np
import pandas as pd

from src.optimization.bayesian_optimization import optimize_model


def test_optuna_smoke_with_grouped_synthetic_data(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    data = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    data["label"] = y
    data["trip_id"] = np.repeat([f"trip_{i}" for i in range(12)], 10)

    result = optimize_model(
        data,
        "Logistic Regression",
        feature_columns=["f1", "f2", "f3", "f4"],
        n_trials=2,
        n_splits=3,
        random_seed=42,
        output_directory=tmp_path,
    )

    assert result["n_trials"] == 2
    assert result["n_splits"] == 3
    assert "C" in result["best_hyperparameters"]
    assert len(result["trials"]) == 2
