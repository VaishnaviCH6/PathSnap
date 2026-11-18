import json
from pathlib import Path

import pandas as pd

from src.models.model_io import load_model


RESULTS = Path("experiments/results")
ARTIFACT = Path("artifacts/random_forest_tuned_20261118T123057Z.joblib")


def test_persistent_optuna_evidence_is_complete():
    trial_csv = RESULTS / "optuna_trials.csv"
    tuning_json = RESULTS / "random_forest_optimization.json"

    assert trial_csv.is_file()
    assert tuning_json.is_file()

    trials = pd.read_csv(trial_csv)
    study = json.loads(tuning_json.read_text(encoding="utf-8"))

    assert len(trials) >= 6
    assert study["completed_trials"] == 6
    assert study["subset_trips"] == 12
    assert study["subset_rows"] == 75_678
    assert study["n_splits"] == 3
    assert study["best_hyperparameters"]["n_estimators"] == 60


def test_tuned_vs_baseline_subset_evidence_is_comparable():
    comparison = pd.read_csv(RESULTS / "tuned_vs_baseline_subset.csv")

    assert set(comparison["configuration"]) == {"baseline", "tuned"}
    assert comparison.groupby("configuration")["fold"].count().to_dict() == {
        "baseline": 3,
        "tuned": 3,
    }

    baseline_f1 = comparison.loc[comparison["configuration"] == "baseline", "f1"].mean()
    tuned_f1 = comparison.loc[comparison["configuration"] == "tuned", "f1"].mean()
    assert tuned_f1 > baseline_f1


def test_tuned_artifact_reloads_and_has_feature_metadata():
    assert ARTIFACT.is_file()
    model, metadata = load_model(ARTIFACT)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert len(metadata["feature_columns"]) == 10
    assert metadata["hyperparameters"]["n_estimators"] == 60
