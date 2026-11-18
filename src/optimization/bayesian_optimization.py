from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.base import clone

from src.evaluation.metrics import calculate_binary_metrics
from src.models.model_factory import get_models
from sklearn.model_selection import GroupKFold


def optimize_model(
    data: pd.DataFrame,
    model_name: str,
    *,
    feature_columns: list[str],
    n_trials: int = 10,
    n_splits: int = 5,
    random_seed: int = 42,
    label_column: str = "label",
    trip_id_column: str = "trip_id",
    output_directory: str | Path = "experiments/results",
) -> dict[str, Any]:
    """TPE-optimize one model using group-aware cross-validation only."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if model_name not in get_models(random_state=random_seed):
        raise ValueError(f"Unknown model: {model_name}")
    if data.empty:
        raise ValueError("Training data cannot be empty.")
    missing = [column for column in [label_column, trip_id_column, *feature_columns] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[label_column], errors="coerce")
    groups = data[trip_id_column]
    if y.isna().any():
        raise ValueError("Labels contain missing or invalid values.")
    if set(y.unique()) != {0, 1}:
        raise ValueError("Bayesian optimization requires both binary label classes.")
    if groups.isna().any():
        raise ValueError("trip_id cannot contain missing values.")

    unique_trips = groups.nunique()
    if unique_trips < 2:
        raise ValueError("Bayesian optimization requires at least two distinct trips.")
    effective_splits = min(n_splits, unique_trips)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        parameters: dict[str, dict] = {}
        if model_name == "Logistic Regression":
            parameters[model_name] = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 500, 3000, step=500),
            }
        elif model_name == "KNN":
            parameters[model_name] = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            }
        elif model_name == "Decision Tree":
            parameters[model_name] = {
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
        elif model_name == "Random Forest":
            parameters[model_name] = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "n_jobs": 1,
            }
        else:
            parameters[model_name] = {
                "max_iter": trial.suggest_int("max_iter", 50, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            }

        model_template = get_models(random_seed, parameters)[model_name]
        f1_scores: list[float] = []

        splitter = GroupKFold(n_splits=effective_splits)
        for train_indices, validation_indices in splitter.split(X, y, groups):
            train_y = y.iloc[train_indices]
            if train_y.nunique() < 2:
                continue

            model = clone(model_template)
            model.fit(X.iloc[train_indices], train_y)
            predictions = model.predict(X.iloc[validation_indices])
            probabilities = model.predict_proba(X.iloc[validation_indices]) if hasattr(model, "predict_proba") else None
            metrics = calculate_binary_metrics(y.iloc[validation_indices], predictions, probabilities)
            f1_scores.append(metrics["f1"])

        if not f1_scores:
            return 0.0
        return float(sum(f1_scores) / len(f1_scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    trial_rows = [
        {
            "trial": trial.number,
            "objective_value": trial.value,
            "parameters": trial.params,
            "state": trial.state.name,
        }
        for trial in study.trials
    ]

    history_path = output_path / f"{model_name.lower().replace(' ', '_')}_optimization_history.csv"
    pd.DataFrame(
        [{"trial": row["trial"], "objective_value": row["objective_value"], **row["parameters"]}
         for row in trial_rows]
    ).to_csv(history_path, index=False)

    best_model = get_models(random_seed, {model_name: study.best_params})[model_name]
    best_model.fit(X, y)

    result = {
        "model_name": model_name,
        "sampler": "TPESampler",
        "random_seed": random_seed,
        "n_trials": n_trials,
        "n_splits": effective_splits,
        "objective": "mean_group_f1",
        "best_hyperparameters": study.best_params,
        "best_objective_value": study.best_value,
        "trials": trial_rows,
        "history_path": str(history_path),
        "model": best_model,
    }

    json_path = output_path / f"{model_name.lower().replace(' ', '_')}_optimization.json"
    json_path.write_text(json.dumps({key: value for key, value in result.items() if key != "model"}, indent=2, default=str), encoding="utf-8")
    result["json_path"] = str(json_path)
    return result
