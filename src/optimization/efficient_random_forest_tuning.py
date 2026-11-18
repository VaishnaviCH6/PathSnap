from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import calculate_binary_metrics
from src.models.model_factory import get_models


FEATURE_COLUMNS = (
    "distance_to_segment_m",
    "direction_difference_deg",
    "gps_step_distance_m",
    "time_difference_s",
    "gps_speed_mps",
    "segment_length_m",
    "projection_fraction",
    "gps_heading_deg",
    "road_bearing_deg",
    "gps_heading_change_deg",
)


def select_representative_trips(
    data: pd.DataFrame,
    n_trips: int = 12,
    random_seed: int = 42,
) -> list[str]:
    """Select trips deterministically across row-count and positive-label strata."""
    if "trip_id" not in data or "label" not in data:
        raise ValueError("Data must contain trip_id and label columns.")

    stats = data.groupby("trip_id", sort=False).agg(
        row_count=("label", "size"),
        positive_count=("label", "sum"),
    ).reset_index()
    stats["positive_proportion"] = stats["positive_count"] / stats["row_count"]
    stats["row_decile"] = pd.qcut(stats["row_count"].rank(method="first"), 4, labels=False, duplicates="drop")
    stats["positive_decile"] = pd.qcut(stats["positive_proportion"].rank(method="first"), 4, labels=False, duplicates="drop")

    rng = __import__("numpy").random.default_rng(random_seed)
    selected: list[str] = []
    strata = stats.groupby(["row_decile", "positive_decile"], sort=False)
    stratum_frames = [frame.reset_index(drop=True) for _, frame in strata]

    while len(selected) < n_trips and stratum_frames:
        rng.shuffle(stratum_frames)
        for frame in stratum_frames:
            remaining = frame.loc[~frame["trip_id"].isin(selected)]
            if remaining.empty:
                continue
            candidate = remaining.iloc[int(rng.integers(len(remaining)))]["trip_id"]
            selected.append(str(candidate))
            if len(selected) >= n_trips:
                break

    if len(selected) < n_trips:
        raise ValueError("Unable to select the requested number of representative trips.")
    return selected


def run_random_forest_tuning(
    candidate_csv: str | Path,
    *,
    n_subset_trips: int = 12,
    n_splits: int = 3,
    n_trials: int = 4,
    random_seed: int = 42,
    pilot: bool = False,
    output_directory: str | Path = "experiments/results",
) -> dict[str, Any]:
    """Run bounded, persistent Random Forest tuning on a representative trip subset."""
    use_columns = ["trip_id", "label", *FEATURE_COLUMNS]
    data = pd.read_csv(
        candidate_csv,
        usecols=use_columns,
        dtype={"trip_id": str},
    )
    data = data[data["label"].isin([0, 1])].copy()

    selected_trips = select_representative_trips(data, n_subset_trips, random_seed)
    subset = data[data["trip_id"].isin(selected_trips)].copy()
    for column in FEATURE_COLUMNS:
        subset[column] = pd.to_numeric(subset[column], errors="coerce").astype("float32")

    X = subset[list(FEATURE_COLUMNS)]
    y = subset["label"].astype("int8")
    groups = subset["trip_id"]

    effective_splits = min(n_splits, subset["trip_id"].nunique())
    if effective_splits < 2:
        raise ValueError("Representative tuning subset needs at least two trips/folds.")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_path / "random_forest_optuna.db"
    storage = f"sqlite:///{sqlite_path.as_posix()}"
    study_name = "random_forest_tuning_pilot" if pilot else "random_forest_tuning"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        parameters = {
            "n_estimators": 40 if pilot else trial.suggest_int("n_estimators", 40, 120, step=20),
            "max_depth": 12 if pilot else trial.suggest_categorical("max_depth", [8, 12, 16, 20, None]),
            "min_samples_split": 2 if pilot else trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": 1 if pilot else trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": "sqrt" if pilot else trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": "balanced" if pilot else trial.suggest_categorical("class_weight", [None, "balanced"]),
            "n_jobs": 1,
            "random_state": random_seed,
        }

        scores: list[dict[str, Any]] = []
        splitter = GroupKFold(n_splits=effective_splits)
        for fold_number, (train_indices, validation_indices) in enumerate(splitter.split(X, y, groups), start=1):
            train_y = y.iloc[train_indices]
            if train_y.nunique() < 2:
                continue

            model_template = get_models(random_state=random_seed, model_parameters={"Random Forest": parameters})["Random Forest"]
            model = clone(model_template)
            model.fit(X.iloc[train_indices], train_y)
            predictions = model.predict(X.iloc[validation_indices])
            probabilities = model.predict_proba(X.iloc[validation_indices])
            metrics = calculate_binary_metrics(y.iloc[validation_indices], predictions, probabilities)
            scores.append({"fold": fold_number, "f1": metrics["f1"], "roc_auc": metrics["roc_auc"]})

        if not scores:
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("duration_seconds", time.perf_counter() - trial_start)
        trial.set_user_attr("subset_trip_ids", selected_trips)
        return float(sum(score["f1"] for score in scores) / len(scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trial_rows = []
    for trial in study.trials:
        trial_rows.append({
            "trial": trial.number,
            "status": trial.state.name,
            "objective_value": trial.value,
            "duration_seconds": trial.user_attrs.get("duration_seconds"),
            "fold_scores": trial.user_attrs.get("fold_scores", []),
            "parameters": trial.params,
            "subset_trip_ids": trial.user_attrs.get("subset_trip_ids", selected_trips),
        })

    csv_path = output_path / "optuna_trials.csv"
    pd.DataFrame([
        {
            "trial": row["trial"],
            "status": row["status"],
            "objective_value": row["objective_value"],
            "duration_seconds": row["duration_seconds"],
            "fold_scores": json.dumps(row["fold_scores"]),
            "parameters": json.dumps(row["parameters"]),
            "subset_trip_ids": json.dumps(row["subset_trip_ids"]),
        }
        for row in trial_rows
    ]).to_csv(csv_path, index=False)

    result = {
        "study_name": study_name,
        "storage": storage,
        "subset_trip_ids": selected_trips,
        "subset_rows": int(len(subset)),
        "subset_trips": int(subset["trip_id"].nunique()),
        "n_splits": effective_splits,
        "n_trials": n_trials,
        "random_seed": random_seed,
        "completed_trials": sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials),
        "best_hyperparameters": study.best_params if study.best_trial else None,
        "best_objective_value": study.best_value if study.best_trial else None,
        "trials": trial_rows,
    }

    json_path = output_path / "random_forest_optimization.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["json_path"] = str(json_path)
    result["csv_path"] = str(csv_path)
    return result
