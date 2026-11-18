from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import calculate_binary_metrics
from src.models.model_factory import get_models
from src.models.model_io import save_model


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


def load_labeled_candidates(path: str | Path) -> pd.DataFrame:
    """Load only supervised candidate columns with efficient numeric dtypes."""
    use_columns = [
        "trip_id",
        "observation_id",
        "candidate_arc_id",
        *FEATURE_COLUMNS,
        "label",
    ]
    data = pd.read_csv(
        path,
        usecols=use_columns,
        dtype={
            "trip_id": str,
            "observation_id": "Int64",
            "candidate_arc_id": "Int64",
        },
    )
    data = data[data["label"].isin([0, 1])].copy()
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    data["label"] = data["label"].astype("int8")
    return data


def top1_accuracy(
    validation_frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> float:
    """Measure whether the highest-probability candidate has label 1 per observation."""
    ranking = validation_frame[["trip_id", "observation_id", "label"]].copy()
    ranking["probability"] = probabilities[:, 1]
    correct = 0
    observations = 0
    for _, group in ranking.groupby(["trip_id", "observation_id"], sort=False):
        observations += 1
        selected = group.iloc[int(np.argmax(group["probability"].to_numpy()))]
        correct += int(selected["label"] == 1)
    return float(correct / observations) if observations else 0.0


def run_real_comparison(
    candidate_csv: str | Path,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
    smoke_rows: int | None = None,
    output_directory: str | Path = "experiments/results",
    model_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Train all five baseline models on labeled real candidate rows using identical folds."""
    data = load_labeled_candidates(candidate_csv)
    if smoke_rows is not None:
        data = data.iloc[:smoke_rows].copy()

    if data.empty:
        raise ValueError("No labeled candidate rows are available.")
    if data["label"].nunique() != 2:
        raise ValueError("Both binary label classes are required.")

    trip_count = int(data["trip_id"].nunique())
    effective_splits = min(n_splits, trip_count)
    if effective_splits < 2:
        raise ValueError("At least two trips/folds are required.")

    splitter = GroupKFold(n_splits=effective_splits)
    splits = list(splitter.split(data, data["label"], groups=data["trip_id"]))

    X = data[list(FEATURE_COLUMNS)]
    y = data["label"]
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    fold_output_name = "real_smoke_fold_metrics.csv" if smoke_rows else "real_fold_metrics.csv"
    fold_output_path = output_path / fold_output_name

    fold_rows: list[dict[str, Any]] = []
    model_results: dict[str, Any] = {}

    available_models = get_models(random_seed)
    if model_names is None:
        selected_models = list(available_models)
    else:
        selected_models = list(model_names)
        unknown = [name for name in selected_models if name not in available_models]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")

    if not smoke_rows and fold_output_path.is_file():
        existing_fold_frame = pd.read_csv(fold_output_path)
        fold_rows = existing_fold_frame.loc[
            ~existing_fold_frame["model"].isin(selected_models)
        ].to_dict(orient="records")

    for model_name in selected_models:
        model_template = available_models[model_name]
        print(f"Starting model: {model_name}", flush=True)
        for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
            train_trips = set(data.iloc[train_indices]["trip_id"])
            validation_trips = set(data.iloc[validation_indices]["trip_id"])
            overlap = train_trips & validation_trips
            if overlap:
                raise RuntimeError("Trip leakage detected between real training and validation folds.")

            train_y = y.iloc[train_indices]
            if train_y.nunique() < 2:
                fold_rows.append({
                    "model": model_name,
                    "fold": fold_number,
                    "status": "skipped_single_class_training_fold",
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                    "top1_accuracy": None,
                })
                continue

            model = clone(model_template)
            fit_start = perf_counter()
            model.fit(X.iloc[train_indices], train_y)
            training_time = perf_counter() - fit_start

            inference_start = perf_counter()
            predictions = model.predict(X.iloc[validation_indices])
            probabilities = model.predict_proba(X.iloc[validation_indices])
            inference_time = perf_counter() - inference_start

            validation_frame = data.iloc[validation_indices].copy()
            metrics = calculate_binary_metrics(
                y.iloc[validation_indices],
                predictions,
                probabilities,
            )
            row = {
                "model": model_name,
                "fold": fold_number,
                "status": "completed",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "top1_accuracy": top1_accuracy(validation_frame, probabilities),
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "train_trips": len(train_trips),
                "validation_trips": len(validation_trips),
                "tn": metrics["confusion_matrix"]["tn"],
                "fp": metrics["confusion_matrix"]["fp"],
                "fn": metrics["confusion_matrix"]["fn"],
                "tp": metrics["confusion_matrix"]["tp"],
            }
            fold_rows.append(row)
            print(f"Completed fold {fold_number}/{effective_splits}: {model_name}", flush=True)

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(fold_output_path, index=False)

        completed = [row for row in fold_rows if row["model"] == model_name and row["status"] == "completed"]
        model_results[model_name] = {"folds": completed}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    fold_frame = pd.DataFrame(fold_rows)
    fold_path = output_path / ("real_smoke_fold_metrics.csv" if smoke_rows else "real_fold_metrics.csv")
    fold_frame.to_csv(fold_path, index=False)

    comparison_rows: list[dict[str, Any]] = []
    for model_name in get_models(random_seed):
        completed = [row for row in fold_rows if row["model"] == model_name and row["status"] == "completed"]
        if not completed:
            continue
        frame = pd.DataFrame(completed)
        numeric_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "top1_accuracy", "training_time_seconds", "inference_time_seconds"]
        row: dict[str, Any] = {"model": model_name, "fold_count": len(completed)}
        for metric in numeric_metrics:
            values = frame[metric].dropna().astype(float)
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else None
            row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) else None
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison["rank"] = comparison["mean_f1"].rank(method="min", ascending=False).astype(int)
        comparison = comparison.sort_values(["rank", "model"]).reset_index(drop=True)

    comparison_path = output_path / ("real_smoke_model_comparison.csv" if smoke_rows else "real_model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    bar_path = output_path / ("real_smoke_model_comparison.png" if smoke_rows else "model_comparison.png")
    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_frame = comparison.sort_values("mean_f1")
        ax.barh(plot_frame["model"], plot_frame["mean_f1"])
        ax.set_xlabel("Mean cross-validation F1")
        ax.set_title("Real PathSnap candidate-level F1 comparison")
        fig.tight_layout()
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    confusion_path = output_path / ("real_smoke_confusion_matrices.png" if smoke_rows else "confusion_matrices.png")
    models_with_folds = [row["model"] for row in fold_rows if row["status"] == "completed"]
    model_names = list(dict.fromkeys(models_with_folds))
    if model_names:
        fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 4))
        if len(model_names) == 1:
            axes = [axes]
        for axis, model_name in zip(axes, model_names):
            completed = [row for row in fold_rows if row["model"] == model_name and row["status"] == "completed"]
            tn = sum(row["tn"] for row in completed)
            fp = sum(row["fp"] for row in completed)
            fn = sum(row["fn"] for row in completed)
            tp = sum(row["tp"] for row in completed)
            matrix = np.array([[tn, fp], [fn, tp]])
            axis.imshow(matrix, cmap="Blues")
            axis.set_title(model_name)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("Actual")
            axis.set_xticks([0, 1], ["0", "1"])
            axis.set_yticks([0, 1], ["0", "1"])
            for r in range(2):
                for c in range(2):
                    axis.text(c, r, matrix[r, c], ha="center", va="center")
        fig.tight_layout()
        fig.savefig(confusion_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    experiment = {
        "timestamp_utc": timestamp,
        "experiment_type": "real_smoke" if smoke_rows else "real_full_baseline",
        "dataset": str(candidate_csv),
        "labeled_rows": int(len(data)),
        "trip_count": trip_count,
        "fold_count": effective_splits,
        "random_seed": random_seed,
        "feature_columns": list(FEATURE_COLUMNS),
        "positive_class": 1,
        "primary_metric": "mean_f1",
    }
    metadata_path = output_path / ("real_smoke_metadata.json" if smoke_rows else "real_baseline_metadata.json")
    metadata_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    return {
        "comparison": comparison,
        "fold_metrics": fold_frame,
        "comparison_path": str(comparison_path),
        "fold_metrics_path": str(fold_path),
        "bar_chart_path": str(bar_path),
        "confusion_chart_path": str(confusion_path),
        "metadata_path": str(metadata_path),
        "experiment": experiment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Member 3 PathSnap model comparison")
    parser.add_argument("--csv", type=Path, default=Path("dataset/m2_handoff/train_labelled_candidates.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-rows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    result = run_real_comparison(
        args.csv,
        n_splits=args.folds,
        random_seed=args.seed,
        smoke_rows=args.smoke_rows,
        output_directory=args.output,
        model_names=args.models,
    )
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
