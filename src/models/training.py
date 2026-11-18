from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.cross_validation import group_kfold_evaluate
from src.models.model_factory import get_models


def train_and_evaluate_models(
    data: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    label_column: str = "label",
    trip_id_column: str = "trip_id",
    n_splits: int = 5,
    random_state: int = 42,
    output_directory: str | Path = "experiments",
    synthetic_data_only: bool = True,
) -> dict[str, Any]:
    """Run group-aware model evaluation and save reproducible experiment results."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a Pandas DataFrame.")

    models = get_models(random_state=random_state)
    results = group_kfold_evaluate(
        data,
        models,
        feature_columns=feature_columns,
        label_column=label_column,
        trip_id_column=trip_id_column,
        n_splits=n_splits,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    experiment = {
        "timestamp_utc": timestamp,
        "random_state": random_state,
        "n_splits": n_splits,
        "grouping_column": trip_id_column,
        "label_column": label_column,
        "synthetic_data_only": synthetic_data_only,
        "fold_groups": results["fold_groups"],
        "feature_columns": results["feature_columns"],
        "models": results["models"],
    }

    json_path = output_path / f"group_cv_{timestamp}.json"
    json_path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for model_name, model_result in results["models"].items():
        for fold in model_result["fold_results"]:
            row = {
                "model": model_name,
                "fold": fold["fold"],
                "status": fold.get("status", "completed"),
                "accuracy": fold.get("accuracy"),
                "precision": fold.get("precision"),
                "recall": fold.get("recall"),
                "f1": fold.get("f1"),
                "roc_auc": fold.get("roc_auc"),
                "training_time_seconds": fold.get("training_time_seconds"),
                "inference_time_seconds": fold.get("inference_time_seconds"),
                "train_samples": fold.get("train_samples"),
                "validation_samples": fold.get("validation_samples"),
                "train_trip_count": fold.get("train_trip_count"),
                "validation_trip_count": fold.get("validation_trip_count"),
            }
            rows.append(row)

    csv_path = output_path / f"group_cv_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    results["json_path"] = str(json_path)
    results["csv_path"] = str(csv_path)
    return results


def main() -> None:
    """Run synthetic or data-file-backed training from the repository root."""
    parser = argparse.ArgumentParser(description="PathSnap Member 3 model training")
    parser.add_argument("--csv", type=Path, help="Candidate feature CSV (real data only)")
    parser.add_argument("--features", nargs="+", help="Predictive feature column names")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--trip-id-column", default="trip_id")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    if args.csv is None:
        raise SystemExit("No data source supplied. Provide --csv for approved candidate-level data.")

    data = pd.read_csv(args.csv)
    results = train_and_evaluate_models(
        data,
        feature_columns=args.features,
        label_column=args.label_column,
        trip_id_column=args.trip_id_column,
        n_splits=args.folds,
        random_state=args.seed,
        output_directory=args.output,
        synthetic_data_only=False,
    )
    print(json.dumps({"csv_path": results["csv_path"], "json_path": results["json_path"]}, indent=2))


if __name__ == "__main__":
    main()
