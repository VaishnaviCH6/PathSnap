from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def build_model_comparison(results: dict[str, Any]) -> pd.DataFrame:
    """Build a ranked model comparison using mean F1 as the primary metric."""
    rows: list[dict[str, Any]] = []
    for model_name, model_result in results["models"].items():
        aggregate = model_result["aggregate"]
        rows.append(
            {
                "model": model_name,
                "mean_f1": aggregate.get("f1_mean"),
                "std_f1": aggregate.get("f1_std"),
                "mean_accuracy": aggregate.get("accuracy_mean"),
                "std_accuracy": aggregate.get("accuracy_std"),
                "mean_precision": aggregate.get("precision_mean"),
                "std_precision": aggregate.get("precision_std"),
                "mean_recall": aggregate.get("recall_mean"),
                "std_recall": aggregate.get("recall_std"),
                "mean_roc_auc": aggregate.get("roc_auc_mean"),
                "std_roc_auc": aggregate.get("roc_auc_std"),
                "fold_count": aggregate["fold_count"],
                "mean_training_seconds": _mean_fold_value(model_result["fold_results"], "training_time_seconds"),
                "mean_inference_seconds": _mean_fold_value(model_result["fold_results"], "inference_time_seconds"),
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    comparison["rank"] = comparison["mean_f1"].rank(method="min", ascending=False).astype(int)
    return comparison.sort_values(["rank", "model"]).reset_index(drop=True)


def _mean_fold_value(folds: list[dict[str, Any]], key: str) -> float | None:
    values = [fold[key] for fold in folds if fold.get(key) is not None]
    return float(sum(values) / len(values)) if values else None


def save_model_comparison(
    results: dict[str, Any],
    output_directory: str | Path,
    experiment_name: str = "synthetic_smoke_test",
) -> tuple[Path, Path]:
    """Save ranked model and fold-wise CSV reports plus comparison metadata."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison = build_model_comparison(results)
    comparison_path = output_path / f"{experiment_name}_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    fold_rows: list[dict[str, Any]] = []
    for model_name, model_result in results["models"].items():
        for fold in model_result["fold_results"]:
            fold_rows.append({"model": model_name, **fold})
    fold_path = output_path / f"{experiment_name}_fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    metadata_path = output_path / f"{experiment_name}_comparison.json"
    metadata_path.write_text(
        json.dumps(
            {
                "primary_metric": "mean_f1",
                "validation_method": "GroupKFold by trip_id",
                "synthetic_data_only": results.get("synthetic_data_only", True),
                "comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return comparison_path, fold_path


def save_comparison_charts(
    results: dict[str, Any],
    output_directory: str | Path,
    experiment_name: str = "synthetic_smoke_test",
) -> tuple[Path, Path]:
    """Export F1/training bar charts and confusion-matrix grids."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison = build_model_comparison(results)
    if comparison.empty:
        raise ValueError("No model comparison rows are available.")

    bar_path = output_path / f"{experiment_name}_model_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    comparison.sort_values("mean_f1").plot(
        x="model", y="mean_f1", kind="barh", ax=axes[0], legend=False,
    )
    axes[0].set_title("Mean F1-score")
    comparison.sort_values("mean_training_seconds").plot(
        x="model", y="mean_training_seconds", kind="barh", ax=axes[1], legend=False,
    )
    axes[1].set_title("Mean training time (seconds)")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    completed = {
        model_name: [
            fold["confusion_matrix"]
            for fold in model_result["fold_results"]
            if "confusion_matrix" in fold
        ]
        for model_name, model_result in results["models"].items()
    }
    model_names = [name for name, matrices in completed.items() if matrices]
    if not model_names:
        raise ValueError("No confusion matrices are available.")

    confusion_path = output_path / f"{experiment_name}_confusion_matrices.png"
    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]
    for axis, model_name in zip(axes, model_names):
        matrices = completed[model_name]
        aggregate = {
            key: sum(matrix[key] for matrix in matrices)
            for key in ("tn", "fp", "fn", "tp")
        }
        matrix = [[aggregate["tn"], aggregate["fp"]], [aggregate["fn"], aggregate["tp"]]]
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row][column], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return bar_path, confusion_path
