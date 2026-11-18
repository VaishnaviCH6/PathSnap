from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = (
    "trip_id",
    "observation_id",
    "candidate_arc_id",
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
    "label",
)

FEATURE_COLUMNS = tuple(REQUIRED_CANDIDATE_COLUMNS[3:13])

REQUIRED_STATUS_COLUMNS = (
    "trip_id",
    "observation_id",
    "candidate_count",
    "candidate_status",
    "alignment_status",
    "supervised_eligible",
)


def validate_m2_handoff(
    candidate_csv: str | Path,
    status_csv: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate Member 2 candidate rows and status against the documented schema."""
    candidate_path = Path(candidate_csv)
    status_path = Path(status_csv)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not status_path.is_file():
        raise FileNotFoundError(status_path)

    candidates = pd.read_csv(
        candidate_path,
        dtype={"trip_id": str, "observation_id": "Int64", "candidate_arc_id": "Int64"},
    )
    status = pd.read_csv(
        status_path,
        dtype={"trip_id": str, "observation_id": "Int64"},
    )

    missing_candidate = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in candidates.columns]
    missing_status = [column for column in REQUIRED_STATUS_COLUMNS if column not in status.columns]
    if missing_candidate:
        raise ValueError(f"Missing candidate columns: {missing_candidate}")
    if missing_status:
        raise ValueError(f"Missing status columns: {missing_status}")

    label_values = set(candidates["label"].dropna().unique().tolist())
    invalid_labels = sorted(label_values.difference({0, 1, 0.0, 1.0}))
    if invalid_labels:
        raise ValueError(f"Invalid binary labels found: {invalid_labels}")

    labeled = candidates[candidates["label"].isin([0, 1])].copy()
    positive_count = int((labeled["label"] == 1).sum())
    negative_count = int((labeled["label"] == 0).sum())
    unlabeled_count = int(candidates["label"].isna().sum())

    pair_columns = ["trip_id", "observation_id", "candidate_arc_id"]
    duplicate_rows = int(candidates.duplicated(pair_columns, keep=False).sum())
    duplicate_keys = int(candidates.duplicated(pair_columns, keep="first").sum())

    numeric_dtypes: dict[str, str] = {}
    missing_features: dict[str, int] = {}
    infinite_features: dict[str, int] = {}
    for column in FEATURE_COLUMNS:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
        numeric_dtypes[column] = str(candidates[column].dtype)
        missing_features[column] = int(candidates[column].isna().sum())
        infinite_features[column] = int(np.isinf(candidates[column].to_numpy(dtype=float)).sum())

    positive_per_observation = (
        labeled[labeled["label"] == 1]
        .groupby(["trip_id", "observation_id"], dropna=False)
        .size()
    )
    positive_distribution = positive_per_observation.value_counts().sort_index().to_dict()

    eligible = status.loc[status["supervised_eligible"] == True, ["trip_id", "observation_id"]].drop_duplicates()
    positive_keys = pd.DataFrame(positive_per_observation.reset_index()[["trip_id", "observation_id"]])
    eligible_missing_positive = int(len(eligible.merge(positive_keys, on=["trip_id", "observation_id"], how="left", indicator=True).query('_merge == "left_only"')))

    status_counts = status["candidate_status"].fillna("missing").value_counts().to_dict()
    alignment_counts = status["alignment_status"].fillna("missing").value_counts().to_dict()

    duplicate_details = candidates.loc[
        candidates.duplicated(pair_columns, keep=False), pair_columns + ["label"]
    ].copy()

    result: dict[str, Any] = {
        "candidate_rows": int(len(candidates)),
        "labeled_candidate_rows": int(len(labeled)),
        "unlabeled_candidate_rows": unlabeled_count,
        "unique_training_trips": int(candidates["trip_id"].nunique()),
        "positive_labels": positive_count,
        "negative_labels": negative_count,
        "positive_class_proportion": float(positive_count / len(labeled)) if len(labeled) else None,
        "duplicate_observation_candidate_rows": duplicate_rows,
        "duplicate_observation_candidate_keys": duplicate_keys,
        "numeric_dtypes": numeric_dtypes,
        "missing_features": missing_features,
        "infinite_features": infinite_features,
        "positive_labels_per_observation": positive_distribution,
        "observations_with_positive_label": int(len(positive_per_observation)),
        "eligible_observations": int(len(eligible)),
        "eligible_observations_missing_positive": eligible_missing_positive,
        "candidate_status_counts": status_counts,
        "alignment_status_counts": alignment_counts,
    }

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for key, value in result.items():
        if not isinstance(value, (dict, type(None), str, int, float)):
            continue
        summary_rows.append({"metric": key, "value": json.dumps(value, default=str)})

    pd.DataFrame(summary_rows).to_csv(output_path / "data_validation.csv", index=False)
    (output_path / "data_validation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if len(duplicate_details):
        duplicate_details.head(1000).to_csv(output_path / "candidate_duplicate_pairs.csv", index=False)

    return result
