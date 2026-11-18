from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.prediction import predict_candidates


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

MODEL_PATH = Path("artifacts/random_forest_tuned_20261118T123057Z.joblib")
DATA_PATH = Path("dataset/m2_handoff/train_labelled_candidates.csv")


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        usecols=[
            "trip_id",
            "observation_id",
            "candidate_arc_id",
            *FEATURE_COLUMNS,
        ],
        dtype={
            "trip_id": str,
            "observation_id": "Int64",
            "candidate_arc_id": "Int64",
        },
        nrows=100,
    )

    output = predict_candidates(data, MODEL_PATH)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
