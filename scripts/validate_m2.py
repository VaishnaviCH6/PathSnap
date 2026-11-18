
from pathlib import Path
import pandas as pd
import numpy as np

DATASET = Path("dataset/train")
OUTPUT = Path("outputs/final_training")

total_invalid_arc_ids = 0
total_invalid_features = 0
trip_results = []

for status_file in sorted(OUTPUT.glob("*_observation_status.csv")):

    trip_id = status_file.name.replace(
        "_observation_status.csv", ""
    )

    feature_file = OUTPUT / f"{trip_id}_candidate_features.csv"
    arcs_file = DATASET / trip_id / f"{trip_id}.arcs"

    status = pd.read_csv(status_file)
    features = pd.read_csv(feature_file)

    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
    )

    # Validate original arc IDs
    invalid_ids = (
        ~features["candidate_arc_id"].between(
            0, len(arcs) - 1
        )
    ).sum()

    total_invalid_arc_ids += invalid_ids

    # Validate numerical features
    numeric_columns = [
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
    ]

    values = features[numeric_columns].to_numpy(dtype=float)

    invalid_values = np.isinf(values).sum()
    total_invalid_features += invalid_values

    # Check important feature ranges
    bad_distance = (
        features["distance_to_segment_m"] < 0
    ).sum()

    bad_fraction = (
        (features["projection_fraction"] < 0) |
        (features["projection_fraction"] > 1)
    ).sum()

    bad_direction = (
        (features["direction_difference_deg"] < 0) |
        (features["direction_difference_deg"] > 180)
    ).sum()

    trip_results.append({
        "trip_id": trip_id,
        "observations": len(status),
        "candidate_rows": len(features),
        "no_candidates": (status.candidate_count == 0).sum(),
        "invalid_arc_ids": invalid_ids,
        "infinite_values": invalid_values,
        "bad_distances": bad_distance,
        "bad_projection": bad_fraction,
        "bad_directions": bad_direction,
    })


results = pd.DataFrame(trip_results)

print("\n========== M2 VALIDATION ==========\n")

print("Trips checked:", len(results))
print("Invalid arc IDs:", total_invalid_arc_ids)
print("Infinite feature values:", total_invalid_features)

print("\nPer-trip validation:")
print(results.to_string(index=False))

print("\nTotal missing candidates:",
      results.no_candidates.sum())

print("\nTotal candidate rows:",
      results.candidate_rows.sum())

print("\nInvalid distance values:",
      results.bad_distances.sum())

print("\nInvalid projection fractions:",
      results.bad_projection.sum())

print("\nInvalid direction differences:",
      results.bad_directions.sum())

results.to_csv(
    OUTPUT / "m2_validation_report.csv",
    index=False
)

print("\nSaved: outputs/final_training/m2_validation_report.csv")
