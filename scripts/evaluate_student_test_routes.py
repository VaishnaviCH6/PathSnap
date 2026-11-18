"""Evaluate classifier-only and HMM map matching on the 10 student-test trips.

Reference .route files and candidate labels are read only after prediction/decoding.
They are used exclusively for evaluation, never as model or decoder inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.map_matching.hmm import DirectedRoadGraph, decode_trip, reconstruct_trajectory
from src.models.prediction import predict_candidates
from src.candidate_generation.candidates import load_road_network


MODEL_PATH = Path("artifacts/random_forest_tuned_20261118T123057Z.joblib")
STUDENT_TEST_DIR = Path("dataset/student_test")
CANDIDATE_DIR = Path("outputs/labelled/student_test")
OUTPUT_DIR = Path("outputs/m2_student_test_evaluation")


def read_track(path: Path) -> pd.DataFrame:
    track = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    track.insert(0, "observation_id", np.arange(len(track), dtype=int))
    return track


def read_reference_route(path: Path) -> list[int]:
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    return [int(value) for value in values.tolist()]


def lcs_length(a: list[int], b: list[int]) -> int:
    """Length of the longest common subsequence, preserving repeated arc visits."""
    previous = [0] * (len(b) + 1)
    for value in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            if value == other:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def levenshtein_distance(a: list[int], b: list[int]) -> int:
    """Insertion/deletion/substitution edit distance for ordered arc sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, value in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (value != other),
            ))
        previous = current
    return previous[-1]


def sequence_metrics(predicted: list[int], reference: list[int]) -> dict[str, float | int]:
    common = lcs_length(predicted, reference)
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    edit = levenshtein_distance(predicted, reference)
    denominator = max(len(predicted), len(reference), 1)
    return {
        "predicted_arc_count": len(predicted),
        "reference_arc_count": len(reference),
        "lcs_arc_count": common,
        "sequence_precision_lcs": precision,
        "sequence_recall_lcs": recall,
        "sequence_f1_lcs": f1,
        "sequence_edit_distance": edit,
        "normalized_edit_similarity": 1.0 - edit / denominator,
    }


def observation_accuracy(
    decoded: pd.DataFrame,
    candidate_labels: pd.DataFrame,
) -> tuple[float, int]:
    """Compare one predicted arc per observation against eligible positive labels."""
    reference = (
        candidate_labels.loc[candidate_labels["label"] == 1,
                             ["observation_id", "candidate_arc_id"]]
        .drop_duplicates("observation_id")
        .set_index("observation_id")["candidate_arc_id"]
    )
    predicted = decoded.dropna(subset=["decoded_arc_id"]).copy()
    predicted["decoded_arc_id"] = predicted["decoded_arc_id"].astype(int)
    comparable = predicted[predicted["observation_id"].isin(reference.index)]
    if comparable.empty:
        return float("nan"), 0
    expected = reference.loc[comparable["observation_id"]].to_numpy(dtype=int)
    actual = comparable["decoded_arc_id"].to_numpy(dtype=int)
    return float(np.mean(actual == expected)), len(comparable)


def classifier_only(candidates: pd.DataFrame, observation_ids: list[int]) -> pd.DataFrame:
    """Top-probability candidate per observation; missing observations remain explicit."""
    if candidates.empty:
        selected = pd.DataFrame(columns=[
            "observation_id", "decoded_arc_id", "decoded_projection_fraction",
            "selected_probability", "decode_status",
        ])
    else:
        ordered = candidates.sort_values(
            ["observation_id", "positive_probability", "candidate_arc_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        selected = ordered.drop_duplicates("observation_id", keep="first")
        selected = selected.rename(columns={
            "candidate_arc_id": "decoded_arc_id",
            "projection_fraction": "decoded_projection_fraction",
            "positive_probability": "selected_probability",
        })
        selected["decode_status"] = "decoded"
        selected = selected[[
            "observation_id", "decoded_arc_id", "decoded_projection_fraction",
            "selected_probability", "decode_status",
        ]]
    base = pd.DataFrame({"observation_id": observation_ids})
    result = base.merge(selected, on="observation_id", how="left", validate="one_to_one")
    result["decode_status"] = result["decode_status"].fillna("no_candidates")
    result["decoded_arc_id"] = result["decoded_arc_id"].astype("Int64")
    return result


def flatten_segments(segments: pd.DataFrame) -> list[int]:
    if segments.empty:
        return []
    route: list[int] = []
    for sequence in segments["arc_sequence"]:
        route.extend(int(arc_id) for arc_id in sequence)
    return route


def route_continuity(decoded: pd.DataFrame, graph: DirectedRoadGraph) -> tuple[float, int, int]:
    """Fraction of adjacent decoded observations with a legal directed transition."""
    ordered = decoded.sort_values("observation_id", kind="stable")
    legal = 0
    checked = 0
    previous = None
    previous_obs = None
    for row in ordered.itertuples(index=False):
        if pd.isna(row.decoded_arc_id) or row.decode_status == "no_candidates":
            previous = None
            previous_obs = None
            continue
        current = (int(row.decoded_arc_id), float(row.decoded_projection_fraction))
        if previous is not None and int(row.observation_id) == previous_obs + 1:
            checked += 1
            if graph.transition_distance(previous[0], previous[1], current[0], current[1]) is not None:
                legal += 1
        previous = current
        previous_obs = int(row.observation_id)
    return (legal / checked if checked else float("nan"), legal, checked)


def evaluate_trip(trip_id: str, model_path: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    trip_dir = STUDENT_TEST_DIR / trip_id
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"
    candidate_path = CANDIDATE_DIR / f"{trip_id}_labelled_candidates.csv"

    required_paths = [track_path, nodes_path, arcs_path, route_path, candidate_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required evaluation inputs:\n" + "\n".join(missing))

    track = read_track(track_path)
    observation_ids = track["observation_id"].astype(int).tolist()
    candidates = pd.read_csv(candidate_path)
    required_candidate_columns = {
        "observation_id", "candidate_arc_id", "projection_fraction",
        "gps_step_distance_m",
    }
    absent = required_candidate_columns - set(candidates.columns)
    if absent:
        raise ValueError(f"{candidate_path} missing columns: {sorted(absent)}")

    # Prediction receives features and identifiers only. Labels/reference routes
    # are not passed to the model or decoder.
    feature_columns = [
        "trip_id", "observation_id", "candidate_arc_id",
        "distance_to_segment_m", "direction_difference_deg",
        "gps_step_distance_m", "time_difference_s", "gps_speed_mps",
        "segment_length_m", "projection_fraction", "gps_heading_deg",
        "road_bearing_deg", "gps_heading_change_deg",
    ]
    absent_features = set(feature_columns) - set(candidates.columns)
    if absent_features:
        raise ValueError(f"{candidate_path} missing model inputs: {sorted(absent_features)}")
    prediction_input = candidates[feature_columns].copy()

    inference_start = time.perf_counter()
    predictions = predict_candidates(prediction_input, model_path)
    inference_seconds = time.perf_counter() - inference_start

    predicted = candidates.merge(
        predictions[["observation_id", "candidate_arc_id", "positive_probability", "predicted_label"]],
        on=["observation_id", "candidate_arc_id"],
        how="left",
        validate="one_to_one",
    )
    if predicted["positive_probability"].isna().any():
        raise RuntimeError(f"Missing model probabilities for trip {trip_id}")

    nodes, arcs = load_road_network(str(nodes_path), str(arcs_path))
    graph = DirectedRoadGraph(nodes, arcs)

    # Candidate labels are isolated for evaluation after model prediction.
    labels = candidates[["observation_id", "candidate_arc_id", "label"]].copy()
    classifier_decoded = classifier_only(predicted, observation_ids)
    hmm_decoded = decode_trip(predicted, observation_ids, arcs, nodes)
    classifier_segments = reconstruct_trajectory(classifier_decoded, arcs, nodes)
    hmm_segments = reconstruct_trajectory(hmm_decoded, arcs, nodes)

    reference_route = read_reference_route(route_path)
    classifier_route = flatten_segments(classifier_segments)
    hmm_route = flatten_segments(hmm_segments)

    classifier_accuracy, classifier_n = observation_accuracy(classifier_decoded, labels)
    hmm_accuracy, hmm_n = observation_accuracy(hmm_decoded, labels)
    classifier_continuity, classifier_legal, classifier_checked = route_continuity(
        classifier_decoded, graph
    )
    hmm_continuity, hmm_legal, hmm_checked = route_continuity(hmm_decoded, graph)

    status_path = CANDIDATE_DIR / f"{trip_id}_observation_status.csv"
    status = pd.read_csv(status_path)
    candidate_misses = int((status["candidate_status"] == "candidate_miss").sum())
    eligible = int(status["supervised_eligible"].astype(bool).sum())
    miss_denominator = candidate_misses + eligible
    miss_rate = candidate_misses / miss_denominator if miss_denominator else float("nan")

    base = {
        "trip_id": trip_id,
        "gps_observations": len(observation_ids),
        "candidate_rows": len(candidates),
        "eligible_observations": eligible,
        "candidate_misses": candidate_misses,
        "candidate_miss_rate": miss_rate,
        "model_inference_seconds": inference_seconds,
        "reference_route_arc_count": len(reference_route),
    }
    classifier_metrics = {
        **base,
        "method": "classifier_only",
        "observation_top1_accuracy": classifier_accuracy,
        "evaluated_observations": classifier_n,
        "route_continuity": classifier_continuity,
        "legal_adjacent_transitions": classifier_legal,
        "checked_adjacent_transitions": classifier_checked,
        **sequence_metrics(classifier_route, reference_route),
        "reconstructed_segments": len(classifier_segments),
    }
    hmm_metrics = {
        **base,
        "method": "classifier_plus_hmm_viterbi",
        "observation_top1_accuracy": hmm_accuracy,
        "evaluated_observations": hmm_n,
        "route_continuity": hmm_continuity,
        "legal_adjacent_transitions": hmm_legal,
        "checked_adjacent_transitions": hmm_checked,
        **sequence_metrics(hmm_route, reference_route),
        "reconstructed_segments": len(hmm_segments),
    }

    # Save auditable predictions/segments per trip.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classifier_decoded.to_csv(OUTPUT_DIR / f"{trip_id}_classifier_only_observations.csv", index=False)
    hmm_decoded.to_csv(OUTPUT_DIR / f"{trip_id}_hmm_observations.csv", index=False)
    classifier_segments.to_json(
        OUTPUT_DIR / f"{trip_id}_classifier_only_segments.json", orient="records", indent=2
    )
    hmm_segments.to_json(
        OUTPUT_DIR / f"{trip_id}_hmm_segments.json", orient="records", indent=2
    )
    return classifier_metrics, hmm_metrics, classifier_decoded, hmm_decoded


def main() -> None:
    global STUDENT_TEST_DIR, CANDIDATE_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--student-test-dir", type=Path, default=STUDENT_TEST_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    STUDENT_TEST_DIR = args.student_test_dir
    CANDIDATE_DIR = args.candidate_dir
    OUTPUT_DIR = args.output_dir

    if not args.model.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model}")
    trip_ids = sorted(
        path.name for path in STUDENT_TEST_DIR.iterdir()
        if path.is_dir() and (path / f"{path.name}.track").exists()
    )
    if len(trip_ids) != 10:
        raise RuntimeError(f"Expected exactly 10 student-test trips; found {len(trip_ids)}: {trip_ids}")

    classifier_rows = []
    hmm_rows = []
    for index, trip_id in enumerate(trip_ids, start=1):
        print(f"[{index}/10] Evaluating trip {trip_id}...", flush=True)
        classifier_metrics, hmm_metrics, _, _ = evaluate_trip(trip_id, args.model)
        classifier_rows.append(classifier_metrics)
        hmm_rows.append(hmm_metrics)

    results = pd.DataFrame(classifier_rows + hmm_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "per_trip_comparison.csv", index=False)

    metric_columns = [
        "observation_top1_accuracy", "route_continuity",
        "sequence_precision_lcs", "sequence_recall_lcs",
        "sequence_f1_lcs", "normalized_edit_similarity",
        "candidate_miss_rate", "model_inference_seconds",
    ]
    summary_rows = []
    for method, group in results.groupby("method", sort=False):
        row = {"method": method, "trips_evaluated": group["trip_id"].nunique()}
        for metric in metric_columns:
            row[f"mean_{metric}"] = float(group[metric].mean(skipna=True))
        row["total_evaluated_observations"] = int(group["evaluated_observations"].sum())
        row["total_reconstructed_segments"] = int(group["reconstructed_segments"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "overall_comparison.csv", index=False)

    report = {
        "student_test_trips": trip_ids,
        "partition": "student_test",
        "reference_route_usage": "evaluation_only",
        "model_artifact": str(args.model),
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "evaluation_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nOVERALL COMPARISON")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
