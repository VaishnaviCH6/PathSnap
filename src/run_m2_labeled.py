"""Generate labelled candidate tables for train or student_test partitions only.

Run from the PathSnap repository root with PYTHONPATH=src. Evaluator partitions
are deliberately not accepted by this command.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from candidate_generation.candidates import load_road_network, build_road_network, generate_candidates
from feature_engineering.features import add_observation_features, extract_candidate_features
from labeling.reference_labels import (
    align_observations_to_route, label_candidate_table, load_reference_route,
)


def load_track(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None,
                        names=["longitude", "latitude", "timestamp"])
    frame.insert(0, "observation_id", range(len(frame)))
    return frame


def process_trip(trip_dir: Path, output_dir: Path, radius: float,
                 max_candidates: int, max_alignment_distance: float,
                 ambiguity_margin: float):
    trip_id = trip_dir.name
    paths = {suffix: trip_dir / f"{trip_id}.{suffix}"
             for suffix in ("track", "nodes", "arcs", "route")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    track = load_track(paths["track"])
    nodes, arcs = load_road_network(str(paths["nodes"]), str(paths["arcs"]))
    network = build_road_network(nodes, arcs)
    candidates, candidate_status = generate_candidates(
        track, network, search_radius_m=radius, max_candidates=max_candidates
    )
    track_features = add_observation_features(track, network.transformer)
    features = extract_candidate_features(
        track_features, candidates, arcs, nodes, network.transformer
    )
    route_ids = load_reference_route(paths["route"])
    alignment = align_observations_to_route(
        track_features, route_ids, network,
        max_alignment_distance_m=max_alignment_distance,
        ambiguity_margin_m=ambiguity_margin,
    )
    labelled, status = label_candidate_table(
        trip_id, features, candidate_status, alignment
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(output_dir / f"{trip_id}_labelled_candidates.csv", index=False)
    status.to_csv(output_dir / f"{trip_id}_observation_status.csv", index=False)
    return labelled, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["train", "student_test"], required=True)
    parser.add_argument("--trip-root", required=True,
                        help="Directory containing trip subdirectories for the selected partition")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-alignment-distance", type=float, default=100.0)
    parser.add_argument("--ambiguity-margin", type=float, default=3.0)
    args = parser.parse_args()

    trip_root = Path(args.trip_root)
    trip_dirs = sorted(p for p in trip_root.iterdir() if p.is_dir())
    if not trip_dirs:
        raise SystemExit(f"No trip directories found under {trip_root}")

    all_labelled, all_status = [], []
    for trip_dir in trip_dirs:
        print(f"Processing {args.partition} trip {trip_dir.name} ...", flush=True)
        labelled, status = process_trip(
            trip_dir, Path(args.output_dir), args.radius, args.max_candidates,
            args.max_alignment_distance, args.ambiguity_margin,
        )
        all_labelled.append(labelled)
        all_status.append(status)

    combined = pd.concat(all_labelled, ignore_index=True)
    statuses = pd.concat(all_status, ignore_index=True)
    out = Path(args.output_dir)
    combined.to_csv(out / f"{args.partition}_labelled_candidates.csv", index=False)
    statuses.to_csv(out / f"{args.partition}_observation_status.csv", index=False)

    eligible = statuses["supervised_eligible"].astype(bool)
    print("\n=== LABEL GENERATION SUMMARY ===")
    print(f"Partition: {args.partition}")
    print(f"Trips: {len(trip_dirs)}")
    print(f"Candidate rows: {len(combined):,}")
    print(f"Supervised-eligible observations: {int(eligible.sum()):,}")
    print(f"Ambiguous alignments: {(statuses.alignment_status == 'ambiguous').sum():,}")
    print(f"Candidate misses: {(statuses.candidate_status == 'candidate_miss').sum():,}")
    print(f"No generated candidates: {(statuses.candidate_status == 'no_generated_candidates').sum():,}")
    print(f"Positive labels: {(combined.label == 1).sum():,}")
    print(f"Negative labels: {(combined.label == 0).sum():,}")
    print(f"Unlabelled candidate rows (excluded): {combined.label.isna().sum():,}")
    print(f"Saved outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
