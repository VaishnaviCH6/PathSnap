"""Run M2 candidate generation and feature extraction for one trip directory."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from candidate_generation.candidates import (
    load_road_network, build_road_network, generate_candidates
)
from feature_engineering.features import (
    add_observation_features, extract_candidate_features
)


def load_track(path: Path) -> pd.DataFrame:
    track = pd.read_csv(path, sep=r"\s+", header=None,
                        names=["longitude", "latitude", "timestamp"])
    track.insert(0, "observation_id", range(len(track)))
    return track


def process_trip(trip_dir: str | Path, output_dir: str | Path,
                 search_radius_m: float = 50.0, max_candidates: int = 8):
    trip_dir, output_dir = Path(trip_dir), Path(output_dir)
    trip_id = trip_dir.name
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    for path in (track_path, nodes_path, arcs_path):
        if not path.exists():
            raise FileNotFoundError(path)

    track = load_track(track_path)
    nodes, arcs = load_road_network(str(nodes_path), str(arcs_path))
    network = build_road_network(nodes, arcs)
    candidates, status = generate_candidates(
        track, network, search_radius_m=search_radius_m,
        max_candidates=max_candidates
    )
    track_features = add_observation_features(track, network.transformer)
    feature_table = extract_candidate_features(
        track_features, candidates, arcs, nodes, network.transformer
    )
    feature_table.insert(0, "trip_id", trip_id)
    status.insert(0, "trip_id", trip_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{trip_id}_candidate_features.csv"
    status_path = output_dir / f"{trip_id}_observation_status.csv"
    feature_table.to_csv(feature_path, index=False)
    status.to_csv(status_path, index=False)
    return feature_table, status, feature_path, status_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    features, status, feature_path, status_path = process_trip(
        args.trip_dir, args.output_dir, args.radius, args.max_candidates
    )
    print(f"Candidate-feature rows: {len(features)}")
    print(f"Observations: {len(status)}")
    print(f"Observations with no candidates: {(status.candidate_count == 0).sum()}")
    print(f"Saved: {feature_path}")
    print(f"Saved: {status_path}")


if __name__ == "__main__":
    main()
