import numpy as np
import pandas as pd
from candidate_generation.candidates import build_road_network, generate_candidates
from feature_engineering.features import add_observation_features, extract_candidate_features


def toy_network():
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "longitude": [8.0, 8.001, 8.0, 8.001],
        "latitude": [49.0, 49.0, 49.001, 49.001],
    })
    arcs = pd.DataFrame({
        "arc_id": [0, 1], "from_node_id": [0, 2], "to_node_id": [1, 3]
    })
    return nodes, arcs


def test_candidate_generation_preserves_arc_ids_and_radius():
    nodes, arcs = toy_network()
    network = build_road_network(nodes, arcs)
    track = pd.DataFrame({
        "observation_id": [0, 1],
        "longitude": [8.0005, 8.0005],
        "latitude": [49.00001, 49.01],
    })
    candidates, status = generate_candidates(track, network, 30, 5)
    assert set(candidates.candidate_arc_id).issubset({0, 1})
    assert candidates[candidates.observation_id == 0].candidate_arc_id.tolist() == [0]
    assert status.loc[status.observation_id == 1, "candidate_status"].iloc[0] == "no_candidate_within_radius"


def test_invalid_coordinate_is_reported_not_crashed():
    nodes, arcs = toy_network()
    network = build_road_network(nodes, arcs)
    track = pd.DataFrame({"observation_id": [0], "longitude": [999.0], "latitude": [49.0]})
    candidates, status = generate_candidates(track, network)
    assert candidates.empty
    assert status.candidate_status.iloc[0] == "invalid_coordinate"


def test_feature_output_has_required_keys_and_units():
    nodes, arcs = toy_network()
    network = build_road_network(nodes, arcs)
    track = pd.DataFrame({
        "observation_id": [0, 1],
        "longitude": [8.0002, 8.0004],
        "latitude": [49.0, 49.0],
        "timestamp": [1.0, 2.0],
    })
    candidates, _ = generate_candidates(track, network, 30, 5)
    tf = add_observation_features(track, network.transformer)
    features = extract_candidate_features(tf, candidates, arcs, nodes, network.transformer)
    assert {"observation_id", "candidate_arc_id", "distance_to_segment_m",
            "direction_difference_deg", "gps_speed_mps"}.issubset(features.columns)
    assert features.candidate_arc_id.isin(arcs.arc_id).all()
    assert features.loc[features.observation_id == 1, "gps_speed_mps"].notna().all()
