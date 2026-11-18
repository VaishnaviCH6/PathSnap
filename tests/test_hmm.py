import numpy as np
import pandas as pd

from src.map_matching.hmm import (
    DirectedRoadGraph,
    decode_trip,
    reconstruct_trajectory,
)


def _network():
    # 0 -> 1 -> 2 is connected; 3 -> 4 is a separate directed component.
    nodes = pd.DataFrame({
        "longitude": [54.0, 54.001, 54.002, 54.01, 54.011],
        "latitude": [56.0, 56.0, 56.0, 56.0, 56.0],
    })
    arcs = pd.DataFrame({
        "arc_id": [0, 1, 2],
        "from_node_id": [0, 1, 3],
        "to_node_id": [1, 2, 4],
    })
    return nodes, arcs


def test_directed_transition_allows_connected_arcs():
    nodes, arcs = _network()
    graph = DirectedRoadGraph(nodes, arcs)
    distance = graph.transition_distance(0, 0.5, 1, 0.5)
    assert distance is not None
    assert distance > 0


def test_disconnected_transition_is_rejected():
    nodes, arcs = _network()
    graph = DirectedRoadGraph(nodes, arcs)
    assert graph.transition_distance(0, 0.5, 2, 0.5) is None


def test_decoder_preserves_missing_observation_as_gap():
    nodes, arcs = _network()
    candidates = pd.DataFrame([
        [0, 0, 0.5, 0.9, np.nan],
        [2, 1, 0.5, 0.8, np.nan],
    ], columns=[
        "observation_id", "candidate_arc_id", "projection_fraction",
        "positive_probability", "gps_step_distance_m",
    ])
    result = decode_trip(candidates, [0, 1, 2], arcs, nodes)
    assert result.loc[1, "decode_status"] == "no_candidates"
    assert pd.isna(result.loc[1, "decoded_arc_id"])


def test_decoder_marks_disconnected_restart():
    nodes, arcs = _network()
    candidates = pd.DataFrame([
        [0, 0, 0.5, 0.9, np.nan],
        [1, 2, 0.5, 0.8, 10.0],
    ], columns=[
        "observation_id", "candidate_arc_id", "projection_fraction",
        "positive_probability", "gps_step_distance_m",
    ])
    result = decode_trip(candidates, [0, 1], arcs, nodes)
    assert result.loc[1, "decode_status"] == "disconnected_restart"
    assert result["decoded_arc_id"].tolist() == [0, 2]


def test_shortest_path_returns_intermediate_arc_ids():
    nodes = pd.DataFrame({
        "longitude": [54.0, 54.001, 54.002, 54.003],
        "latitude": [56.0, 56.0, 56.0, 56.0],
    })
    arcs = pd.DataFrame({
        "arc_id": [10, 11, 12],
        "from_node_id": [0, 1, 2],
        "to_node_id": [1, 2, 3],
    })
    graph = DirectedRoadGraph(nodes, arcs)
    assert graph.shortest_path_arc_ids(1, 3) == [11, 12]


def test_reconstruction_includes_intermediate_roads_and_no_duplicate_same_arc():
    nodes, arcs = _network()
    decoded = pd.DataFrame([
        [0, 0, 0.2, 0.9, "decoded"],
        [1, 0, 0.8, 0.8, "decoded"],
        [2, 1, 0.5, 0.7, "decoded"],
    ], columns=[
        "observation_id", "decoded_arc_id", "decoded_projection_fraction",
        "selected_probability", "decode_status",
    ])
    result = reconstruct_trajectory(decoded, arcs, nodes)
    assert len(result) == 1
    assert result.iloc[0]["arc_sequence"] == [0, 1]
    assert result.iloc[0]["observation_ids"] == [0, 1, 2]


def test_reconstruction_preserves_missing_and_disconnected_breaks():
    nodes, arcs = _network()
    decoded = pd.DataFrame([
        [0, 0, 0.5, "decoded"],
        [1, pd.NA, np.nan, "no_candidates"],
        [2, 1, 0.5, "decoded"],
        [3, 2, 0.5, "disconnected_restart"],
    ], columns=[
        "observation_id", "decoded_arc_id", "decoded_projection_fraction",
        "decode_status",
    ])
    decoded["decoded_arc_id"] = decoded["decoded_arc_id"].astype("Int64")

    result = reconstruct_trajectory(decoded, arcs, nodes)

    assert len(result) == 3
    assert result["arc_sequence"].tolist() == [[0], [1], [2]]
    assert result["segment_start_reason"].tolist() == [
        "trip_start",
        "after_gap",
        "disconnected_restart",
    ]