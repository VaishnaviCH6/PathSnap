"""M4 contract tests against the real M2 Viterbi implementation."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.m2_hmm_adapter import decode_trip_for_m4
from backend.pipeline import TripData, validate_prediction
from backend.main import create_app
from frontend.app import build_map


def sample_trip():
    # Two disconnected directed components: 0->1 and 2->3.
    return TripData(
        trip_id="00000000",
        track=np.array([[8.0002, 49.0, 0], [8.0012, 49.0, 1],
                        [8.0102, 49.0, 2], [8.0108, 49.0, 3]], dtype=float),
        nodes=np.array([[8.0, 49.0], [8.002, 49.0],
                        [8.01, 49.0], [8.012, 49.0]], dtype=float),
        arcs=np.array([[0, 1], [2, 3]], dtype=int),
    )


def candidate_frames(trip):
    data = pd.DataFrame({
        "trip_id": [trip.trip_id] * 3,
        "observation_id": [0, 1, 3],
        "candidate_arc_id": [0, 0, 1],
        "projection_fraction": [0.1, 0.6, 0.4],
        "gps_step_distance_m": [np.nan, 90, 800],
    })
    scores = data[["trip_id", "observation_id", "candidate_arc_id"]].copy()
    scores["emission_probability"] = [0.9, 0.7, 0.8]
    return data, scores


def test_adapter_preserves_missing_and_disconnected_route_breaks():
    trip = sample_trip()
    features, scores = candidate_frames(trip)
    pred = decode_trip_for_m4(trip=trip, candidate_features=features, candidate_scores=scores)
    assert pred.observation_arc_ids == [0, 0, None, 1]
    assert pred.route_arc_ids == [0, 1]
    assert pred.route_break_indices == [1]
    validate_prediction(trip, pred)


def test_adapter_rejects_mismatched_candidate_scores():
    trip = sample_trip()
    features, scores = candidate_frames(trip)
    with pytest.raises(ValueError, match="score rows must correspond"):
        decode_trip_for_m4(trip=trip, candidate_features=features,
                           candidate_scores=scores.iloc[:1])

def test_integrated_api_reports_breaks_and_frontend_does_not_bridge_them(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PATHSNAP_EVALUATOR", raising=False)
    trip = sample_trip()
    root = tmp_path / "student_test"
    for number in range(10):
        tid = f"{number:08d}"
        target = root / tid
        target.mkdir(parents=True)
        np.savetxt(target / f"{tid}.track", trip.track)
        np.savetxt(target / f"{tid}.nodes", trip.nodes)
        np.savetxt(target / f"{tid}.arcs", trip.arcs, fmt="%d")
        (target / f"{tid}.route").write_text("NEVER READ")

    def factory():
        def predict(loaded):
            features, scores = candidate_frames(loaded)
            return decode_trip_for_m4(trip=loaded, candidate_features=features,
                                      candidate_scores=scores)
        return predict

    with TestClient(create_app(root=root, mode="integrated", factory=factory)) as client:
        response = client.post("/reconstruct", json={"trip_id": "00000000"})
        assert response.status_code == 200
        value = response.json()
        assert value["route_break_indices"] == [1]
        assert value["route_segment_count"] == 2
        assert value["route_continuity"] == 0
        assert value["accuracy"] is None
        assert len(value["segments"]) == 2
        assert "L.polyline" in build_map(value)
