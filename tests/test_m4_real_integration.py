"""Synthetic integration tests. No reported experimental model accuracy.

Run in a clone containing the actual M2 src modules.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from backend.main import create_app
from backend.m2_bridge import FEATURE_COLUMNS, prepare_candidates
from backend.pipeline import Prediction, TripData
from backend.real_inference import load_pipeline


def synthetic_trip():
    return TripData(
        "00000000",
        np.array([[8.00025, 49.00000, 0], [8.00075, 49.00000, 1]], dtype=float),
        np.array([[8.0, 49.0], [8.001, 49.0]], dtype=float),
        np.array([[0, 1]], dtype=int),
    )


def synthetic_decoder(*, trip, candidate_features, candidate_scores):
    assert list(candidate_scores.columns) == ["trip_id", "observation_id",
                                               "candidate_arc_id", "emission_probability"]
    assert set(candidate_scores["candidate_arc_id"]) == {0}
    assert candidate_scores["emission_probability"].between(0, 1).all()
    assert list(candidate_features.columns[1:4]) == [
        "observation_id", "candidate_arc_id", "distance_to_segment_m"
    ]
    obs = [None] * len(trip.track)
    for obs_id, arc_id in candidate_scores[["observation_id", "candidate_arc_id"]].itertuples(index=False, name=None):
        obs[int(obs_id)] = int(arc_id)
    return Prediction(obs, [0])


def test_actual_m2_bridge_generates_features_and_keeps_ids():
    trip = synthetic_trip()
    prepared = prepare_candidates(trip)
    assert list(prepared.features["observation_id"].unique()) == [0, 1]
    assert list(prepared.features["candidate_arc_id"].unique()) == [0]
    assert list(prepared.features["trip_id"].unique()) == ["00000000"]
    assert set(FEATURE_COLUMNS).issubset(prepared.features.columns)
    assert len(prepared.status) == 2


def test_m2_no_candidate_keeps_empty_status():
    trip = synthetic_trip()
    track = trip.track.copy()
    track[-1, 0] = 9.0
    far_trip = TripData(trip.trip_id, track, trip.nodes, trip.arcs)
    prepared = prepare_candidates(far_trip)
    assert prepared.status.candidate_count.tolist() == [1, 0]


def test_real_factory_calls_saved_model_and_decoder(monkeypatch, tmp_path):
    # This test model is deliberately fitted on fake unit-test rows only.
    x = pd.DataFrame(np.ones((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    model = DummyClassifier(strategy="prior").fit(x, [0, 1, 1, 0])
    model_path = tmp_path / "synthetic.joblib"
    joblib.dump(model, model_path)
    config_path = tmp_path / "synthetic.json"
    config_path.write_text(json.dumps({"feature_columns": FEATURE_COLUMNS,
                                      "positive_class": 1}))
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(model_path))
    monkeypatch.setenv("PATHSNAP_MODEL_CONFIG", str(config_path))
    monkeypatch.setenv("PATHSNAP_HMM_DECODER",
                       "tests.test_m4_real_integration:synthetic_decoder")
    predict = load_pipeline()
    pred = predict(synthetic_trip())
    assert pred == Prediction([0, 0], [0])


def test_real_factory_refuses_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("PATHSNAP_MODEL_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("PATHSNAP_HMM_DECODER",
                       "tests.test_m4_real_integration:synthetic_decoder")
    with pytest.raises(FileNotFoundError):
        load_pipeline()
