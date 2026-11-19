"""Regression tests for the M4-only speed path; no reference labels are accessed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.fast_hmm import FastRoadGraph
from backend.m2_hmm_adapter import decode_trip_for_m4
from backend.pipeline import Prediction, TripData
from src.map_matching.hmm import DirectedRoadGraph


@pytest.mark.parametrize('seed', list(range(5)))
def test_fast_shortest_paths_match_m2_on_directed_graphs(seed):
    rng = np.random.default_rng(seed)
    count = 15
    nodes = pd.DataFrame({
        'longitude': 8 + rng.normal(0, .002, count),
        'latitude': 49 + rng.normal(0, .002, count),
    })
    edges = np.stack([rng.integers(0, count, size=45),
                      rng.integers(0, count, size=45)], axis=1)
    # Both parallel arcs and self-loops are common in source road datasets.
    edges = np.concatenate([edges, edges[:4]], axis=0)
    arcs = pd.DataFrame(edges, columns=['from_node_id', 'to_node_id'])
    arcs.insert(0, 'arc_id', np.arange(len(arcs)))
    old = DirectedRoadGraph(nodes, arcs)
    fast = FastRoadGraph(nodes, arcs)
    assert set(old.arc_info) == set(fast.arc_info)
    for source in range(count):
        for target in range(count):
            assert fast.shortest_path_arc_ids(source, target) == old.shortest_path_arc_ids(source, target)
    for prev_arc, curr_arc in [(0, 1), (1, 3), (5, 7), (12, 22)]:
        x = old.transition_distance(prev_arc, .5, curr_arc, .4)
        y = fast.transition_distance(prev_arc, .5, curr_arc, .4)
        if x is None:
            assert y is None
        else:
            assert y == pytest.approx(x, abs=1e-5)


def test_m4_fast_and_original_reconstruct_identically(monkeypatch):
    trip = TripData(
        trip_id='00000000',
        track=np.array([[8.0001, 49., 0], [8.0008, 49., 1],
                        [8.0015, 49., 2], [8.0095, 49., 3]], dtype=float),
        nodes=np.array([[8., 49.], [8.001, 49.], [8.002, 49.],
                        [8.009, 49.], [8.010, 49.]], dtype=float),
        arcs=np.array([[0, 1], [1, 2], [2, 1], [3, 4]], dtype=int),
    )
    feature = pd.DataFrame({
        'trip_id': [trip.trip_id] * 6,
        'observation_id': [0, 1, 1, 2, 2, 3],
        'candidate_arc_id': [0, 0, 1, 1, 2, 3],
        'projection_fraction': [.1, .9, .1, .7, .1, .4],
        'gps_step_distance_m': [np.nan, 80, 80, 80, 80, 500],
    })
    scores = feature[['trip_id', 'observation_id', 'candidate_arc_id']].copy()
    scores['emission_probability'] = [.99, .1, .9, .9, .1, .8]
    monkeypatch.setenv('PATHSNAP_HMM_FAST', '0')
    original = decode_trip_for_m4(trip=trip, candidate_features=feature, candidate_scores=scores)
    monkeypatch.setenv('PATHSNAP_HMM_FAST', '1')
    fast = decode_trip_for_m4(trip=trip, candidate_features=feature, candidate_scores=scores)
    assert original == fast
    assert fast.route_break_indices == [2]


def test_m4_prediction_cache_checks_input_content(monkeypatch):
    import backend.real_inference as inference
    counts = {'score': 0, 'decode': 0}

    def score(trip):
        counts['score'] += 1
        return pd.DataFrame(), pd.DataFrame()

    def decoder(*, trip, candidate_features, candidate_scores):
        counts['decode'] += 1
        return Prediction([0] * len(trip.track), [0])

    monkeypatch.setattr(inference, '_make_scorer', lambda: score)
    monkeypatch.setattr(inference, 'import_function', lambda spec: decoder)
    monkeypatch.setenv('PATHSNAP_HMM_DECODER', 'mock:decoder')
    monkeypatch.setenv('PATHSNAP_PREDICTION_CACHE_SIZE', '2')
    pipeline = inference.load_pipeline()
    trip = TripData('00000000', np.array([[8.,49.,0.]]),
                    np.array([[8.,49.],[8.001,49.]]), np.array([[0,1]]))
    assert pipeline(trip) == pipeline(trip)
    assert counts == {'score': 1, 'decode': 1}
    changed = TripData(trip.trip_id, np.array([[8.0002,49.,0.]]),trip.nodes,trip.arcs)
    pipeline(changed)
    assert counts == {'score': 2, 'decode': 2}


def test_m4_cache_can_be_disabled(monkeypatch):
    import backend.real_inference as inference
    calls = []
    monkeypatch.setenv('PATHSNAP_PREDICTION_CACHE_SIZE','0')
    monkeypatch.setenv('PATHSNAP_HMM_DECODER','mock:decoder')
    monkeypatch.setattr(inference,'_make_scorer',lambda: lambda t: (None,None))
    def decoder(*,trip,candidate_features,candidate_scores):
        calls.append(trip.trip_id)
        return Prediction([0] * len(trip.track),[0])
    monkeypatch.setattr(inference,'import_function',lambda s:decoder)
    predict=inference.load_pipeline()
    trip=TripData('00000000',np.array([[8.,49.,0.]]),
                  np.array([[8.,49.],[8.001,49.]]),np.array([[0,1]]))
    predict(trip);predict(trip)
    assert calls == ['00000000','00000000']
