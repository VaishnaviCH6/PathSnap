"""Integration smoke tests with the REAL approved M3 handoff artifact.

These tests do NOT evaluate real route quality or held-out accuracy.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from backend.m2_bridge import FEATURE_COLUMNS
from backend.pipeline import Prediction, TripData
from backend.real_inference import _load_m3_components, load_model_only_pipeline, load_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'artifacts/m3/random_forest_tuned_20261118T123057Z.joblib'


def configure(monkeypatch):
    if not MODEL.exists():
        pytest.skip("Approved M3 artifact not installed (code-only M4 package)")
    monkeypatch.setenv('PATHSNAP_MODEL_PATH',str(MODEL))
    monkeypatch.setenv('PATHSNAP_MODEL_CONFIG',str(MODEL.with_suffix('.json')))


def test_real_m3_handoff_metadata_and_sample_scores(monkeypatch):
    configure(monkeypatch)
    model, metadata, columns, positive_index = _load_m3_components()
    assert columns == FEATURE_COLUMNS
    assert metadata['feature_schema_version'] == 'M2-v1'
    assert int(model.classes_[positive_index]) == 1
    sample = pd.read_csv(ROOT / 'docs/m3_sample_candidates.csv')
    scores = model.predict_proba(sample[columns])[:,positive_index]
    assert scores.shape == (len(sample),)
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()


def test_real_m3_scores_actual_m2_features(monkeypatch):
    configure(monkeypatch)
    # Small geometry fixture from synthetic values; M2 still generates ALL features.
    trip = TripData('00000000',
                    np.array([[8.00025,49.0,0],[8.00075,49.0,1]], dtype=float),
                    np.array([[8.,49.],[8.001,49.]], dtype=float),
                    np.array([[0,1]], dtype=int))
    pred = load_model_only_pipeline()(trip)
    assert pred == Prediction([0,0],[0])


def test_integrated_mode_refuses_to_invent_decoder(monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv('PATHSNAP_HMM_DECODER', raising=False)
    with pytest.raises(ValueError, match='PATHSNAP_HMM_DECODER'):
        load_pipeline()


def test_model_artifact_integrity_check(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv('PATHSNAP_MODEL_SHA256','0'*64)
    with pytest.raises(ValueError, match='checksum'):
        _load_m3_components()
