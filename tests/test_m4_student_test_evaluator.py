"""Tests for evaluation-only score computation; no fitted model required."""
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.student_test_evaluator import evaluate_observation_top1


def setup_labels(monkeypatch, tmp_path, trip_id="00000052"):
    label_dir = tmp_path / 'student_test'
    label_dir.mkdir()
    pd.DataFrame([
        {'observation_id': 0, 'candidate_arc_id': 11, 'label': 1},
        {'observation_id': 0, 'candidate_arc_id': 12, 'label': 0},
        {'observation_id': 1, 'candidate_arc_id': 22, 'label': 1},
        {'observation_id': 1, 'candidate_arc_id': 23, 'label': 0},
        {'observation_id': 2, 'candidate_arc_id': 32, 'label': float('nan')},
    ]).to_csv(label_dir / f'{trip_id}_labelled_candidates.csv', index=False)
    monkeypatch.setenv('PATHSNAP_MODE', 'integrated')
    monkeypatch.setenv('PATHSNAP_PARTITION', 'student_test')
    monkeypatch.setenv('PATHSNAP_STUDENT_TEST_LABEL_DIR', str(label_dir))
    return label_dir


def test_accuracy_excludes_unlabelled(monkeypatch, tmp_path):
    setup_labels(monkeypatch, tmp_path)
    prediction = SimpleNamespace(observation_arc_ids=[11, 23, 32])
    assert evaluate_observation_top1('00000052', prediction) == .5


def test_accuracy_excludes_missing_predictions(monkeypatch, tmp_path):
    setup_labels(monkeypatch, tmp_path)
    prediction = SimpleNamespace(observation_arc_ids=[11, None, 32])
    assert evaluate_observation_top1('00000052', prediction) == 1.


def test_cannot_use_evaluator_test(monkeypatch, tmp_path):
    setup_labels(monkeypatch, tmp_path)
    monkeypatch.setenv('PATHSNAP_PARTITION', 'evaluator_test')
    with pytest.raises(RuntimeError, match='only'):
        evaluate_observation_top1('00000052', SimpleNamespace(observation_arc_ids=[11]))


def test_missing_label_file_fails_not_fabricates_accuracy(monkeypatch, tmp_path):
    setup_labels(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_observation_top1('00000000', SimpleNamespace(observation_arc_ids=[11]))


def test_no_valid_supervision_fails_not_reports_zero(monkeypatch, tmp_path):
    setup_labels(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='No comparable'):
        evaluate_observation_top1('00000052', SimpleNamespace(observation_arc_ids=[None, None, 32]))
