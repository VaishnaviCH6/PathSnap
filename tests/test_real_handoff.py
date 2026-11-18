from pathlib import Path

import pytest

from src.evaluation.data_validation import validate_m2_handoff


CANDIDATE_CSV = Path("dataset/m2_handoff/train_labelled_candidates.csv")
STATUS_CSV = Path("dataset/m2_handoff/train_observation_status.csv")


@pytest.mark.skipif(
    not CANDIDATE_CSV.is_file() or not STATUS_CSV.is_file(),
    reason="Member 2 real-data handoff is not present",
)
def test_real_handoff_validation(tmp_path):
    result = validate_m2_handoff(CANDIDATE_CSV, STATUS_CSV, tmp_path)

    assert result["candidate_rows"] == 1_084_912
    assert result["labeled_candidate_rows"] == 567_307
    assert result["unlabeled_candidate_rows"] == 517_605
    assert result["unique_training_trips"] == 80
    assert result["positive_labels"] == 111_306
    assert result["negative_labels"] == 456_001
    assert result["duplicate_observation_candidate_keys"] == 0
    assert result["eligible_observations_missing_positive"] == 0
    assert all(count == 0 for count in result["infinite_features"].values())
