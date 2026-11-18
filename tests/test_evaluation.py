import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_binary_metrics, calculate_candidate_ranking_metrics


def test_binary_metrics_and_confusion_matrix():
    result = calculate_binary_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]),
    )

    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_undefined_roc_auc_for_single_class_validation():
    result = calculate_binary_metrics([0, 0, 0], [0, 0, 1])
    assert result["roc_auc"] is None


def test_candidate_ranking_metrics():
    data = pd.DataFrame({
        "trip_id": ["a", "a", "b", "b"],
        "candidate_id": [10, 11, 20, 21],
        "label": [0, 1, 1, 0],
    })
    result = calculate_candidate_ranking_metrics(
        data,
        np.array([0.2, 0.8, 0.9, 0.1]),
    )
    assert result["top1_candidate_accuracy"] == 1.0
