import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_validation import get_group_splits, resolve_feature_columns


def test_resolve_feature_columns_excludes_contract_fields():
    data = pd.DataFrame({
        "trip_id": [1, 2],
        "gps_id": [10, 11],
        "arc_id": [20, 21],
        "label": [0, 1],
        "ground_truth_route": [1, 1],
        "distance_m": [5.0, 7.0],
    })
    features = resolve_feature_columns(data)
    assert features == ["distance_m"]


def test_insufficient_unique_trips():
    data = pd.DataFrame({"trip_id": ["only", "only"]})
    with pytest.raises(ValueError, match="at least two"):
        get_group_splits(data)


def test_missing_trip_id_column():
    with pytest.raises(ValueError, match="grouping"):
        get_group_splits(pd.DataFrame({"distance_m": [1.0]}))
