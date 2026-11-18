import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.model_factory import get_models


@pytest.fixture(scope="module")
def synthetic_data():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def test_get_models_returns_exactly_five_models():
    models = get_models(random_state=42)

    assert len(models) == 5
    assert set(models) == {
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "HistGradientBoosting",
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "HistGradientBoosting",
    ],
)
def test_model_fit_predict_and_predict_proba(model_name, synthetic_data):
    X_train, X_test, y_train, _ = synthetic_data
    model = get_models(random_state=42)[model_name]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert set(predictions).issubset(set(y_train))
    assert np.all((predictions == 0) | (predictions == 1))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
