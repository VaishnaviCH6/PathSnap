
from collections.abc import Mapping

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = (
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "HistGradientBoosting",
)


def get_models(
    random_state: int = 42,
    model_parameters: Mapping[str, Mapping] | None = None,
):
    """Return reproducible classification pipelines for candidate-road binary labels."""
    parameters = model_parameters or {}

    logistic_parameters = {
        "C": 1.0,
        "max_iter": 2000,
        "random_state": random_state,
        **parameters.get("Logistic Regression", {}),
    }
    knn_parameters = {
        "n_neighbors": 5,
        "metric": "minkowski",
        **parameters.get("KNN", {}),
    }
    tree_parameters = {
        "max_depth": 10,
        "min_samples_leaf": 1,
        "random_state": random_state,
        **parameters.get("Decision Tree", {}),
    }
    forest_parameters = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "random_state": random_state,
        "n_jobs": -1,
        **parameters.get("Random Forest", {}),
    }
    hist_parameters = {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "random_state": random_state,
        **parameters.get("HistGradientBoosting", {}),
    }

    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(**logistic_parameters),
        ),
        "KNN": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(**knn_parameters),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(**tree_parameters),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(**forest_parameters),
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(**hist_parameters),
        ),
    }
