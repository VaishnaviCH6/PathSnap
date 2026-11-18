# PathSnap Member 3: Machine Learning and Optimisation

## Responsibility

Member 3 provides candidate-road binary classification, trip-aware model evaluation, Bayesian hyperparameter optimisation, model artifacts, and an inference interface. It does not implement HMM/Viterbi decoding or route reconstruction.

## Models

- Logistic Regression: linear baseline with scaling and median imputation.
- K-Nearest Neighbours: nonlinear local-distance baseline with scaling and imputation.
- Decision Tree: interpretable nonlinear baseline.
- Random Forest: ensemble baseline for nonlinear interactions.
- HistGradientBoosting: efficient boosted baseline for large numeric feature sets.

All models expose sklearn `.fit()`, `.predict()`, and `.predict_proba()` through pipelines or compatible estimators.

## Synthetic versus Real Data

Synthetic data in tests and the notebook is used only to verify code paths. It must not be reported as PathSnap accuracy or generalisation performance. Real training requires Member 2 candidate-level feature and label data.

## Leakage Prevention

`GroupKFold` receives `trip_id`. Every trip is fully contained in either the training or validation partition for a fold. Imputation and scaling are fitted inside each training fold. Hyperparameter optimisation uses the same group-aware objective.

## Training Interface

```python
from src.models.training import train_and_evaluate_models

results = train_and_evaluate_models(
    candidate_features,
    feature_columns=["distance_m", "heading_difference", "speed_mps", "time_difference"],
    label_column="label",
    trip_id_column="trip_id",
    n_splits=5,
    random_state=42,
)
```

The data frame must contain the label and grouping columns. Feature identifiers such as `trip_id`, `gps_id`, `candidate_id`, `arc_id`, and ground-truth fields must not be predictive features.

## Metrics

Primary candidate-ranking metric: mean cross-validation F1. Secondary metrics include accuracy, precision, recall, ROC-AUC, confusion matrices, and training/inference durations. ROC-AUC is `None` when a fold does not contain both classes.

## Bayesian Optimisation

```python
from src.optimization.bayesian_optimization import optimize_model

result = optimize_model(
    candidate_features,
    "Logistic Regression",
    feature_columns=["distance_m", "heading_difference", "speed_mps", "time_difference"],
    n_trials=20,
    n_splits=5,
    random_seed=42,
)
```

Optuna uses a TPE sampler and maximises mean fold F1. The returned tuning-fold value must not be treated as held-out generalisation performance.

## Model Saving

```python
from src.models.model_io import save_model

model_path, metadata_path = save_model(
    model,
    metadata,
    output_directory="artifacts",
    model_name="Random Forest",
)
```

Artifacts store model parameters and metadata. Joblib files from untrusted sources must not be loaded.

## Prediction Contract

Input columns:

- Feature columns declared in artifact metadata
- Optional `trip_id`
- Optional `gps_id`
- Optional `candidate_id`
- Optional `arc_id`

Output columns:

- Preserved identifier columns
- `predicted_label`
- `positive_probability`
- `model_type`

```python
from src.models.prediction import predict_candidates

output = predict_candidates(candidate_features, "artifacts/model.joblib")
```

Member 2 can preserve candidates and probabilities for HMM emissions. Member 4 can expose this interface through a backend service.

## Commands

Install dependencies:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run all tests:

```powershell
& .\.venv\Scripts\python.exe -m pytest -v
```

Run one component:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/test_group_training.py tests/test_optimization.py -v
```

## Outstanding Data Dependencies

Member 2 must provide candidate-level numeric feature names, valid binary labels, candidate IDs, GPS IDs, arc IDs, and trip IDs. Member 1 must provide agreed permitted 80-trip and held-out 10-trip partitions. The private evaluator partition must never be used for training or tuning.

## Known Limitations

- No real PathSnap training benchmark is claimed.
- Candidate-ranking and route-reconstruction accuracy are separate metrics.
- Probabilities are not calibrated unless calibration is separately evaluated.
- End-to-end HMM/Viterbi integration is blocked until Member 2 provides its interface and test data.
