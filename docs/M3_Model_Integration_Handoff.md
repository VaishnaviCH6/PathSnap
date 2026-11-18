# PathSnap Member 3 Model Integration Handoff

## Final model

Use the tuned baseline artifact selected by the matched subset tuning protocol:

- Model: `artifacts/random_forest_tuned_20261118T123057Z.joblib`
- Metadata: `artifacts/random_forest_tuned_20261118T123057Z.json`
- Model name: Random Forest Tuned
- Training rows: 567,307 labeled candidate rows
- Training trips: 80
- Feature schema: M2-v1
- Random seed: 42

## Required feature order

The input DataFrame must provide these numeric features in this exact order:

1. `distance_to_segment_m`
2. `direction_difference_deg`
3. `gps_step_distance_m`
4. `time_difference_s`
5. `gps_speed_mps`
6. `segment_length_m`
7. `projection_fraction`
8. `gps_heading_deg`
9. `road_bearing_deg`
10. `gps_heading_change_deg`

The model metadata is authoritative. Member 4 can read `feature_columns` and reorder input columns before inference.

## Identifier columns

Preserve these optional identifiers alongside predictions:

- `trip_id`
- `observation_id`
- `candidate_arc_id`

Additional supported identifier names are `gps_id`, `candidate_id`, and `arc_id`.

Identifiers must not be used as predictive features.

## Input schema

```text
trip_id                 string or integer
observation_id          integer
candidate_arc_id        integer
one row for one candidate road
10 numeric feature columns
```

Missing numeric values are valid only where the source schema permits them. The saved pipeline contains median imputation. Infinite values should be rejected or converted to unavailable before prediction. Missing candidate observations should remain absent or be explicitly excluded; they must not become class 0 labels.

## Loading and predicting

```python
from src.models.prediction import predict_candidates

output = predict_candidates(
    candidate_dataframe,
    "artifacts/random_forest_tuned_20261118T123057Z.joblib",
)
```

The output contains:

- preserved `trip_id`, `observation_id`, and `candidate_arc_id` when present;
- `predicted_label`, with `1` as the positive candidate-road label and `0` as negative;
- `positive_probability`, in `[0, 1]`;
- `model_type`.

The result can also be obtained with the lower-level model IO API:

```python
from src.models.model_io import load_model, validate_feature_frame

model, metadata = load_model("artifacts/random_forest_tuned_20261118T123057Z.joblib")
features = validate_feature_frame(candidate_dataframe, metadata["feature_columns"])
labels = model.predict(features)
probability_matrix = model.predict_proba(features)
positive_probability = probability_matrix[:, 1]
```

## HMM/Viterbi handoff

Member 2 should group the returned candidates by `trip_id` and `observation_id`, then pass positive probabilities to its emission model. Candidate candidate probability must remain associated with `candidate_arc_id`.

Missing candidates or observations should not receive an invented positive or negative label. They should remain explicit unavailable states for Member 2's candidate-generation/HMM integration logic.

Probability values are uncalibrated candidate scores unless a separate calibration evaluation is performed.

## FastAPI handoff

Member 4 can load the model once during application startup or after an approved deployment artifact is copied into the service environment. The FastAPI request should provide the ten numeric features and optional identifiers. The endpoint should return the identifier columns, predicted label and positive probability.

Do not place raw `.joblib` files from untrusted sources in an application. Verify artifact provenance and load only the approved model artifact.

## Working example

A repository-root example is available:

```powershell
& .\.venv\Scripts\python.exe scripts/predict_tuned_model.py
```

## Held-out status

The final tuned artifact was retrained on the permitted labeled training rows. No valid 10-trip student-test partition or private evaluator partition was available, so no held-out or route reconstruction result is claimed here.
