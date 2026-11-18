# PathSnap Member 3: Real Model Training and Evaluation Report

**Date:** 18 November 2026

**Branch:** `ml-models`

**Dataset:** Member 2 training handoff

**Validation:** Five-fold trip-grouped cross-validation

**Primary metric:** Mean fold F1-score
**Status:** Baseline five-model experiment complete; bounded Random Forest tuning complete; final held-out student-test and HMM/Viterbi route evaluation pending

## 1. Dataset Verification

The ZIP `C:\Users\ADMIN\Downloads\M2_training_handoff.zip` was readable and extracted flat into:

`C:\Users\ADMIN\Downloads\PathSnap\dataset\m2_handoff`

Extracted files:

- `train_labelled_candidates.csv`
- `train_observation_status.csv`
- `FEATURES.md`

The dataset files are excluded by the repository's existing `dataset/` `.gitignore` rule.

### Actual candidate dataset counts

| Property | Verified value |
|---|---:|
| Candidate rows | 1,084,912 |
| Labeled candidate rows | 567,307 |
| Unlabeled candidate rows | 517,605 |
| Positive labels | 111,306 |
| Negative labels | 456,001 |
| Positive class proportion | 19.62% |
| Unique training trips | 80 |
| Eligible observations | 111,306 |
| Duplicate observation-candidate keys | 0 |
| Eligible observations missing a positive label | 0 |

The label distribution is reproduced as 456,001 negatives and 111,306 positives. Missing labels were excluded from supervised fitting and were never replaced with zero.

### Observation status

| Status | Count |
|---|---:|
| Candidates present | 189,902 |
| No generated candidates | 18,230 |
| Candidate miss | 285 |
| Aligned observations | 112,313 |
| Ambiguous alignments | 76,112 |
| No route alignment | 19,992 |

The handoff also defines 111,306 supervised-eligible observations. Every eligible observation has exactly one positive candidate label. This validates label/status agreement for the delivered candidate table, but it does not independently prove route-alignment correctness.

### Feature missing and invalid values

The ten predictive features are all numeric. Missing values were observed in movement-derived fields:

| Feature | Missing in candidate rows |
|---|---:|
| `direction_difference_deg` | 534 |
| `gps_step_distance_m` | 534 |
| `time_difference_s` | 534 |
| `gps_speed_mps` | 534 |
| `gps_heading_deg` | 534 |
| `gps_heading_change_deg` | 1,054 |

No infinite values were detected in the ten feature columns. The existing median imputation pipeline handles genuine missing values.

### Schema and identifiers

The required candidate columns are present:

- `trip_id`
- `observation_id`
- `candidate_arc_id`
- `distance_to_segment_m`
- `direction_difference_deg`
- `gps_step_distance_m`
- `time_difference_s`
- `gps_speed_mps`
- `segment_length_m`
- `projection_fraction`
- `gps_heading_deg`
- `road_bearing_deg`
- `gps_heading_change_deg`
- `label`

The Handoff CSV stores trip identifiers as strings in the numeric form supplied by Member 2. They are loaded as pandas strings and used only as grouping keys. Predictive features exclude `trip_id`, `observation_id`, `candidate_arc_id`, `label`, and ground-truth fields.

## 2. Training Methodology

A 50,000-row smoke comparison verified the runner and all five model pipelines. The final experiment then used all 567,307 labeled candidate rows.

Cross-validation used five-fold `GroupKFold`, grouped by `trip_id`:

- 80 unique training trips;
- 64 trips in each training partition;
- 16 trips in each validation partition;
- approximately 453,000 training rows per fold;
- approximately 113,000 validation rows per fold;
- identical splits for all five models.

A pipeline clone was fitted for every model and fold. Median imputation and `StandardScaler`, where applicable, were fitted inside each training fold. No trip appeared in both partitions of a fold.

All five models used these ten features:

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

## 3. Actual Five-Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Top-1 Candidate Accuracy | Mean Training Time (s) | Mean Inference Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.941328 | 0.865782 | 0.840560 | 0.852856 | 0.975768 | 0.881163 | 145.768374 | 3.125984 |
| HistGradientBoosting | 0.938244 | 0.844570 | 0.853335 | 0.848818 | 0.974858 | 0.872329 | 4.235474 | 0.455638 |
| KNN | 0.929994 | 0.825531 | 0.831403 | 0.828291 | 0.947426 | 0.865198 | 1.741898 | 57.910523 |
| Decision Tree | 0.926072 | 0.807714 | 0.827468 | 0.817371 | 0.960129 | 0.849343 | 11.721761 | 0.112070 |
| Logistic Regression | 0.822222 | 0.566300 | 0.416373 | 0.478810 | 0.868097 | 0.556101 | 0.852298 | 0.047074 |

### Fold standard deviations

| Model | Accuracy Std | Precision Std | Recall Std | F1 Std | ROC-AUC Std | Top-1 Std |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.026637 | 0.046012 | 0.058963 | 0.052234 | 0.015475 | 0.055270 |
| HistGradientBoosting | 0.028282 | 0.050798 | 0.055302 | 0.052239 | 0.013972 | 0.058951 |
| KNN | 0.031412 | 0.049703 | 0.062270 | 0.055031 | 0.029253 | 0.058773 |
| Decision Tree | 0.024396 | 0.035292 | 0.046074 | 0.039987 | 0.018455 | 0.041043 |
| Logistic Regression | 0.029577 | 0.020432 | 0.036762 | 0.025289 | 0.021696 | 0.109183 |

These values are calculated from completed folds and are stored in `experiments/results/real_fold_metrics.csv`.

## 4. Best Baseline Model

The selected baseline winner is **Random Forest**.

Evidence:

- Highest mean cross-validation F1: `0.852856`;
- Highest mean top-1 candidate accuracy: `0.881163`;
- Highest mean ROC-AUC: `0.975768`;
- Highest mean accuracy: `0.941328`;
- Highest mean precision: `0.865782`;
- Recall: `0.840560`.

The choice is based on measured validation F1, while top-1 accuracy and ROC-AUC support the result. HistGradientBoosting is close in every primary metric, particularly F1 (`0.848818`) and ROC-AUC (`0.974858`), so it remains a strong candidate. Random Forest also has substantially higher mean training time (`145.768374 s`) than HistGradientBoosting (`4.235474 s`) and much higher inference cost than Decision Tree or Logistic Regression.

## 5. Bayesian Optimisation

A bounded Optuna TPE study completed six trials on a representative subset of 12 trips containing 75,678 labeled rows. The subset was sampled deterministically by trip row count and positive-label proportion. Three GroupKFold folds were used with random seed 42.

The subset objective was mean fold F1. Trial 1 was best, with an objective value of `0.8896789542` and parameters:

- 60 estimators;
- maximum depth 20;
- minimum samples split 4;
- minimum samples leaf 4;
- 50% max features;
- balanced class weight.

A matched subset comparison was run for baseline and tuned configurations on identical trips and folds:

| Configuration | Mean F1 | Mean Accuracy | Mean Precision | Mean Recall | Mean ROC-AUC | Mean Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Random Forest | 0.876232 | 0.955284 | 0.906528 | 0.849471 | 0.981558 | 0.912079 |
| Tuned Random Forest | 0.889679 | 0.957681 | 0.862982 | 0.919948 | 0.983661 | 0.920796 |

The tuned configuration improved subset mean F1 by `0.013447` and subset top-1 accuracy by `0.008717`. It was not evaluated against a held-out student-test partition, so subset tuning scores are not presented as unbiased generalisation estimates.

## 6. Saved Model Artifacts

The original baseline Random Forest remains saved:

- `artifacts/random_forest_20261118T114843Z.joblib`
- `artifacts/random_forest_20261118T114843Z.json`

The tuned configuration was retrained on all 567,307 labeled candidate rows because it improved the matched subset protocol:

- `artifacts/random_forest_tuned_20261118T123057Z.joblib`
- `artifacts/random_forest_tuned_20261118T123057Z.json`

The tuned artifact configuration records:

- 60 estimators;
- maximum depth 20;
- minimum samples split 4;
- minimum samples leaf 4;
- max features 0.5;
- balanced class weight;
- random seed 42;
- one worker;
- ten M2-v1 feature columns;
- 567,307 full labeled training rows.

Reload verification succeeded:

- 100 sample predictions reproduced successfully;
- probability matrix shape: `(100, 2)`;
- positive probability range: `0.0` to `0.9999404762`.

## 7. Label Alignment Limitation

The training labels are Member 2-generated reference alignment labels. The status table reports 76,112 ambiguous observations and 19,992 observations with no route alignment. This means binary candidate labels should not be described as independently verified route truth until alignment quality is assessed separately.

The experiment evaluates agreement with the delivered generated labels. It does not prove route reconstruction accuracy.

## 8. Route-Level and Held-Out Status

- Candidate-level metrics: complete for five models using delivered training labels;
- Observation-level top-1 ranking: complete for cross-validation folds;
- Bayesian optimization: a bounded 12-trip, 3-fold, six-trial study completed with persistent SQLite storage;
- Student-test held-out evaluation: not completed;
- Evaluator-test evaluation: not performed;
- HMM/Viterbi integration: not verified;
- Route reconstruction accuracy: not measured.

The 10 student-test trips and private evaluator trips must remain outside model selection and tuning. No such held-out PathSnap partition was supplied or used.

## 9. Evidence Files

### Data validation

- `experiments/results/data_validation.csv`
- `experiments/results/data_validation.json`

### Real baseline results

- `experiments/results/real_model_comparison.csv`
- `experiments/results/real_fold_metrics.csv`
- `experiments/results/model_comparison.png`
- `experiments/results/confusion_matrices.png`
- `experiments/results/real_baseline_metadata.json`
- `experiments/results/optuna_trials.csv`
- `experiments/results/random_forest_optimization.json`
- `experiments/results/random_forest_optuna.db`
- `experiments/results/optuna_optimization_report.md`
- `experiments/results/tuned_vs_baseline_subset.csv`

### Synthetic smoke evidence

- `experiments/results/real_smoke_model_comparison.csv`
- `experiments/results/real_smoke_fold_metrics.csv`
- `experiments/results/real_smoke_model_comparison.png`
- `experiments/results/real_smoke_confusion_matrices.png`

Synthetic smoke results are not used for model selection.

## 10. Testing

Full pytest execution:

```text
26 passed in 9.22 seconds
Python 3.13.6
pytest 9.2.0
```

Tests cover model factories, real handoff counts, missing and infinite values, group isolation, metrics, serialization, prediction, Optuna smoke behavior and integration contracts.

## 11. Git Status

Active branch: `ml-models`

Code and report changes are separate from the raw dataset. `dataset/` is excluded by `.gitignore`, and no raw training CSV or ZIP was committed or pushed. No commit or push was created during this workflow.

## 12. Completion Status

| Requirement | Status |
|---|---|
| ZIP extraction | COMPLETE |
| Dataset validation | COMPLETE |
| Real five-model baseline | COMPLETE |
| GroupKFold cross-validation | COMPLETE |
| Candidate top-1 ranking | COMPLETE |
| Real model comparison | COMPLETE |
| Best baseline selection | COMPLETE |
| Bayesian optimisation | COMPLETE for bounded representative subset tuning |
| Tuned-versus-baseline subset comparison | COMPLETE |
| Tuned model artifact | COMPLETE |
| Final baseline artifact | COMPLETE |
| Automated testing | COMPLETE |
| Student-test held-out evaluation | BLOCKED - no partition supplied |
| HMM/Viterbi integration | BLOCKED - separate Member 2 integration unavailable |
| Route-level accuracy | BLOCKED - HMM/Viterbi and held-out route evaluation unavailable |
