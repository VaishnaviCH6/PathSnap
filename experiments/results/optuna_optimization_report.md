# Random Forest Optuna Optimisation Report

## Status

**COMPLETE for a bounded real-data subset study.** Six of six trials completed in persistent SQLite-backed Optuna storage.

This is tuning evidence on 12 representative trips, not an equivalent full-data GroupKFold baseline and not an unbiased held-out estimate.

## Pilot

- Dataset subset: 12 trips
- Subset rows: 75,678
- Folds: 2
- Trial count: 1
- Random seed: 42
- Configuration: 40 estimators, depth 12
- Completed objective F1: 0.842380
- Trial duration: 4.6688 seconds

## Tuning subset

The subset was selected deterministically by trip-level row count and positive-label proportion. It contains 12 trips: `35, 62, 36, 98, 44, 46, 38, 80, 50, 29, 58, 5`.

- Subset rows: 75,678
- Subset trips: 12
- GroupKFold folds: 3
- Random seed: 42
- Objective: mean cross-validation F1
- Storage: `experiments/results/random_forest_optuna.db`
- CSV: `experiments/results/optuna_trials.csv`
- JSON: `experiments/results/random_forest_optimization.json`

## Completed trials

| Trial | Status | Mean F1 | Duration (s) | Estimators | Depth | Min Split | Min Leaf | Max Features | Class Weight |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | COMPLETE | 0.777493 | 10.7608 | 60 | 8 | 2 | 5 | log2 | None |
| 1 | COMPLETE | 0.889679 | 25.2305 | 60 | 20 | 4 | 4 | 0.5 | balanced |
| 2 | COMPLETE | 0.873671 | 10.8433 | 40 | 20 | 2 | 5 | sqrt | balanced |
| 3 | COMPLETE | 0.873148 | 20.5226 | 80 | 20 | 7 | 2 | log2 | None |
| 4 | COMPLETE | 0.884912 | 44.5978 | 120 | 16 | 2 | 2 | 0.5 | None |
| 5 | COMPLETE | 0.874324 | 20.3311 | 80 | 20 | 3 | 1 | sqrt | None |

## Best completed trial

- Trial: 1
- Mean cross-validation F1: 0.8896789542
- Fold F1: 0.9392767356, 0.8673803802, 0.8623797468
- Fold ROC-AUC: 0.9914710423, 0.9872117899, 0.9723013286

Hyperparameters:

```json
{
  "n_estimators": 60,
  "max_depth": 20,
  "min_samples_split": 4,
  "min_samples_leaf": 4,
  "max_features": 0.5,
  "class_weight": "balanced"
}
```

## Independent same-subset comparison

Both configurations were evaluated on exactly the same 12 trips and identical 3 folds.

| Configuration | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean ROC-AUC | Mean Top-1 | Mean Train Time (s) | Mean Inference Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.955284 | 0.906528 | 0.849471 | 0.876232 | 0.981558 | 0.912079 | 7.528122 | 0.390090 |
| Tuned | 0.957681 | 0.862982 | 0.919948 | 0.889679 | 0.983661 | 0.920796 | 7.536322 | 0.205126 |

The tuned configuration improved mean F1 by 0.013447 and mean top-1 accuracy by 0.008717 on this same-subset protocol.

This improvement is measured only on the tuning subset. It is not a full-data held-out result. The full-data baseline remains the primary real-data evidence until a comparable independent held-out evaluation is available.

## Persistence

- SQLite: `experiments/results/random_forest_optuna.db`
- Incremental trial rows: `experiments/results/optuna_trials.csv`
- Study metadata and best parameters: `experiments/results/random_forest_optimization.json`
