# M4 – Student-test Top-1 Accuracy (evaluation-only)

**Purpose:** Display live decoded observation Top-1 accuracy for permitted
`student_test` trips after inference, without supplying reference labels to M2/M3
or reading `.route` files from the prediction pipeline.

This uses the approved outputs of `src/run_m2_labeled.py` generated from original
student-test reference routes. It is **not** route-sequence LCS F1. The latter
remains in `outputs/m2_student_test_evaluation/per_trip_comparison.csv` and
`overall_comparison.csv`.

## Files in this overlay

- `backend/student_test_evaluator.py` — isolated evaluator hook
- `frontend/index.html` — explicitly labels student-test Top-1 accuracy
- `tests/test_m4_student_test_evaluator.py` — evaluation contract tests

No M1/M2/M3 modules are modified. The M4 FastAPI backend already supports
`PATHSNAP_EVALUATOR` with signature `(trip_id, prediction) -> float`; no update
to `backend/main.py` is needed.

## Setup: PowerShell

```powershell
cd D:\PathSnap
$env:PATHSNAP_MODE = 'integrated'
$env:PATHSNAP_PARTITION = 'student_test'
$env:PATHSNAP_TRIP_ROOT = (Resolve-Path '.\dataset\student_test').Path
$env:PATHSNAP_MODEL_PATH = (Resolve-Path '.\artifacts\m3\random_forest_tuned_20261118T123057Z.joblib').Path
$env:PATHSNAP_MODEL_CONFIG = (Resolve-Path '.\artifacts\m3\random_forest_tuned_20261118T123057Z.json').Path
$env:PATHSNAP_PIPELINE_FACTORY = 'backend.real_inference:load_pipeline'
$env:PATHSNAP_HMM_DECODER = 'backend.m2_hmm_adapter:decode_trip_for_m4'
$env:PATHSNAP_EVALUATOR = 'backend.student_test_evaluator:evaluate_observation_top1'
$env:PATHSNAP_STUDENT_TEST_LABEL_DIR = (Resolve-Path '.\outputs\labelled\student_test').Path
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In the frontend terminal:

```powershell
cd D:\PathSnap
$env:PATHSNAP_API_URL = 'http://127.0.0.1:8000'
python -m uvicorn frontend.app:app --host 127.0.0.1 --port 8501
```

Open http://127.0.0.1:8501 and reconstruct a trip. For example, student-test
trip `00000052` should display **Student-test Top-1 accuracy** instead of N/A,
provided the live prediction agrees with the corresponding labels. The M2
batch evaluation previously reported 1.0 (100%) for this trip, but the live
hook recomputes the score for the **current** prediction rather than copying
that historic number.

## Warnings

- Never use this hook for `evaluator_test`. The hook explicitly refuses it.
- Keep original `.route` files outside the live inference root. This hook
  needs only the labeled candidate CSV generated offline by M2.
- Do not pass labels or evaluation scores into candidate generation, model
  selection, HMM emissions, or hyperparameter tuning.
- Accuracy here means eligible-observation arc Top-1 accuracy; it is not
  route-level LCS F1 or a proof of generalisation to the private evaluator set.
- Never commit original reference routes or large student-test data unless
  your team has expressly authorised distributing them.
