# PathSnap student-test route evaluation

Run from the repository root after confirming the tuned M3 model artifact and the
10 student-test candidate CSVs are present:

```powershell
python scripts/evaluate_student_test_routes.py
```

Optional path overrides:

```powershell
python scripts/evaluate_student_test_routes.py `
  --model artifacts/random_forest_tuned_20261118T123057Z.joblib `
  --student-test-dir dataset/student_test `
  --candidate-dir outputs/labelled/student_test `
  --output-dir outputs/m2_student_test_evaluation
```

The script:
- Predicts candidate scores using the M3 model.
- Compares top-score classifier-only predictions with HMM/Viterbi predictions.
- Reconstructs ordered road-arc sequences while retaining segment breaks.
- Uses candidate labels and `.route` files only after prediction/decoding for evaluation.
- Reports observation Top-1 accuracy, candidate miss rate, route continuity,
  LCS-based ordered sequence precision/recall/F1, normalized edit similarity,
  and inference time.
- Writes per-trip observation predictions, reconstructed segments, a per-trip
  comparison CSV, and an overall comparison CSV.

The `.route` data is evaluation-only. It is never passed to model inference,
Viterbi decoding, or route reconstruction. Do not use evaluator-test data.
The reference route is supplied dataset information and may have limitations;
interpret route metrics accordingly.
