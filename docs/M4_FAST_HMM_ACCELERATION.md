# PathSnap M4 — Fast HMM + bounded prediction cache

This overlay speeds up the **real** M2+M3+M4 integrated route reconstruction, without changing the trained Random Forest or the team's M2 source modules. This is not a drawing-only workaround or a timeout extension.

## Files installed

- `backend/fast_hmm.py`: SciPy Dijkstra with a bounded NumPy-array cache (128 source nodes), target-directed shortest **arc** path recovery, and a single reused projected road graph for both decoding and reconstruction. The Viterbi recurrence and break handling follow the original M2 implementation.
- `backend/m2_hmm_adapter.py`: selects the M4 fast decoder by default. `PATHSNAP_HMM_FAST=0` reverts to the original unchanged M2 decoder.
- `backend/real_inference.py`: an LRU **prediction-only** cache keyed by trip ID and a fingerprint of track/nodes/arcs; defaults to two trips. A lock prevents memory-heavy parallel decodes. This never caches or reads ground truth or labels.
- `tests/test_m4_fast_hmm.py`: shortest-path parity, decoder parity, and prediction cache tests.
- `scripts/benchmark_m4_fast_hmm.py`: small-trip parity and real-model benchmark tool.

## Install on Windows (repository root `D:\PathSnap`)

1. Stop your backend server with Ctrl+C. Stop frontend only if you need to restart its HTTP proxy.
2. Before extracting, back up the two existing M4 backend files:

```powershell
cd D:\PathSnap
Copy-Item .\backend\real_inference.py .\backend\real_inference.py.bak
Copy-Item .\backend\m2_hmm_adapter.py .\backend\m2_hmm_adapter.py.bak
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\PathSnap_M4_Fast_Backend_Overlay.zip" -DestinationPath "D:\PathSnap" -Force
```

3. Run:

```powershell
python -m pytest -q tests/test_m4_fast_hmm.py tests/test_m4_hmm_adapter.py tests/test_m4_real_integration.py
python scripts/benchmark_m4_fast_hmm.py --trip 00000052 --compare
```

4. Start the original backend as before, with these additional environment variables:

```powershell
$env:PATHSNAP_HMM_FAST = "1"
$env:PATHSNAP_PREDICTION_CACHE_SIZE = "2"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The other variables `PATHSNAP_MODE`, `PATHSNAP_PARTITION`, `PATHSNAP_TRIP_ROOT`, `PATHSNAP_MODEL_PATH`, `PATHSNAP_MODEL_CONFIG`, `PATHSNAP_PIPELINE_FACTORY`, `PATHSNAP_HMM_DECODER`, and optional `PATHSNAP_EVALUATOR` must remain configured as previously. A full example:

```powershell
$env:PATHSNAP_MODE = "integrated"
$env:PATHSNAP_PARTITION = "student_test"
$env:PATHSNAP_TRIP_ROOT = (Resolve-Path ".\dataset\student_test").Path
$env:PATHSNAP_MODEL_PATH = (Resolve-Path ".\artifacts\m3\random_forest_tuned_20261118T123057Z.joblib").Path
$env:PATHSNAP_MODEL_CONFIG = (Resolve-Path ".\artifacts\m3\random_forest_tuned_20261118T123057Z.json").Path
$env:PATHSNAP_PIPELINE_FACTORY = "backend.real_inference:load_pipeline"
$env:PATHSNAP_HMM_DECODER = "backend.m2_hmm_adapter:decode_trip_for_m4"
$env:PATHSNAP_HMM_FAST = "1"
$env:PATHSNAP_PREDICTION_CACHE_SIZE = "2"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Leave the frontend (`uvicorn frontend.app:app --port 8501`) unchanged. Refresh the browser.

## Local test evidence (19 November 2026)

Tested against the supplied real student-test track, nodes, arcs, and saved Random Forest, *without using ground truth*. Different machine/library versions will vary:

| Trip | Observations | Fast HMM decode + reconstruction | End-to-end frontend HTTP on first request | Same trip on second request |
|---|---:|---:|---:|---:|
| 00000052 | 262 | ~0.42 seconds | ~0.71 seconds | not measured |
| 00000040 | 2,995 | ~3.64 seconds | not measured | not measured |
| 00000056 | 3,958 | ~11.37 seconds | ~14.27 seconds | ~0.29 seconds |

For trips `00000052` and `00000007`, old and optimized HMM selected **identical observation arcs, reconstructed route arcs, and break positions**. Parity tests also verified directed shortest paths on small randomly generated road graphs. We have **not** exhaustively proven parity for all ten trips or the hidden evaluator partition. Do not claim that these speed tests show improved route accuracy.

The testing environment had scikit-learn 1.8.0, but the saved model was created with 1.9.1; scikit-learn emitted deserialization warnings. For actual deployment, use the model's matching scikit-learn 1.9.1 version as required by the project.

## If a problem occurs

- To switch off optimized HMM without restoring files, set `$env:PATHSNAP_HMM_FAST = "0"`, then restart backend.
- To disable cached predictions for a fair uncached performance measurement, set `$env:PATHSNAP_PREDICTION_CACHE_SIZE = "0"`, then restart backend.
- For rollback to original M4 files, restore the two `.bak` files and restart backend. The extra `backend/fast_hmm.py` is unused after rollback.
- Cache is **in-memory only** and is lost on backend restart; it retains predictions for at most two different input fingerprint keys by default.

## Important data separation

This code reads current-trip GPS, road nodes/arcs, features and model probabilities only. It does **not** read `.route` files or student-test labels. The external student-test evaluator remains separate. Do not use evaluation data to tune decoder parameters or tune model predictions.
