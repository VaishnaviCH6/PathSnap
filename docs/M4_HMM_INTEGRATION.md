# PathSnap — M4 HMM/Viterbi Integration

This **M4-only overlay** integrates the published M2 directed-road Viterbi code
with the real M3 Random Forest currently running in M4 `model_only` mode.
It does not overwrite `src/` teammate code or include a model or dataset.

## 1. Update your team branch

In VS Code PowerShell, inside the **actual Git repository** (not `D:\PathSnap-team`):

```powershell
cd D:\PathSnap
git switch feature/m4-fastapi-frontend
git status --short
```

If there are uncommitted changes, commit or stash them **before merging**. Keep your
working tree clean before `git merge`.

```powershell
git fetch origin
git merge origin/main
Test-Path .\src\map_matching\hmm.py
```

The last command must return `True`. If `False`, M2 has not yet pushed the code
shown in the uploaded snapshot to `origin/main`; request that code be pushed
rather than inventing a different HMM. If Git reports conflicts, resolve them
and rerun the tests before proceeding.

## 2. Install the M4 integration overlay

Download `PathSnap_M4_HMM_Integration_Overlay.zip`, save it in your Downloads
folder, then extract its **repo-root-relative** contents after reviewing any
existing modifications:

```powershell
$zip = Join-Path $env:USERPROFILE 'Downloads\PathSnap_M4_HMM_Integration_Overlay.zip'
Test-Path $zip
Expand-Archive -LiteralPath $zip -DestinationPath 'D:\PathSnap' -Force
git status --short
```

This changes **M4 only**: `backend/`, `frontend/`, `tests/`, and `docs/`.
It does not replace M2 source.

## 3. Run tests

```powershell
cd D:\PathSnap
python -m pytest -q tests/test_m4.py tests/test_m4_v2.py tests/test_m4_real_integration.py tests/test_m4_hmm_adapter.py tests/test_hmm.py
```

The new M4 decoder adapter:

- Receives M2 feature rows and M3 `emission_probability` per
  `(trip_id, observation_id, candidate_arc_id)` key.
- Renames the probability to M2's `positive_probability`.
- Calls `src.map_matching.hmm.decode_trip()` and
  `src.map_matching.hmm.reconstruct_trajectory()` with the correct graph tables.
- Preserves explicit `None` for observations without candidates and preserves
  segment breaks in `Prediction.route_break_indices`.
- Never reads `.route` reference data or labels.

## 4. Terminal 1: real M3 + M2 HMM

Stop the previous backend first (`Ctrl+C` in Terminal 1) to free port 8000.

```powershell
cd D:\PathSnap
$env:PATHSNAP_MODE = 'integrated'
$env:PATHSNAP_PARTITION = 'student_test'
$env:PATHSNAP_TRIP_ROOT = (Resolve-Path '.\dataset\student_test').Path
$env:PATHSNAP_MODEL_PATH = (Resolve-Path '.\artifacts\m3\random_forest_tuned_20261118T123057Z.joblib').Path
$env:PATHSNAP_MODEL_CONFIG = (Resolve-Path '.\artifacts\m3\random_forest_tuned_20261118T123057Z.json').Path
$env:PATHSNAP_PIPELINE_FACTORY = 'backend.real_inference:load_pipeline'
$env:PATHSNAP_HMM_DECODER = 'backend.m2_hmm_adapter:decode_trip_for_m4'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Expected startup: `PathSnap ready: mode=integrated partition=student_test trips=10`.

## 5. Terminal 2: frontend

Restart your frontend so it serves the updated map and route-break behaviour:

```powershell
cd D:\PathSnap
$env:PATHSNAP_API_URL = 'http://127.0.0.1:8000'
python -m uvicorn frontend.app:app --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501` and choose trip `00000052`.

Test the backend separately:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{trip_id='00000052'} | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://127.0.0.1:8000/reconstruct -Method POST -ContentType 'application/json' -Body $body
$r | Select-Object trip_id,mode,observation_count,matched_count,route_arc_count,route_segment_count,route_continuity,inference_ms,accuracy
```

Expected `mode=integrated`, `accuracy` remains null until a *separately
configured, authorised* evaluator computes a specified accuracy metric. The
reported `route_continuity` is an arc-adjacency metric (explicit breaks count as
non-continuous); it is not ground-truth accuracy.

## 6. What changed

- New `backend/m2_hmm_adapter.py`: invokes the real M2 decoder without touching
  teammate code.
- `backend/pipeline.py`: adds optional `route_break_indices` with validation.
- `backend/main.py`: includes route breaks/count in response and does not
  count an explicit break as a connected arc transition.
- `frontend/app.py`: honours route breaks when drawing Folium polylines.
- `frontend/index.html`: marks arc-sequence breaks using `| ROUTE BREAK |`.
- `tests/test_m4_hmm_adapter.py`: tests missing observations, breaks, and the
  integrated API and frontend map.

## Important evaluation boundaries

The student-test `.track`, `.nodes`, and `.arcs` files are legitimate inputs to
inference. Do not pass `.route` reference data, candidate labels, or any
private evaluator ground truth into feature generation, Random Forest scoring,
or Viterbi decoding. The upstream M2 evaluation script reads permitted student-
test references only *after* prediction/decoding to compute evaluation metrics.
The previous `model_only` route should not be presented as the decoded result.
