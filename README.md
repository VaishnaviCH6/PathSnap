# PathSnap — GPS Map Matching & Route Reconstruction

**UE24CS352A | Machine Learning Hackathon | PES University**

PathSnap is a distributed machine learning system designed to reconstruct realistic vehicle trajectories from noisy GPS observations by matching them to an underlying road network.

The project integrates **GPS preprocessing, road-segment candidate generation, machine learning classification, Hidden Markov Models (HMM), Viterbi decoding, and interactive map visualization**.

A FastAPI backend performs route reconstruction, while a separate frontend provides an interactive interface using Folium and OpenStreetMap.

**GitHub Repository:** https://github.com/VaishnaviCH6/PathSnap

## 1. Project Objectives

- Analyze and preprocess GPS trajectory and road-network datasets.
- Identify possible road segments for each GPS observation.
- Extract spatial, temporal, and road-network features.
- Train and optimize machine learning models for candidate road classification.
- Use HMM and Viterbi decoding to reconstruct a continuous road-network trajectory.
- Develop a FastAPI backend for route reconstruction.
- Display reconstructed routes using interactive maps.
- Evaluate model accuracy, route continuity, and reconstruction quality.
- Optimize backend execution time for large GPS trajectories.

## 2. Project Modules

### M1 — Dataset Exploration and Validation

Responsible for understanding and preparing the GPS and road-network datasets.

**Key responsibilities:**
- Dataset exploration and statistical analysis.
- GPS trajectory validation.
- Road-network data inspection.
- Handling invalid and missing observations.
- Dataset partition management.

### M2 — Candidate Generation and Route Reconstruction

Responsible for generating possible road matches and reconstructing trajectories.

**Key responsibilities:**
- Candidate road-segment generation.
- Spatial and temporal feature engineering.
- Reference-route alignment for permitted evaluation.
- Hidden Markov Model implementation.
- Viterbi-based route reconstruction.
- Road-network connectivity validation.

### M3 — Machine Learning Model Training and Optimization

Responsible for identifying the best model for road-segment candidate classification.

**Key responsibilities:**
- Training and comparing machine learning classifiers.
- Model selection and cross-validation.
- Bayesian hyperparameter optimization using Optuna.
- Random Forest model optimization.
- Model serialization and metadata management.
- Candidate probability prediction.

The selected Random Forest model provides candidate probabilities to the HMM decoder.

### M4 — Backend, Frontend, and System Integration

Responsible for integrating the complete pipeline into a working application.

**Key responsibilities:**
- FastAPI backend development.
- Random Forest and HMM integration.
- Trip selection and route reconstruction APIs.
- Interactive frontend development.
- Folium and OpenStreetMap visualization.
- Backend performance optimization.
- Route caching.
- Student-test evaluation integration.
- Application testing and deployment.

## 3. System Architecture

```text
                    GPS Trajectory Data
                            |
                            v
                  Dataset Preprocessing
                            |
                            v
                 Candidate Road Generation
                            |
                            v
                Spatial & Temporal Features
                            |
                            v
                  M3 Random Forest Model
                            |
                            v
                 Candidate Probabilities
                            |
                            v
                   HMM + Viterbi Decoder
                            |
                            v
                 Reconstructed Road Route
                            |
                            v
                     FastAPI Backend
                            |
                            v
                    Frontend Application
                            |
                            v
                Interactive OpenStreetMap
```

Student-test reference labels are used separately for evaluation after prediction. They are not provided to the Random Forest or HMM during inference.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Hyperparameter Optimization | Optuna |
| Route Reconstruction | Hidden Markov Model, Viterbi |
| Graph Algorithms | Dijkstra, SciPy |
| Backend Framework | FastAPI |
| Application Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Map Visualization | Folium, OpenStreetMap |
| Data Processing | NumPy, Pandas |
| Testing | Pytest |
| Version Control | Git, GitHub |

## 5. Project Directory Structure

```text
PathSnap/
|
|-- backend/
|   |-- main.py
|   |-- pipeline.py
|   |-- real_inference.py
|   |-- fast_hmm.py
|   |-- m2_hmm_adapter.py
|   |-- student_test_evaluator.py
|
|-- frontend/
|   |-- app.py
|   |-- index.html
|
|-- src/
|   |-- candidate_generation/
|   |-- feature_engineering/
|   |-- labeling/
|   |-- map_matching/
|   |-- models/
|   |-- optimization/
|
|-- dataset/
|-- artifacts/
|-- outputs/
|-- experiments/
|-- scripts/
|-- tests/
|-- docs/
|
|-- requirements-m4.txt
|-- README.md
```

The `dataset/`, `outputs/`, and local model artifact directories contain files that should be handled according to the team's data-sharing and version-control policy.

## 6. Installation and Setup

### Step 1 — Clone the repository

```powershell
git clone https://github.com/VaishnaviCH6/PathSnap.git
cd PathSnap
```

### Step 2 — Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install -r requirements-m4.txt
python -m pip install optuna
```

### Step 4 — Configure the trained model

The trained M3 Random Forest model and its JSON metadata must be placed under:

```text
artifacts/m3/
```

Required local files:

```text
random_forest_tuned_20261118T123057Z.joblib
random_forest_tuned_20261118T123057Z.json
```

Only load trusted model artifacts, using a compatible scikit-learn version.

### Step 5 — Prepare the dataset

Student-test trip folders should contain the GPS track and road-network input files:

```text
dataset/student_test/
    00000052/
        00000052.track
        00000052.nodes
        00000052.arcs
```

Reference `.route` files are reserved for authorised evaluation and are not required as inference inputs.

## 7. Running the Application

### Terminal 1 — Start the Backend

```powershell
cd D:\PathSnap
.\venv\Scripts\Activate.ps1

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

Backend URL:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### Terminal 2 — Start the Frontend

```powershell
cd D:\PathSnap
.\venv\Scripts\Activate.ps1

$env:PATHSNAP_API_URL = "http://127.0.0.1:8000"

python -m uvicorn frontend.app:app --host 127.0.0.1 --port 8501
```

Frontend URL:

http://127.0.0.1:8501

## 8. Application Features

The PathSnap frontend supports:

- Selection of GPS trajectory trips.
- Reconstruction of road-network routes.
- Visualization of observed GPS points.
- Visualization of predicted road routes.
- Display of matched GPS observations.
- Display of reconstructed road-arc IDs.
- Route-continuity reporting.
- Backend inference-time measurement.
- Optional student-test accuracy reporting.
- Interactive zooming and map navigation.
- Map-layer selection.
- Loading indicators and error handling.

**Map colors:**

- Blue — Observed GPS trajectory.
- Green — Reconstructed road-network route.

A fully connected route is not necessarily a correct route. Route correctness is assessed separately using reference-based evaluation metrics.

## 9. Backend Performance Optimization

The optimized M4 backend introduces improvements to HMM route reconstruction.

### Faster Shortest-Path Search

SciPy-based shortest-path operations reduce the cost of graph traversal on suitable road networks.

### Efficient Memory Management

Bounded graph-distance storage reduces memory overhead compared with large, unrestricted shortest-path caches.

### Graph Reuse

The reconstructed road graph is reused during decoding and route generation where supported.

### Prediction Caching

Previously reconstructed trips can be served from memory without repeating the complete inference pipeline.

**Configuration:**

```powershell
$env:PATHSNAP_HMM_FAST = "1"
$env:PATHSNAP_PREDICTION_CACHE_SIZE = "2"
```

Set `PATHSNAP_HMM_FAST=0` to return to the original M2 decoder.

Set `PATHSNAP_PREDICTION_CACHE_SIZE=0` to disable caching during performance benchmarking.

### Measured Backend Performance

| Trip ID | GPS Observations | Processing Time |
|---|---:|---:|
| 00000052 | 262 | 3.965 seconds |
| 00000040 | 2,995 | 8.218 seconds |
| 00000056 | 3,958 | 25.273 seconds |

These are measured local benchmark results. Execution times vary depending on the system, graph size, and caching.

For trip `00000052`, the optimized decoder produced identical observation assignments, route arcs, and break indices to the original decoder. The original decoder was faster on this small trip, so the optimized implementation should not be considered faster for every input.

## 10. Machine Learning Evaluation Results

The system was evaluated on ten permitted student-testing trips.

| Evaluation Metric | Random Forest Only | Random Forest + HMM |
|---|---:|---:|
| Observation Top-1 Accuracy | 94.51% | **98.37%** |
| Route Continuity | 99.81% | **99.99%** |
| Ordered-Route LCS F1 | 27.82% | **70.59%** |
| Normalized Edit Similarity | 17.18% | **56.29%** |

The reported values are means across ten student-test trips.

### Evaluation Highlights

- The Random Forest predicts probabilities for candidate road segments.
- HMM/Viterbi incorporates road-network transition constraints.
- Observation-level accuracy improved after HMM decoding.
- Ordered-route sequence quality improved substantially.
- Route continuity approached 100%.

**Note:** Observation Top-1 accuracy is not equivalent to complete-route accuracy. These results were obtained from the evaluated decoder configuration and must be revalidated when prediction behavior changes.

## 11. Student-Test Accuracy Integration

The frontend can optionally display accuracy for permitted student-test trips.

Configure the backend with:

```powershell
$env:PATHSNAP_EVALUATOR = "backend.student_test_evaluator:evaluate_observation_top1"

$env:PATHSNAP_STUDENT_TEST_LABEL_DIR = (Resolve-Path ".\outputs\labelled\student_test").Path
```

Restart the backend after configuring the variables.

The evaluator compares the predicted road-segment assignments against the authorised student-test labels.

When no evaluator is configured, the frontend displays:

```text
Student-test Top-1 accuracy: N/A
```

Reference labels must never influence model inference or HMM decoding. Do not use student-test evaluation labels for private evaluator-test predictions.

## 12. Frontend Map Display Synchronization Fix

The frontend includes a map-display synchronization improvement.

### Original Issue

When the user selected another trip, the dropdown could display the new trip while the map continued showing the previous route during reconstruction.

### Implemented Fix

- Clears the old map when another trip is selected.
- Disables trip selection during an active reconstruction.
- Displays a loading placeholder while processing.
- Prevents stale route visualization after a failed request.
- Synchronizes the displayed route with the selected trip.

This fix affects only the frontend interface.

It does not alter machine learning predictions, road-network geometry, HMM decoding, or evaluation results.

Sharp detours in a completed route must still be investigated using GPS observations and reference-route metrics.

## 13. Testing and Validation

The project uses Pytest for unit and integration testing.

Run the complete test suite:

```powershell
python -m pytest -q
```

**Latest reported local test results:**

```text
81 passed, 1 skipped, 1 warning
```

Run selected M4 integration tests:

```powershell
python -m pytest -q `
  tests/test_m4_fast_hmm.py `
  tests/test_m4_hmm_adapter.py `
  tests/test_m4_real_integration.py
```

The automated tests cover the available functionality, including integration behavior and the optimized HMM implementation.

The remaining deprecation warning concerns FastAPI/Starlette test-client compatibility and does not represent a failing test.

## 14. Two-Laptop Deployment

PathSnap supports a distributed deployment model.

**Laptop 1 — Backend**

Runs the trained Random Forest model, HMM decoder, graph processing, and FastAPI API.

**Laptop 2 — Frontend**

Runs the frontend service and communicates with the backend over HTTP.

Backend startup on a trusted private network:

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Frontend configuration:

```powershell
$env:PATHSNAP_API_URL = "http://BACKEND_LAN_IP:8000"
```

Replace `BACKEND_LAN_IP` with the actual private-network IP address of the backend laptop.

Restrict access to authorised devices and configure the firewall appropriately.

## 15. Security and Dataset Guidelines

- Never commit sensitive credentials or API keys.
- Do not include virtual environments in Git.
- Keep datasets and generated labels outside Git unless explicitly authorised.
- Avoid committing large trained-model binaries without an agreed distribution method.
- Do not expose ground-truth reference routes to inference functions.
- Keep evaluation data separate from model prediction.
- Validate changes using the available tests and permitted datasets.

## 16. GitHub Development Workflow

M4 development branch:

```text
feature/m4-fastapi-frontend
```

Recommended workflow:

```powershell
git status
git add README.md backend/ frontend/ tests/ scripts/ requirements-m4.txt

git commit -m "Complete M4 FastAPI frontend and optimized HMM integration"

git push origin feature/m4-fastapi-frontend
```

Review the staged files before committing to avoid including temporary files or unrelated project artifacts.

Merge the M4 branch into `main` using a pull request after team review.

---

## Conclusion

PathSnap demonstrates how machine learning and probabilistic sequence modeling can be combined for GPS map matching and road-network route reconstruction.

By integrating candidate generation, Random Forest classification, HMM/Viterbi decoding, optimized backend execution, and interactive map visualization, the project provides a complete system for reconstructing vehicle trajectories from noisy GPS observations.

The student-test evaluation demonstrates improved observation-level accuracy and ordered-route reconstruction quality when HMM/Viterbi decoding is combined with the trained classifier.

**PathSnap — Turning Noisy GPS Observations into Road-Network Trajectories.**
