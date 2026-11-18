# M1 – Exploratory Data Analysis, Folium Visualisation and Dataset Validation

**Project:** PathSnap
**Milestone:** M1
**Objective:** Dataset Exploration, Validation and Geographical Visualisation
**Status:** Core analysis completed; final anomaly investigation and handover pending.

---

## 1. Introduction

The objective of M1 is to explore, validate and prepare the PathSnap dataset for the downstream distributed map-matching pipeline.

The work includes dataset structure identification, data-quality analysis, exploratory data analysis, road-network connectivity analysis, reusable dataset loading and validation scripts, and geographical visualisation using Folium.

The analysis preserves the original dataset files and identifiers.

## 2. Dataset Structure

The dataset is located in the project's `dataset/` directory.

The following partitions were identified:

| Partition | Number of trips |
|---|---:|
| Training | 80 |
| Student testing | 10 |
| **Total** | **90** |

All 90 identified trip directories contained the four expected files.

**Note:** An evaluator testing partition was not present in the inspected dataset directory. Its availability must be confirmed separately.

### 2.1 File formats

| File | Columns / contents | Description |
|---|---|---|
| `.track` | Longitude, Latitude, Timestamp | Ordered GPS observations |
| `.nodes` | Longitude, Latitude | Road-network node coordinates |
| `.arcs` | Source node, Target node | Road-network connections represented by node indices |
| `.route` | Arc ID | Reference route represented by road-arc identifiers |

The files are associated through the trip identifier.

### 2.2 Sample observations

Sample trip: `00000001`

Example GPS observation:

- Longitude: 8.595203
- Latitude: 49.844053
- Timestamp: 1.000000

Example road-network node:

- Longitude: 8.581679
- Latitude: 49.857880

Example arc:

- Source node: 0
- Target node: 281

Example route entry:

- Arc ID: 2641

## 3. Dataset Statistics

The following statistics were obtained from the actual dataset files.

| Metric | Training | Student Test |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Road nodes | 1,933,485 | 90,689 |
| Road arcs | 3,876,904 | 185,041 |
| Reference route entries | 31,336 | 3,477 |

**Combined totals:**

- Trips: 90
- GPS observations: 227,577
- Road nodes (summed across trips): 2,024,174
- Road arcs (summed across trips): 4,061,945
- Reference route entries: 34,813

Node and arc totals are sums across individual trip networks, not counts of unique network-wide entities.

## 4. Data Quality Analysis

Data-quality checks were performed on the training and student testing partitions.

### 4.1 GPS observations

| Check | Training | Student Test |
|---|---:|---:|
| Missing GPS values | 0 | 0 |
| Duplicate observations | 0 | 0 |
| Invalid coordinate ranges | 0 | 0 |

No missing GPS values, duplicate GPS observations or invalid longitude/latitude ranges were detected.

### 4.2 Timestamp analysis

| Check | Training | Student Test |
|---|---:|---:|
| Timestamp gaps greater than 1 second | 7,286 | 264 |
| Timestamp reversals | 0 | 0 |

Timestamp intervals were examined using consecutive timestamp differences.

The minimum observed interval was 1 second.

Unusually large timestamp gaps were identified:

- Maximum training timestamp gap: 153,064 seconds.
- Maximum student-test timestamp gap: 210 seconds.

The largest training gap corresponds to approximately 42.5 hours.

These observations require further investigation. They have not been removed or modified.

### 4.3 Road-network reference validation

The following checks were performed:

- Arc source indices must reference existing road nodes.
- Arc target indices must reference existing road nodes.
- Route arc identifiers must reference existing road arcs.

| Check | Training | Student Test |
|---|---:|---:|
| Invalid arc-to-node references | 0 | 0 |
| Invalid route-to-arc references | 0 | 0 |

All checked road-network references were valid.

## 5. Road-Network Connectivity Analysis

Road-network connectivity was analysed using SciPy sparse graph representations.

Connected components were calculated by treating the road network as undirected for weak-connectivity analysis.

### 5.1 Results

| Metric | Training | Student Test |
|---|---:|---:|
| Isolated nodes | 0 | 0 |
| Connected components (summed across trips) | 3,121 | 227 |
| Average largest-component percentage | 96.90% | 96.39% |

### 5.2 Observations

- No isolated road nodes were detected.
- Multiple connected components were identified across individual trip networks.
- The largest connected component contains approximately 96–97% of the nodes on average.
- Smaller disconnected components exist and should be considered during downstream map-matching development.

These results represent weak connectivity, not directed reachability.

## 6. Exploratory Data Analysis

EDA was performed to understand GPS observation distributions and timestamp characteristics.

The following visualisations were generated:

| Visualisation | Output |
|---|---|
| GPS observations per trip | `gps_observations_per_trip.png` |
| GPS observation distribution | `gps_observation_distribution.png` |
| Timestamp gap distribution | `timestamp_gap_distribution.png` |

The following statistical outputs were also generated:

- `trip_statistics.csv`
- `timestamp_gaps.csv`
- `road_network_connectivity.csv`

All plots and statistical summaries were generated using actual dataset values.

### 6.1 Observations

1. The training partition contains substantially more GPS observations than the student testing partition.
2. GPS observation counts vary across individual trips.
3. Timestamp intervals are not uniformly distributed.
4. Large timestamp gaps occur more frequently in the training partition than in student testing.
5. No invalid coordinate ranges or duplicate GPS observations were detected.

## 7. Folium Geographical Visualisation

An interactive geographical visualisation was generated using Folium.

**Visualised trip:** `00000001`

The map includes:

- GPS trajectory represented by a blue polyline.
- Green starting-point marker.
- Red ending-point marker.
- Sampled road-network segments.
- Interactive zoom and pan controls.

The basemap uses Esri World Street Map tiles because the initially attempted OpenStreetMap and CARTO tile services returned access errors.

**Output:**

`results/m1/maps/trip_00000001_map.html`

The road-network rendering is sampled for browser performance. It does not display every road arc.

## 8. Reusable Dataset Processing Code

Two reusable Python modules were implemented.

### 8.1 Dataset loader

**File:** `src/data_loader.py`

Responsibilities:

- Load individual trips.
- Read `.track`, `.nodes`, `.arcs` and `.route` files.
- Preserve trip identifiers.
- Load complete supported partitions.
- Detect missing files.

### 8.2 Dataset validator

**File:** `src/data_validator.py`

Responsibilities:

- Check missing GPS values.
- Detect duplicate GPS observations.
- Validate longitude and latitude ranges.
- Identify timestamp reversals and repeated timestamps.
- Identify timestamp gaps.
- Validate arc-to-node references.
- Validate route-to-arc references.

The validator distinguishes structural validity from timestamp-gap warnings.

Original dataset files are preserved.

## 9. Data Preparation and Downstream Integration

The validated dataset structure and reusable loading functions provide the foundation for downstream feature engineering and map-matching development.

The following principles are maintained:

1. Original trip identifiers are preserved.
2. Original dataset files are not overwritten.
3. Training and student testing partitions remain distinguishable.
4. Timestamp irregularities are recorded rather than silently removed.
5. Evaluator labels must not be used for training, feature selection or model selection.

The loader currently supports the two partitions identified in the inspected dataset.

## 10. Limitations and Pending Investigation

The following points require attention before final milestone closure:

1. **Timestamp anomalies:** Large timestamp gaps, particularly in training, require investigation at the affected trip level.
2. **Evaluator partition:** No evaluator partition was identified in the inspected dataset directory. Its location and availability need confirmation.
3. **Connectivity:** The current connectivity analysis treats road arcs as undirected. Directed reachability has not been evaluated.
4. **Map visualisation:** Road arcs are sampled to keep the interactive map manageable.
5. **Geographical visualisation:** One representative training trip has been mapped. Additional maps can be generated if required.

## 11. Output Directory Structure

The current M1 outputs are organised as follows:

```text
PathSnap/
│
├── dataset/
│   ├── train/
│   └── student_test/
│
├── src/
│   ├── data_loader.py
│   └── data_validator.py
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── results/
│   └── m1/
│       ├── plots/
│       │   ├── gps_observations_per_trip.png
│       │   ├── gps_observation_distribution.png
│       │   └── timestamp_gap_distribution.png
│       │
│       ├── maps/
│       │   └── trip_00000001_map.html
│       │
│       ├── trip_statistics.csv
│       ├── timestamp_gaps.csv
│       └── road_network_connectivity.csv
│
└── docs/
    └── M1_EDA_Technical_Report.md
```

## 12. Conclusion

M1 established the initial dataset understanding and validation foundation for the PathSnap distributed map-matching pipeline.

A total of 90 trips containing 227,577 GPS observations were explored across the training and student testing partitions.

GPS data-quality checks and road-network reference validation did not identify missing observations, duplicate GPS records, invalid coordinate ranges or invalid road references.

Road-network connectivity analysis identified no isolated nodes, while multiple connected components were observed across trip networks.

EDA plots, statistical summaries and an interactive Folium visualisation were generated. Reusable dataset loading and validation modules were also implemented.

The resulting outputs provide a structured starting point for M2 and subsequent feature engineering and map-matching development.

**M1 outcome:** Core dataset exploration, validation, EDA and geographical visualisation implemented. Timestamp anomaly investigation, evaluator partition confirmation and final integration handover remain outstanding.
