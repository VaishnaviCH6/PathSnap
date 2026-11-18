"""Reference-route alignment and candidate-level binary labels for PathSnap.

The .route file is used only by this offline labeling module. It must never be
called by inference or included among model features.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

CANDIDATE_COLUMNS = [
    "trip_id", "observation_id", "candidate_arc_id",
    "distance_to_segment_m", "direction_difference_deg",
    "gps_step_distance_m", "time_difference_s", "gps_speed_mps",
    "segment_length_m", "projection_fraction", "gps_heading_deg",
    "road_bearing_deg", "gps_heading_change_deg", "label",
]
STATUS_COLUMNS = [
    "trip_id", "observation_id", "candidate_count", "candidate_status",
    "alignment_status", "supervised_eligible",
]


def load_reference_route(route_path: str | Path) -> list[int]:
    """Read .route as a chronological, one-integer-per-line arc-ID sequence."""
    values = pd.read_csv(route_path, sep=r"\s+", header=None).iloc[:, 0]
    numeric = pd.to_numeric(values, errors="raise").to_numpy()
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(".route must contain finite integer arc IDs")
    return numeric.astype(np.int64).tolist()


def _route_occurrences(route_arc_ids, network, max_alignment_distance_m):
    """Build spatially indexed route occurrences, preserving repeated arc visits."""
    arc_to_geom = {int(a): g for a, g in zip(network._arc_lookup, network.geometries)}
    missing = sorted(set(map(int, route_arc_ids)) - set(arc_to_geom))
    if missing:
        raise ValueError(f"Reference route contains invalid arc IDs: {missing[:10]}")
    geoms, arc_ids, starts, lengths = [], [], [], []
    cumulative = 0.0
    for arc_id in route_arc_ids:
        geom = arc_to_geom[int(arc_id)]
        geoms.append(geom)
        arc_ids.append(int(arc_id))
        starts.append(cumulative)
        lengths.append(float(geom.length))
        cumulative += float(geom.length)
    if not geoms:
        raise ValueError("Reference route is empty")
    tree = STRtree(geoms)
    return geoms, np.asarray(arc_ids), np.asarray(starts), np.asarray(lengths), tree


def align_observations_to_route(track_features: pd.DataFrame, route_arc_ids,
                                network, max_alignment_distance_m: float = 100.0,
                                ambiguity_margin_m: float = 3.0) -> pd.DataFrame:
    """Align each GPS observation to one route occurrence using monotonic DP.

    Local states are route occurrences within max_alignment_distance_m. The DP
    favours close geometry and route progress consistent with GPS step distance.
    An observation is ambiguous when a different reference arc is within
    ambiguity_margin_m of its selected geometric distance.
    """
    if max_alignment_distance_m <= 0 or ambiguity_margin_m < 0:
        raise ValueError("Alignment distance must be positive and ambiguity margin non-negative")
    geoms, arc_ids, starts, lengths, tree = _route_occurrences(
        route_arc_ids, network, max_alignment_distance_m
    )
    states_by_obs = []
    for row in track_features.itertuples(index=False):
        lon, lat = float(row.longitude), float(row.latitude)
        if (not np.isfinite(lon) or not np.isfinite(lat) or
                lon < -180 or lon > 180 or lat < -90 or lat > 90):
            states_by_obs.append([])
            continue
        x, y = network.transformer.transform(lon, lat)
        point = Point(x, y)
        idxs = np.asarray(tree.query(point.buffer(max_alignment_distance_m)), dtype=int)
        states = []
        for idx in idxs:
            geom = geoms[int(idx)]
            d = float(point.distance(geom))
            if d <= max_alignment_distance_m:
                fraction = float(geom.project(point) / lengths[int(idx)])
                progress = float(starts[int(idx)] + fraction * lengths[int(idx)])
                states.append({"occ": int(idx), "arc": int(arc_ids[int(idx)]),
                               "distance": d, "progress": progress})
        states.sort(key=lambda s: (s["occ"], s["distance"], s["arc"]))
        states_by_obs.append(states)

    # Dynamic programming over contiguous runs that have at least one state.
    selected = [None] * len(states_by_obs)
    i = 0
    while i < len(states_by_obs):
        if not states_by_obs[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(states_by_obs) and states_by_obs[j + 1]:
            j += 1
        costs = np.asarray([s["distance"] for s in states_by_obs[i]], dtype=float)
        backpointers = [np.full(len(costs), -1, dtype=int)]
        for obs_idx in range(i + 1, j + 1):
            prev_states = states_by_obs[obs_idx - 1]
            curr_states = states_by_obs[obs_idx]
            row = track_features.iloc[obs_idx]
            gps_step = pd.to_numeric(pd.Series([row.get("gps_step_distance_m", np.nan)]), errors="coerce").iloc[0]
            new_costs = np.full(len(curr_states), np.inf)
            back = np.full(len(curr_states), -1, dtype=int)
            for ci, curr in enumerate(curr_states):
                for pi, prev in enumerate(prev_states):
                    advance = curr["progress"] - prev["progress"]
                    if advance < -5.0:
                        continue
                    transition = 0.0
                    if np.isfinite(gps_step) and gps_step >= 0:
                        transition = 0.15 * abs(max(0.0, advance) - float(gps_step))
                    value = costs[pi] + curr["distance"] + transition
                    if value < new_costs[ci]:
                        new_costs[ci], back[ci] = value, pi
            if not np.isfinite(new_costs).any():
                # Restart alignment after a discontinuity; do not fabricate a transition.
                costs = np.asarray([s["distance"] for s in curr_states], dtype=float)
                back = np.full(len(curr_states), -1, dtype=int)
            else:
                costs = new_costs
            backpointers.append(back)
        state_idx = int(np.argmin(costs))
        for offset in range(j - i, -1, -1):
            obs_idx = i + offset
            selected[obs_idx] = states_by_obs[obs_idx][state_idx]
            if offset > 0:
                prev_idx = int(backpointers[offset][state_idx])
                if prev_idx < 0:
                    prev_idx = int(np.argmin([s["distance"] for s in states_by_obs[obs_idx - 1]]))
                state_idx = prev_idx
        i = j + 1

    records = []
    for obs_idx, (states, chosen) in enumerate(zip(states_by_obs, selected)):
        if not states:
            records.append({"observation_id": obs_idx, "reference_arc_id": pd.NA,
                            "alignment_status": "no_route_alignment"})
            continue
        # Distinct alternative reference arcs close to the selected geometry are ambiguous.
        alternatives = [s for s in states if s["arc"] != chosen["arc"] and
                        s["distance"] <= chosen["distance"] + ambiguity_margin_m]
        records.append({
            "observation_id": obs_idx,
            "reference_arc_id": chosen["arc"],
            "alignment_status": "ambiguous" if alternatives else "aligned",
        })
    return pd.DataFrame(records)


def label_candidate_table(trip_id: str, candidates: pd.DataFrame,
                          observation_status: pd.DataFrame,
                          alignment: pd.DataFrame):
    """Return exact M3 candidate schema and observation-level status contract.

    Only observations with one unambiguous aligned reference arc present in the
    candidate set are supervised-eligible. Other candidate labels remain NaN.
    """
    merged = observation_status.merge(alignment, on="observation_id", how="left", validate="one_to_one")
    merged["alignment_status"] = merged["alignment_status"].fillna("no_route_alignment")
    candidate_groups = {int(k): g for k, g in candidates.groupby("observation_id")} if not candidates.empty else {}
    labels = candidates.copy()
    if labels.empty:
        labels = pd.DataFrame(columns=[c for c in CANDIDATE_COLUMNS if c not in {"trip_id", "label"}])
    labels["label"] = np.nan
    eligible = []
    for row in merged.itertuples(index=False):
        obs_id = int(row.observation_id)
        group = candidate_groups.get(obs_id)
        if row.candidate_count == 0:
            candidate_state = "no_generated_candidates"
            is_eligible = False
        elif row.alignment_status == "ambiguous":
            candidate_state = "candidates_present"
            is_eligible = False
        elif row.alignment_status != "aligned" or pd.isna(row.reference_arc_id):
            candidate_state = "candidates_present"
            is_eligible = False
        elif group is None or int(row.reference_arc_id) not in set(group.candidate_arc_id.astype(int)):
            candidate_state = "candidate_miss"
            is_eligible = False
        else:
            candidate_state = "candidates_present"
            is_eligible = True
            mask = labels.observation_id.astype(int) == obs_id
            labels.loc[mask, "label"] = (
                labels.loc[mask, "candidate_arc_id"].astype(int) == int(row.reference_arc_id)
            ).astype(int)
        eligible.append((candidate_state, is_eligible))
    merged["candidate_status"] = [x[0] for x in eligible]
    merged["supervised_eligible"] = [x[1] for x in eligible]
    merged.insert(0, "trip_id", str(trip_id))
    if not labels.empty:
        labels.insert(0, "trip_id", str(trip_id))
    else:
        labels.insert(0, "trip_id", pd.Series(dtype="object"))
        labels["label"] = pd.Series(dtype="float64")
    labels = labels.reindex(columns=CANDIDATE_COLUMNS)
    status = merged.reindex(columns=STATUS_COLUMNS)
    return labels, status
