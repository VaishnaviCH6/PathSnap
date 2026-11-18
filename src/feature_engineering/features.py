"""Candidate-level spatial and temporal feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bearing_deg(x1, y1, x2, y2):
    """Clockwise bearing from north, in [0, 360)."""
    dx, dy = x2 - x1, y2 - y1
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _angle_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def add_observation_features(track: pd.DataFrame, transformer) -> pd.DataFrame:
    """Add per-observation movement features without dropping observation IDs.

    Timestamp values are interpreted as seconds, as specified by the dataset.
    Original row order defines observation_id; timestamps are not silently sorted.
    """
    required = {"observation_id", "longitude", "latitude", "timestamp"}
    if not required.issubset(track.columns):
        raise ValueError(f"track must contain columns: {sorted(required)}")
    result = track.copy()
    coords = []
    for lon, lat in result[["longitude", "latitude"]].itertuples(index=False, name=None):
        if (pd.notna(lon) and pd.notna(lat) and np.isfinite(lon) and np.isfinite(lat)
            and -180 <= lon <= 180 and -90 <= lat <= 90):
            coords.append(transformer.transform(float(lon), float(lat)))
        else:
            coords.append((np.nan, np.nan))
    result["gps_x_m"] = [p[0] for p in coords]
    result["gps_y_m"] = [p[1] for p in coords]

    n = len(result)
    step_distance = np.full(n, np.nan)
    time_difference = np.full(n, np.nan)
    gps_speed = np.full(n, np.nan)
    gps_heading = np.full(n, np.nan)
    heading_change = np.full(n, np.nan)

    ts = pd.to_numeric(result["timestamp"], errors="coerce").to_numpy(dtype=float)
    xs = result["gps_x_m"].to_numpy(dtype=float)
    ys = result["gps_y_m"].to_numpy(dtype=float)

    for i in range(1, n):
        dt = ts[i] - ts[i - 1] if np.isfinite(ts[i]) and np.isfinite(ts[i - 1]) else np.nan
        time_difference[i] = dt
        valid_xy = np.isfinite(xs[i-1:i+1]).all() and np.isfinite(ys[i-1:i+1]).all()
        if valid_xy:
            d = float(np.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]))
            step_distance[i] = d
            if np.isfinite(dt) and dt > 0:
                gps_speed[i] = d / dt
                gps_heading[i] = _bearing_deg(xs[i-1], ys[i-1], xs[i], ys[i])
                if i >= 2 and np.isfinite(gps_heading[i-1]):
                    heading_change[i] = _angle_difference(gps_heading[i], gps_heading[i-1])

    result["gps_step_distance_m"] = step_distance
    result["time_difference_s"] = time_difference
    result["gps_speed_mps"] = gps_speed
    result["gps_heading_deg"] = gps_heading
    result["gps_heading_change_deg"] = heading_change
    return result


def extract_candidate_features(track_features: pd.DataFrame,
                               candidates: pd.DataFrame,
                               arcs: pd.DataFrame,
                               nodes: pd.DataFrame,
                               transformer) -> pd.DataFrame:
    """Join candidate geometry with observation movement features.

    Road direction uses the stored arc direction (from_node_id -> to_node_id).
    Direction difference is undirected-road compatible: the smaller angle to
    either orientation, because many road-network arcs are represented as
    directed pairs but the GPS bearing alone does not establish travel legality.
    """
    if candidates.empty:
        return pd.DataFrame(columns=[
            "observation_id",
            "candidate_arc_id",
            "distance_to_segment_m",
            "direction_difference_deg",
            "gps_step_distance_m",
            "time_difference_s",
            "gps_speed_mps",
            "segment_length_m",
            "projection_fraction",
            "gps_heading_deg",
            "road_bearing_deg",
            "gps_heading_change_deg",
        ])
    node_xy = {}
    for node in nodes.itertuples(index=False):
        node_xy[int(node.node_id)] = transformer.transform(float(node.longitude),
                                                             float(node.latitude))
    arc_rows = arcs.set_index("arc_id")
    records = []
    obs = track_features.set_index("observation_id")

    for cand in candidates.itertuples(index=False):
        arc_id = int(cand.candidate_arc_id)
        arc = arc_rows.loc[arc_id]
        x1, y1 = node_xy[int(arc.from_node_id)]
        x2, y2 = node_xy[int(arc.to_node_id)]
        road_bearing = _bearing_deg(x1, y1, x2, y2)
        o = obs.loc[int(cand.observation_id)]
        gps_bearing = o["gps_heading_deg"]
        if pd.notna(gps_bearing):
            diff_forward = _angle_difference(float(gps_bearing), road_bearing)
            diff_reverse = _angle_difference(float(gps_bearing), (road_bearing + 180) % 360)
            direction_difference = min(diff_forward, diff_reverse)
        else:
            direction_difference = np.nan

        records.append({
            "observation_id": int(cand.observation_id),
            "candidate_arc_id": arc_id,
            "distance_to_segment_m": float(cand.distance_to_segment_m),
            "direction_difference_deg": direction_difference,
            "gps_step_distance_m": o["gps_step_distance_m"],
            "time_difference_s": o["time_difference_s"],
            "gps_speed_mps": o["gps_speed_mps"],
            "segment_length_m": float(cand.segment_length_m),
            "projection_fraction": float(cand.projection_fraction),
            "gps_heading_deg": gps_bearing,
            "road_bearing_deg": road_bearing,
            "gps_heading_change_deg": o["gps_heading_change_deg"],
        })

    return pd.DataFrame.from_records(records)
