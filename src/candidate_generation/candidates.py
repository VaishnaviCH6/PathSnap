"""Generate road-segment candidates for GPS observations.

Arc IDs are the zero-based row positions in the original .arcs file.
Node IDs are the zero-based row positions in the original .nodes file.
Coordinates in source files are longitude, latitude (EPSG:4326).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


@dataclass
class RoadNetwork:
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    geometries: list
    transformer: Transformer
    crs: CRS
    tree: STRtree


def _read_whitespace(path: str, names: list[str], dtypes=None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtypes)


def load_road_network(nodes_path: str, arcs_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load nodes/arcs while preserving source row IDs."""
    nodes = _read_whitespace(nodes_path, ["longitude", "latitude"])
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    arcs = _read_whitespace(
        arcs_path, ["from_node_id", "to_node_id"],
        {"from_node_id": "int64", "to_node_id": "int64"},
    )
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))

    coords = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Road-network node coordinates contain non-finite values.")
    if ((coords[:, 0] < -180) | (coords[:, 0] > 180) |
        (coords[:, 1] < -90) | (coords[:, 1] > 90)).any():
        raise ValueError("Road-network coordinates are outside longitude/latitude bounds.")
    if len(arcs) and (
        (arcs[["from_node_id", "to_node_id"]].to_numpy() < 0).any() or
        (arcs[["from_node_id", "to_node_id"]].to_numpy() >= len(nodes)).any()
    ):
        raise ValueError("An arc references a node ID that does not exist.")
    return nodes, arcs


def _local_metric_transformer(longitudes: Iterable[float], latitudes: Iterable[float]):
    """Choose a local UTM CRS for metre-based calculations."""
    lon = np.asarray(list(longitudes), dtype=float)
    lat = np.asarray(list(latitudes), dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise ValueError("Cannot choose a projected CRS without valid coordinates.")
    lon0, lat0 = float(np.mean(lon[valid])), float(np.mean(lat[valid]))
    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return crs, transformer


def build_road_network(nodes: pd.DataFrame, arcs: pd.DataFrame,
                       transformer: Transformer | None = None) -> RoadNetwork:
    """Build projected LineStrings and a Shapely spatial index."""
    if transformer is None:
        crs, transformer = _local_metric_transformer(
            nodes["longitude"], nodes["latitude"]
        )
    else:
        crs = CRS.from_user_input(transformer.target_crs)

    xy = [transformer.transform(lon, lat)
          for lon, lat in nodes[["longitude", "latitude"]].itertuples(index=False, name=None)]
    geometries = []
    valid_arc_ids = []
    for row in arcs.itertuples(index=False):
        start = xy[int(row.from_node_id)]
        end = xy[int(row.to_node_id)]
        line = LineString([start, end])
        if line.length > 0 and np.isfinite(line.length):
            geometries.append(line)
            valid_arc_ids.append(int(row.arc_id))
    if not geometries:
        raise ValueError("Road network has no non-zero-length valid segments.")

    # Keep a positional mapping because STRtree returns geometry positions.
    tree = STRtree(geometries)
    arc_lookup = np.asarray(valid_arc_ids, dtype=np.int64)
    network = RoadNetwork(nodes, arcs, geometries, transformer, crs, tree)
    network._arc_lookup = arc_lookup
    return network


def generate_candidates(track: pd.DataFrame, network: RoadNetwork,
                        search_radius_m: float = 50.0,
                        max_candidates: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (candidate rows, per-observation status).

    track requires observation_id, longitude, latitude. A candidate is retained
    when its projected segment is within search_radius_m of the GPS point.
    Candidate rows are sorted by distance for each observation.
    """
    if search_radius_m <= 0 or max_candidates < 1:
        raise ValueError("search_radius_m must be > 0 and max_candidates >= 1.")
    required = {"observation_id", "longitude", "latitude"}
    if not required.issubset(track.columns):
        raise ValueError(f"track must contain columns: {sorted(required)}")

    candidate_rows, status_rows = [], []
    geometries = network.geometries
    arc_lookup = network._arc_lookup

    for row in track.itertuples(index=False):
        obs_id = int(row.observation_id)
        lon, lat = float(row.longitude), float(row.latitude)
        if (not np.isfinite(lon) or not np.isfinite(lat) or
            lon < -180 or lon > 180 or lat < -90 or lat > 90):
            status_rows.append({"observation_id": obs_id, "candidate_count": 0,
                                "candidate_status": "invalid_coordinate"})
            continue

        x, y = network.transformer.transform(lon, lat)
        point = Point(x, y)
        # Buffer query is spatially indexed; exact distance is then measured.
        indices = network.tree.query(point.buffer(search_radius_m))
        ranked = []
        for idx in np.asarray(indices, dtype=int).tolist():
            geom = geometries[idx]
            distance = float(point.distance(geom))
            if distance <= search_radius_m:
                ranked.append((distance, int(arc_lookup[idx]), geom))
        ranked.sort(key=lambda item: (item[0], item[1]))
        ranked = ranked[:max_candidates]

        if not ranked:
            status_rows.append({"observation_id": obs_id, "candidate_count": 0,
                                "candidate_status": "no_candidate_within_radius"})
            continue

        for distance, arc_id, geom in ranked:
            projected_distance = geom.project(point)
            fraction = float(projected_distance / geom.length) if geom.length else np.nan
            nearest = geom.interpolate(projected_distance)
            candidate_rows.append({
                "observation_id": obs_id,
                "candidate_arc_id": arc_id,
                "distance_to_segment_m": distance,
                "segment_length_m": float(geom.length),
                "projection_fraction": fraction,
                "projected_x": float(nearest.x),
                "projected_y": float(nearest.y),
            })
        status_rows.append({"observation_id": obs_id, "candidate_count": len(ranked),
                            "candidate_status": "ok"})

    candidates = pd.DataFrame(candidate_rows, columns=[
        "observation_id", "candidate_arc_id", "distance_to_segment_m",
        "segment_length_m", "projection_fraction", "projected_x", "projected_y"
    ])
    status = pd.DataFrame(status_rows, columns=[
        "observation_id", "candidate_count", "candidate_status"
    ])
    return candidates, status
