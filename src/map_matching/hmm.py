"""Directed-road HMM/Viterbi decoder and trajectory reconstruction for PathSnap."""
from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np
import pandas as pd

from src.candidate_generation.candidates import build_road_network


_REQUIRED_CANDIDATES = {
    "observation_id",
    "candidate_arc_id",
    "projection_fraction",
    "positive_probability",
}
_REQUIRED_ARCS = {"arc_id", "from_node_id", "to_node_id"}
_REQUIRED_NODES = {"longitude", "latitude"}


class DirectedRoadGraph:
    """Directed graph whose edge weights are projected road-segment lengths."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        if not _REQUIRED_NODES.issubset(nodes.columns):
            raise ValueError(f"nodes must contain {sorted(_REQUIRED_NODES)}")
        if not _REQUIRED_ARCS.issubset(arcs.columns):
            raise ValueError(f"arcs must contain {sorted(_REQUIRED_ARCS)}")

        network = build_road_network(nodes, arcs)
        valid_ids = network._arc_lookup.tolist()
        arc_rows = arcs.set_index("arc_id")
        self.arc_info: dict[int, tuple[int, int, float]] = {}
        self.adjacency: dict[int, list[tuple[int, float, int]]] = {}

        for arc_id, geom in zip(valid_ids, network.geometries):
            arc_id = int(arc_id)
            row = arc_rows.loc[arc_id]
            source = int(row["from_node_id"])
            target = int(row["to_node_id"])
            length = float(geom.length)
            self.arc_info[arc_id] = (source, target, length)
            self.adjacency.setdefault(source, []).append((target, length, arc_id))

        self._distances_from = lru_cache(maxsize=256)(self._dijkstra_distances)
        self._paths_from = lru_cache(maxsize=256)(self._dijkstra_paths)

    def _dijkstra_distances(self, source: int) -> dict[int, float]:
        distances = {int(source): 0.0}
        queue = [(0.0, int(source))]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            for neighbour, weight, _ in self.adjacency.get(node, []):
                candidate = distance + weight
                if candidate < distances.get(neighbour, float("inf")):
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        return distances

    def _dijkstra_paths(self, source: int) -> dict[int, tuple[float, tuple[int, ...]]]:
        best: dict[int, tuple[float, tuple[int, ...]]] = {int(source): (0.0, ())}
        queue = [(0.0, int(source))]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != best[node][0]:
                continue
            for neighbour, weight, arc_id in self.adjacency.get(node, []):
                candidate_distance = distance + weight
                candidate_path = best[node][1] + (arc_id,)
                previous = best.get(neighbour)
                if (
                    previous is None
                    or candidate_distance < previous[0] - 1e-9
                    or (
                        abs(candidate_distance - previous[0]) <= 1e-9
                        and candidate_path < previous[1]
                    )
                ):
                    best[neighbour] = (candidate_distance, candidate_path)
                    heapq.heappush(queue, (candidate_distance, neighbour))
        return best

    def shortest_path_arc_ids(self, source_node: int, target_node: int) -> list[int] | None:
        """Return ordered directed arc IDs from source_node to target_node."""
        result = self._paths_from(int(source_node)).get(int(target_node))
        return None if result is None else list(result[1])

    def transition_distance(
        self,
        previous_arc_id: int,
        previous_fraction: float,
        current_arc_id: int,
        current_fraction: float,
        backward_tolerance_m: float = 10.0,
    ) -> float | None:
        """Directed distance between projected positions; None means disconnected."""
        previous_arc_id = int(previous_arc_id)
        current_arc_id = int(current_arc_id)
        if previous_arc_id not in self.arc_info or current_arc_id not in self.arc_info:
            return None

        p_source, p_target, p_length = self.arc_info[previous_arc_id]
        c_source, c_target, c_length = self.arc_info[current_arc_id]
        pf = float(previous_fraction)
        cf = float(current_fraction)

        if not (np.isfinite(pf) and np.isfinite(cf) and 0 <= pf <= 1 and 0 <= cf <= 1):
            return None

        if previous_arc_id == current_arc_id:
            backward_m = (pf - cf) * p_length
            if backward_m > backward_tolerance_m:
                return None
            return max(0.0, cf - pf) * p_length

        middle = self._distances_from(p_target).get(c_source)
        if middle is None:
            return None

        return (1.0 - pf) * p_length + middle + cf * c_length

    def transition_arc_ids(
        self,
        previous_arc_id: int,
        previous_fraction: float,
        current_arc_id: int,
        current_fraction: float,
        backward_tolerance_m: float = 10.0,
    ) -> list[int] | None:
        """Road arcs traversed from one projected candidate position to the next."""
        distance = self.transition_distance(
            previous_arc_id, previous_fraction,
            current_arc_id, current_fraction,
            backward_tolerance_m,
        )
        if distance is None:
            return None

        previous_arc_id = int(previous_arc_id)
        current_arc_id = int(current_arc_id)
        if previous_arc_id == current_arc_id:
            return [previous_arc_id]

        _, previous_target, _ = self.arc_info[previous_arc_id]
        current_source, _, _ = self.arc_info[current_arc_id]
        middle = self.shortest_path_arc_ids(previous_target, current_source)
        if middle is None:
            return None
        return [previous_arc_id, *middle, current_arc_id]


def decode_trip(
    candidates: pd.DataFrame,
    observation_ids,
    arcs: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    transition_scale_m: float = 50.0,
    backward_tolerance_m: float = 10.0,
) -> pd.DataFrame:
    """Viterbi decode with safe restarts when all reachable states lose connectivity."""
    missing = _REQUIRED_CANDIDATES - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    if transition_scale_m <= 0:
        raise ValueError("transition_scale_m must be positive")

    obs_ids = [int(value) for value in observation_ids]
    if len(obs_ids) != len(set(obs_ids)):
        raise ValueError("observation_ids must be unique and ordered")

    data = candidates.copy()
    data["observation_id"] = pd.to_numeric(data["observation_id"], errors="raise").astype(int)
    data["candidate_arc_id"] = pd.to_numeric(data["candidate_arc_id"], errors="raise").astype(int)
    data["projection_fraction"] = pd.to_numeric(data["projection_fraction"], errors="coerce")
    data["positive_probability"] = pd.to_numeric(data["positive_probability"], errors="coerce")

    if data.duplicated(["observation_id", "candidate_arc_id"]).any():
        raise ValueError("Duplicate candidate arc for an observation")
    if not data["positive_probability"].between(0, 1).all():
        raise ValueError("positive_probability values must be finite and in [0, 1]")
    if not data["projection_fraction"].between(0, 1).all():
        raise ValueError("projection_fraction values must be in [0, 1]")

    graph = DirectedRoadGraph(nodes, arcs)
    groups = {
        int(obs_id): group.reset_index(drop=True)
        for obs_id, group in data.groupby("observation_id", sort=False)
    }

    output = []
    run_obs: list[int] = []
    run_groups: list[pd.DataFrame] = []
    run_status: list[str] = []
    run_scores: np.ndarray | None = None
    run_backpointers: list[np.ndarray] = []

    def start_run(obs_id: int, group: pd.DataFrame, status: str) -> None:
        nonlocal run_obs, run_groups, run_status, run_scores, run_backpointers
        emissions = np.log(np.clip(
            group["positive_probability"].to_numpy(dtype=float), 1e-12, 1.0
        ))
        run_obs = [obs_id]
        run_groups = [group]
        run_status = [status]
        run_scores = emissions
        run_backpointers = [np.full(len(group), -1, dtype=int)]

    def flush_run() -> None:
        nonlocal run_obs, run_groups, run_status, run_scores, run_backpointers
        if not run_obs:
            return
        assert run_scores is not None
        state_index = int(np.argmax(run_scores))
        selected = [None] * len(run_obs)
        for position in range(len(run_obs) - 1, -1, -1):
            selected[position] = run_groups[position].iloc[state_index]
            if position:
                state_index = int(run_backpointers[position][state_index])

        for position, (obs_id, chosen) in enumerate(zip(run_obs, selected)):
            output.append({
                "observation_id": obs_id,
                "decoded_arc_id": int(chosen["candidate_arc_id"]),
                "decoded_projection_fraction": float(chosen["projection_fraction"]),
                "selected_probability": float(chosen["positive_probability"]),
                "decode_status": run_status[position],
            })

        run_obs, run_groups, run_status = [], [], []
        run_scores = None
        run_backpointers = []

    for obs_id in obs_ids:
        group = groups.get(obs_id)
        if group is None or group.empty:
            flush_run()
            output.append({
                "observation_id": obs_id,
                "decoded_arc_id": pd.NA,
                "decoded_projection_fraction": np.nan,
                "selected_probability": np.nan,
                "decode_status": "no_candidates",
            })
            continue

        if not run_obs:
            start_run(obs_id, group, "decoded")
            continue

        assert run_scores is not None
        previous_group = run_groups[-1]
        emissions = np.log(np.clip(
            group["positive_probability"].to_numpy(dtype=float), 1e-12, 1.0
        ))
        gps_step = float(group.iloc[0].get("gps_step_distance_m", np.nan))
        new_scores = np.full(len(group), -np.inf, dtype=float)
        back = np.full(len(group), -1, dtype=int)

        # Only predecessor states with finite accumulated Viterbi scores count.
        for current_index, current in group.iterrows():
            for previous_index, previous in previous_group.iterrows():
                if not np.isfinite(run_scores[previous_index]):
                    continue
                road_distance = graph.transition_distance(
                    previous["candidate_arc_id"], previous["projection_fraction"],
                    current["candidate_arc_id"], current["projection_fraction"],
                    backward_tolerance_m,
                )
                if road_distance is None:
                    continue
                transition_log_score = 0.0
                if np.isfinite(gps_step) and gps_step >= 0:
                    transition_log_score = (
                        -abs(road_distance - gps_step) / transition_scale_m
                    )
                score = (
                    run_scores[previous_index]
                    + transition_log_score
                    + emissions[current_index]
                )
                if score > new_scores[current_index]:
                    new_scores[current_index] = score
                    back[current_index] = previous_index

        if not np.isfinite(new_scores).any():
            # No reachable Viterbi state can reach this observation.
            # Finalize the old segment and start a new, explicitly marked one.
            flush_run()
            start_run(obs_id, group, "disconnected_restart")
            continue

        run_obs.append(obs_id)
        run_groups.append(group)
        run_status.append("decoded")
        run_scores = new_scores
        run_backpointers.append(back)

    flush_run()

    result = pd.DataFrame(output).sort_values("observation_id").reset_index(drop=True)
    result["decoded_arc_id"] = result["decoded_arc_id"].astype("Int64")
    return result


def reconstruct_trajectory(
    decoded: pd.DataFrame,
    arcs: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    backward_tolerance_m: float = 10.0,
) -> pd.DataFrame:
    """Build ordered road-arc sequences, preserving missing/disconnected breaks.

    Returns one row per continuous segment. `arc_sequence` is an ordered list of
    original arc IDs. No route is fabricated across a missing observation or a
    disconnected restart.
    """
    required = {
        "observation_id", "decoded_arc_id", "decoded_projection_fraction", "decode_status"
    }
    missing = required - set(decoded.columns)
    if missing:
        raise ValueError(f"decoded data missing columns: {sorted(missing)}")

    graph = DirectedRoadGraph(nodes, arcs)
    ordered = decoded.sort_values("observation_id", kind="stable")
    segments = []
    current_obs = []
    current_arcs = []
    previous = None
    break_reason = "trip_start"

    def flush():
        nonlocal current_obs, current_arcs, previous, break_reason
        if current_obs:
            segments.append({
                "segment_id": len(segments),
                "start_observation_id": current_obs[0],
                "end_observation_id": current_obs[-1],
                "observation_ids": current_obs.copy(),
                "observation_count": len(current_obs),
                "arc_sequence": current_arcs.copy(),
                "arc_count": len(current_arcs),
                "segment_start_reason": break_reason,
            })
        current_obs = []
        current_arcs = []
        previous = None

    for row in ordered.itertuples(index=False):
        obs_id = int(row.observation_id)
        status = str(row.decode_status)
        if pd.isna(row.decoded_arc_id) or status == "no_candidates":
            flush()
            break_reason = "after_missing_candidates"
            continue

        arc_id = int(row.decoded_arc_id)
        fraction = float(row.decoded_projection_fraction)
        if previous is None or status == "disconnected_restart":
            flush()
            current_obs = [obs_id]
            current_arcs = [arc_id]
            previous = (arc_id, fraction)
            if status == "disconnected_restart":
                break_reason = "disconnected_restart"
            elif not segments:
                break_reason = "trip_start"
            else:
                break_reason = "after_gap"
            continue

        transition = graph.transition_arc_ids(
            previous[0], previous[1], arc_id, fraction, backward_tolerance_m
        )
        if transition is None:
            flush()
            current_obs = [obs_id]
            current_arcs = [arc_id]
            previous = (arc_id, fraction)
            break_reason = "disconnected_restart"
            continue

        for path_arc_id in transition:
            if not current_arcs or current_arcs[-1] != path_arc_id:
                current_arcs.append(path_arc_id)
        current_obs.append(obs_id)
        previous = (arc_id, fraction)

    flush()
    return pd.DataFrame(segments, columns=[
        "segment_id", "start_observation_id", "end_observation_id",
        "observation_ids", "observation_count", "arc_sequence", "arc_count",
        "segment_start_reason",
    ])
