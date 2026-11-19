"""Adapt M2's directed HMM/Viterbi functions to M4's prediction contract.

This adapter receives only current-trip M2 candidate features and M3 classifier
probabilities. It never reads reference-route files, labels, or evaluator data.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from backend.pipeline import Prediction, TripData, validate_prediction
from src.map_matching.hmm import decode_trip as m2_decode_trip, reconstruct_trajectory as m2_reconstruct_trajectory

IDS = ["trip_id", "observation_id", "candidate_arc_id"]


def decode_trip_for_m4(*, trip: TripData,
                       candidate_features: pd.DataFrame,
                       candidate_scores: pd.DataFrame) -> Prediction:
    """Run M2 Viterbi and return observed arcs + route arcs with explicit breaks."""
    missing_features = {"projection_fraction", "gps_step_distance_m", *IDS} - set(candidate_features)
    missing_scores = {"emission_probability", *IDS} - set(candidate_scores)
    if missing_features or missing_scores:
        raise ValueError(f"Missing HMM inputs: features={sorted(missing_features)}; "
                         f"scores={sorted(missing_scores)}")
    for frame, label in ((candidate_features, "features"), (candidate_scores, "scores")):
        if frame.duplicated(IDS).any():
            raise ValueError(f"Duplicate trip/observation/arc identifiers in {label}")
        if not frame["trip_id"].eq(trip.trip_id).all():
            raise ValueError(f"{label} trip IDs do not match requested trip")

    if len(candidate_features) != len(candidate_scores):
        raise ValueError("M3 score rows must correspond one-to-one with M2 candidates")

    candidates = candidate_features.merge(
        candidate_scores.loc[:, [*IDS, "emission_probability"]],
        on=IDS, how="left", validate="one_to_one",
    )
    if candidates["emission_probability"].isna().any():
        raise ValueError("Missing M3 probability for one or more M2 candidates")
    candidates = candidates.rename(columns={"emission_probability": "positive_probability"})
    if len(candidates) and not np.isfinite(
        candidates["positive_probability"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite M3 scores are not valid HMM emissions")

    nodes = pd.DataFrame(trip.nodes, columns=["longitude", "latitude"])
    arcs = pd.DataFrame(trip.arcs, columns=["from_node_id", "to_node_id"])
    arcs = arcs.astype("int64")
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))

    # Accelerated M4 path; M2 and M3 teammate source code remains unchanged.
    # Set PATHSNAP_HMM_FAST=0 to restore the original decoder for comparison.
    if os.getenv("PATHSNAP_HMM_FAST", "1").strip().lower() in {"0", "false", "no", "off"}:
        decoded = m2_decode_trip(
            candidates, observation_ids=range(len(trip.track)), arcs=arcs, nodes=nodes,
        )
        reconstructed = m2_reconstruct_trajectory(decoded, arcs=arcs, nodes=nodes)
    else:
        from backend.fast_hmm import FastRoadGraph, decode_trip, reconstruct_trajectory

        # Reuse one projected graph for both Viterbi decoding and arc recovery.
        graph = FastRoadGraph(nodes, arcs)
        decoded = decode_trip(
            candidates, observation_ids=range(len(trip.track)), arcs=arcs,
            nodes=nodes, graph=graph,
        )
        reconstructed = reconstruct_trajectory(
            decoded, arcs=arcs, nodes=nodes, graph=graph,
        )

    observations: list[int | None] = [
        None if pd.isna(arc) else int(arc) for arc in decoded["decoded_arc_id"]
    ]
    route: list[int] = []
    break_indices: list[int] = []
    for segment in reconstructed.itertuples(index=False):
        if route:
            break_indices.append(len(route))
        route.extend(int(arc_id) for arc_id in segment.arc_sequence)

    prediction = Prediction(
        observation_arc_ids=observations,
        route_arc_ids=route,
        route_break_indices=break_indices,
    )
    validate_prediction(trip, prediction)
    return prediction
