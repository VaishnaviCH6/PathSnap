"""Inference interface. Reference routes are deliberately absent from TripData."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class TripData:
    trip_id: str
    track: np.ndarray  # longitude, latitude, timestamp
    nodes: np.ndarray  # longitude, latitude
    arcs: np.ndarray   # from_node_id, to_node_id; row index is arc_id


@dataclass(frozen=True)
class Prediction:
    observation_arc_ids: list[int | None]  # one per GPS observation
    route_arc_ids: list[int]  # chronological arcs, including connecting arcs if provided
    route_break_indices: list[int] = field(default_factory=list)  # starts of new disconnected segments


def load_trip(directory: Path, trip_id: str) -> TripData:
    arrays = []
    for suffix, columns in [('track', 3), ('nodes', 2), ('arcs', 2)]:
        a = np.loadtxt(directory / f'{trip_id}.{suffix}', ndmin=2,
                       dtype=np.int64 if suffix == 'arcs' else float)
        if not len(a) or a.shape[1] != columns or not np.isfinite(a).all():
            raise ValueError(f'Invalid {suffix} data')
        a.setflags(write=False)
        arrays.append(a)
    track, nodes, arcs = arrays
    for coordinates in (track[:, :2], nodes):
        if (abs(coordinates[:, 0]) > 180).any() or (abs(coordinates[:, 1]) > 90).any():
            raise ValueError('Invalid geographic coordinates')
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError('Invalid arc node ID')
    if (np.diff(track[:, 2]) <= 0).any():
        raise ValueError('Timestamps must strictly increase')
    return TripData(trip_id, track, nodes, arcs)


def validate_prediction(trip: TripData, prediction: Prediction) -> None:
    from numbers import Integral
    if not isinstance(prediction, Prediction):
        raise TypeError('Pipeline must return Prediction')
    if len(prediction.observation_arc_ids) != len(trip.track):
        raise ValueError('Prediction must have one entry per GPS observation')
    for a in prediction.observation_arc_ids:
        if a is not None and (not isinstance(a, Integral) or isinstance(a, bool) or not 0 <= a < len(trip.arcs)):
            raise ValueError('Invalid observation arc ID')
    breaks = prediction.route_break_indices
    if (any(not isinstance(index, Integral) or isinstance(index, bool)
            or index <= 0 or index >= len(prediction.route_arc_ids)
            for index in breaks)
            or breaks != sorted(set(breaks))):
        raise ValueError('Invalid route break indices')
    for a in prediction.route_arc_ids:
        if not isinstance(a, Integral) or isinstance(a, bool) or not 0 <= a < len(trip.arcs):
            raise ValueError('Invalid route arc ID')
    if any(a is not None for a in prediction.observation_arc_ids) and not prediction.route_arc_ids:
        raise ValueError('Nonempty observations require route arcs')
