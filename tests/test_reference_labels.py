import numpy as np
import pandas as pd
from types import SimpleNamespace
from pyproj import Transformer
from shapely.geometry import LineString
from labeling.reference_labels import align_observations_to_route, label_candidate_table


def test_alignment_and_one_positive_per_observation():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    xy = [transformer.transform(lon, 50.0) for lon in (8.0, 8.001, 8.002, 8.003)]
    geometries = [LineString([xy[i], xy[i + 1]]) for i in range(3)]
    network = SimpleNamespace(_arc_lookup=np.array([0, 1, 2]),
                              geometries=geometries, transformer=transformer)
    coords = [(8.0002, 50.00001), (8.0015, 50.00001), (8.0027, 50.00001)]
    track = pd.DataFrame({"observation_id": [0, 1, 2],
                          "longitude": [p[0] for p in coords],
                          "latitude": [p[1] for p in coords],
                          "gps_step_distance_m": [90.0, 90.0, 90.0]})
    alignment = align_observations_to_route(track, [0, 1, 2], network)
    candidates = pd.DataFrame({
        "observation_id": [0, 0, 1, 1, 2, 2],
        "candidate_arc_id": [0, 2, 0, 1, 1, 2],
        "distance_to_segment_m": [1, 5, 2, 1, 4, 1],
        "direction_difference_deg": [0] * 6,
        "gps_step_distance_m": [90] * 6, "time_difference_s": [1] * 6,
        "gps_speed_mps": [90] * 6, "segment_length_m": [70] * 6,
        "projection_fraction": [.2] * 6, "gps_heading_deg": [90] * 6,
        "road_bearing_deg": [90] * 6, "gps_heading_change_deg": [0] * 6,
    })
    candidate_status = pd.DataFrame({"observation_id": [0, 1, 2],
                                     "candidate_count": [2, 2, 2],
                                     "candidate_status": ["ok"] * 3})
    labelled, status = label_candidate_table("trip", candidates, candidate_status, alignment)
    assert labelled.groupby("observation_id").label.sum().tolist() == [1, 1, 1]
    assert status.supervised_eligible.all()
    assert labelled.columns.tolist() == [
        "trip_id", "observation_id", "candidate_arc_id", "distance_to_segment_m",
        "direction_difference_deg", "gps_step_distance_m", "time_difference_s",
        "gps_speed_mps", "segment_length_m", "projection_fraction", "gps_heading_deg",
        "road_bearing_deg", "gps_heading_change_deg", "label"]
