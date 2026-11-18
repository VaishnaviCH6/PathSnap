
def validate_trip(trip):
    """Validate GPS observations and road-network references."""

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    results = {
        "trip_id": trip["trip_id"],
        "missing_gps": int(track.isna().sum().sum()),
        "duplicate_gps": int(track.duplicated().sum()),
        "invalid_coordinates": int((
            ~track["longitude"].between(-180, 180) |
            ~track["latitude"].between(-90, 90)
        ).sum()),
        "timestamp_reversals": int(
            (track["timestamp"].diff() < 0).sum()
        ),
        "zero_timestamp_intervals": int(
            (track["timestamp"].diff() == 0).sum()
        ),
        "timestamp_gaps": int(
            (track["timestamp"].diff() > 1).sum()
        ),
        "invalid_arc_nodes": int((
            (arcs["source"] < 0) |
            (arcs["target"] < 0) |
            (arcs["source"] >= len(nodes)) |
            (arcs["target"] >= len(nodes))
        ).sum()),
        "invalid_route_arcs": int((
            (route["arc_id"] < 0) |
            (route["arc_id"] >= len(arcs))
        ).sum()),
    }

    results["valid"] = not any(
        results[key] > 0
        for key in [
            "missing_gps",
            "duplicate_gps",
            "invalid_coordinates",
            "timestamp_reversals",
            "zero_timestamp_intervals",
            "invalid_arc_nodes",
            "invalid_route_arcs"
        ]
    )

    results["has_timestamp_gaps"] = results["timestamp_gaps"] > 0

    return results
