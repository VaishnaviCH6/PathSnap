
from pathlib import Path
import pandas as pd


PARTITIONS = ("train", "student_test")

def load_trip(trip_dir):
    """Load all four files belonging to one trip."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    files = {
        "track": trip_dir / f"{trip_id}.track",
        "nodes": trip_dir / f"{trip_id}.nodes",
        "arcs": trip_dir / f"{trip_id}.arcs",
        "route": trip_dir / f"{trip_id}.route",
    }

    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    return {
        "trip_id": trip_id,
        "track": pd.read_csv(
            files["track"], sep=r"\s+", header=None,
            names=["longitude", "latitude", "timestamp"]
        ),
        "nodes": pd.read_csv(
            files["nodes"], sep=r"\s+", header=None,
            names=["longitude", "latitude"]
        ),
        "arcs": pd.read_csv(
            files["arcs"], sep=r"\s+", header=None,
            names=["source", "target"]
        ),
        "route": pd.read_csv(
            files["route"], sep=r"\s+", header=None,
            names=["arc_id"]
        ),
    }


def load_partition(data_dir, partition):
    """Load every trip in a supported dataset partition."""
    if partition not in PARTITIONS:
        raise ValueError(f"Unsupported partition: {partition}")

    folder = Path(data_dir) / partition

    if not folder.is_dir():
        raise FileNotFoundError(folder)

    return [
        load_trip(path)
        for path in sorted(folder.iterdir())
        if path.is_dir()
    ]
