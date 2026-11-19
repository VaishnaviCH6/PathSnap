"""M4 integration adapter for the actual M3 Random Forest and M2 candidates.

No ground-truth .route data is read here. This module offers two explicit paths:
- model_only: actual M3 candidate scores, independent per-GPS argmax, NO HMM
- integrated: actual M3 candidate scores passed to the team's M2 HMM decoder

Both use M2 feature generation and the saved M3 training-time imputer.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import logging
from collections import OrderedDict
from threading import RLock
from time import perf_counter
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from backend.m2_bridge import FEATURE_COLUMNS, prepare_candidates
from backend.pipeline import Prediction, TripData, validate_prediction

FORBIDDEN_FEATURE_NAMES = {
    "label", "target", "target_label", "correct_arc_id", "ground_truth",
    "reference_arc_id", "is_correct_candidate", "is_correct", "route", "arc_id",
}
REQUIRED_ID_COLUMNS = ("trip_id", "observation_id", "candidate_arc_id")
SCORE_COLUMNS = (*REQUIRED_ID_COLUMNS, "emission_probability")
log = logging.getLogger(__name__)


def import_function(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("Expected callable reference module.submodule:function")
    module_name, function_name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module_name), function_name)
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn


def _validate_sha256(path: Path):
    """Verify known approved handoff checksum before deserializing a model."""
    expected = os.getenv("PATHSNAP_MODEL_SHA256")
    if not expected:
        # Optional sidecar for other team-approved models.
        sidecar = path.with_suffix(".sha256")
        if sidecar.is_file():
            expected = sidecar.read_text(encoding="utf-8").split()[0]
    if expected:
        expected = expected.lower().strip()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Invalid model SHA-256 checksum format")
        with path.open("rb") as artifact:
            actual = hashlib.file_digest(artifact, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Model checksum does not match approved handoff")


def _load_m3_components():
    model_path = Path(os.environ["PATHSNAP_MODEL_PATH"]).expanduser().resolve()
    config_path = Path(os.environ["PATHSNAP_MODEL_CONFIG"]).expanduser().resolve()
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("M3 model .joblib or adjacent .json metadata is missing")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    columns = config.get("feature_columns")
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(c, str) and c for c in columns)):
        raise ValueError("M3 metadata must declare ordered feature_columns")
    if len(set(columns)) != len(columns) or FORBIDDEN_FEATURE_NAMES.intersection(columns):
        raise ValueError("Duplicate, forbidden, or ground-truth feature column")
    if columns != FEATURE_COLUMNS:
        raise ValueError("M3 feature schema/order differs from the current M2-v1 feature schema")
    schema = config.get("feature_schema_version")
    if schema and schema != "M2-v1":
        raise ValueError(f"Unsupported M3 feature schema: {schema}")

    # The approved M3 handoff metadata identifies the positive candidate class
    # as integer 1, but does not contain the optional 'positive_class' field.
    if "positive_class" in config:
        positive_class = config["positive_class"]
    elif schema == "M2-v1":
        positive_class = 1
    else:
        raise ValueError("Unknown positive class; provide positive_class in metadata")

    _validate_sha256(model_path)
    # SECURITY: Joblib loads executable pickle content. Trust team artifacts only.
    model = joblib.load(model_path)
    if not callable(getattr(model, "predict_proba", None)) or not hasattr(model, "classes_"):
        raise TypeError("M3 model must expose fitted predict_proba() and classes_")
    classes = list(model.classes_)
    if positive_class not in classes:
        raise ValueError(f"Positive class {positive_class!r} is absent from fitted classes {classes!r}")
    positive_index = classes.index(positive_class)
    return model, config, columns, positive_index


def _make_scorer():
    model, config, columns, positive_index = _load_m3_components()

    def score(trip: TripData) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepared = prepare_candidates(trip)
        features = prepared.features.copy()
        if features.empty:
            return features, pd.DataFrame(columns=list(SCORE_COLUMNS))
        if not features["trip_id"].eq(trip.trip_id).all():
            raise ValueError("M2 feature trip identifiers do not match the requested trip")
        if not features["observation_id"].between(0, len(trip.track) - 1).all():
            raise ValueError("M2 observation IDs outside source trip")
        if features.duplicated(list(REQUIRED_ID_COLUMNS)).any():
            raise ValueError("Duplicate candidate IDs detected")

        numeric = features.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        probability_matrix = np.asarray(model.predict_proba(numeric), dtype=float)
        if probability_matrix.shape != (len(features), len(model.classes_)):
            raise ValueError("M3 probability output has an unexpected shape")
        positive_scores = probability_matrix[:, positive_index]
        if not np.isfinite(positive_scores).all() or (positive_scores < 0).any() or (positive_scores > 1).any():
            raise ValueError("M3 produced invalid positive-class scores")
        scored = features.loc[:, list(REQUIRED_ID_COLUMNS)].copy()
        scored["emission_probability"] = positive_scores
        return features, scored

    return score


def load_model_only_pipeline():
    """Use real M3 scores without HMM, for non-evaluator development only."""
    score = _make_scorer()  # Model is loaded ONCE during FastAPI lifespan.

    def predict(trip: TripData) -> Prediction:
        features, scored = score(trip)
        observation_arcs: list[int | None] = [None] * len(trip.track)
        if len(scored):
            ranked = scored.join(features[["distance_to_segment_m"]]).sort_values(
                ["observation_id", "emission_probability", "distance_to_segment_m", "candidate_arc_id"],
                ascending=[True, False, True, True], kind="stable",
            ).drop_duplicates("observation_id", keep="first")
            for obs, arc in ranked[["observation_id", "candidate_arc_id"]].itertuples(index=False, name=None):
                observation_arcs[int(obs)] = int(arc)
        # This is NOT a Viterbi route. These independent arcs can be disconnected.
        route: list[int] = []
        for arc in observation_arcs:
            if arc is not None and (not route or route[-1] != arc):
                route.append(arc)
        result = Prediction(observation_arc_ids=observation_arcs, route_arc_ids=route)
        validate_prediction(trip, result)
        return result

    return predict


def load_pipeline():
    """Real M3 model + M2 decoder; fails at STARTUP if decoder is unavailable."""
    spec = os.environ.get("PATHSNAP_HMM_DECODER")
    if not spec:
        raise ValueError("PATHSNAP_HMM_DECODER is required for integrated mode")
    decoder = import_function(spec)
    score = _make_scorer()

    # Bounded cache: immutable predictions only, never model labels/references.
    # Content fingerprints ensure a changed trip does not reuse stale results.
    cache_capacity = max(0, min(10, int(os.getenv("PATHSNAP_PREDICTION_CACHE_SIZE", "2"))))
    cache: OrderedDict[tuple[str, str], Prediction] = OrderedDict()
    lock = RLock()  # Limit simultaneous resource-heavy decodes on modest laptops.

    def fingerprint(trip: TripData) -> tuple[str, str]:
        digest = hashlib.blake2b(digest_size=16)
        for array in (trip.track, trip.nodes, trip.arcs):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return trip.trip_id, digest.hexdigest()

    def predict(trip: TripData) -> Prediction:
        key = fingerprint(trip) if cache_capacity else None
        with lock:
            if key is not None and key in cache:
                cache.move_to_end(key)
                log.info("M4 cache hit: trip=%s", trip.trip_id)
                return cache[key]

            started = perf_counter()
            features, candidate_scores = score(trip)
            scored_seconds = perf_counter() - started
            result = decoder(trip=trip, candidate_features=features,
                             candidate_scores=candidate_scores)
            if not isinstance(result, Prediction):
                raise TypeError("HMM decoder must return backend.pipeline.Prediction")
            validate_prediction(trip, result)
            log.info("M4 predict trip=%s scoring=%.3fs hmm=%.3fs", trip.trip_id,
                     scored_seconds, perf_counter() - started - scored_seconds)
            if key is not None:
                cache[key] = result
                cache.move_to_end(key)
                if len(cache) > cache_capacity:
                    cache.popitem(last=False)
            return result

    return predict
