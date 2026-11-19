"""PathSnap M4 FastAPI entry point.

Demo mode: synthetic trips, no dataset/model necessary.
Preview mode: real student-test GPS and teammate M2 nearest-candidate baseline.
Model-only mode: actual M3 model + M2 candidates, but NOT the HMM decoder.
Integrated mode: teammate M2 + saved M3 model + actual M2 HMM/Viterbi.

Ground-truth .route files are NOT read by the prediction pipeline.
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.demo import DEMO_IDS, demo_prediction, make_demo_trip
from backend.pipeline import load_trip, validate_prediction

log = logging.getLogger("uvicorn.error")
ID_PATTERN = r"^(?:[0-9]{8}|DEMO-[0-9]{3})$"


class ReconstructionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str = Field(pattern=ID_PATTERN, examples=["DEMO-001", "00000052"])


class GPSPoint(BaseModel):
    lat: float
    lon: float
    timestamp: float


class RouteSegment(BaseModel):
    arc_id: int
    coordinates: list[tuple[float, float]]  # latitude, longitude


class ReconstructionResult(BaseModel):
    trip_id: str
    mode: Literal["demo", "preview", "model_only", "integrated"]
    partition: Literal["synthetic", "student_test", "evaluator_test"]
    gps: list[GPSPoint]
    observation_arc_ids: list[int | None]
    route_arc_ids: list[int]
    route_break_indices: list[int] = Field(default_factory=list)
    route_segment_count: int = 0
    segments: list[RouteSegment]
    observation_count: int
    matched_count: int
    route_arc_count: int
    route_continuity: float | None
    inference_ms: float
    accuracy: float | None = Field(default=None, ge=0, le=1)
    notice: str


def _import_callable(spec: str):
    if ":" not in spec:
        raise ValueError("Callable setting must be module.path:function_name")
    module_name, func_name = spec.split(":", 1)
    func = getattr(importlib.import_module(module_name), func_name)
    if not callable(func):
        raise TypeError(f"Configured {spec!r} is not callable")
    return func


def create_app(root=None, mode=None, partition=None, factory=None, evaluator=None):
    # An explicitly passed root defaults to the historical preview mode for
    # backwards-compatible tests; a no-argument startup defaults to demo mode.
    root_given = root is not None or bool(os.getenv("PATHSNAP_TRIP_ROOT"))
    mode = mode or os.getenv("PATHSNAP_MODE") or ("preview" if root_given else "demo")
    partition = partition or os.getenv("PATHSNAP_PARTITION") or ("synthetic" if mode == "demo" else "student_test")
    root = Path(root or os.getenv("PATHSNAP_TRIP_ROOT") or f"dataset/{partition}")
    if mode not in ("demo", "preview", "model_only", "integrated"):
        raise ValueError(f"Unknown mode {mode!r}")
    if mode == "demo" and partition != "synthetic":
        raise ValueError("Synthetic demonstration requires partition=synthetic")
    if mode != "demo" and partition not in ("student_test", "evaluator_test"):
        raise ValueError("Data-backed mode requires student_test or evaluator_test")
    if mode in ("preview", "model_only") and partition != "student_test":
        raise ValueError("Preview and model-only modes are not permitted for evaluator data")
    if mode != "demo" and root.name != partition:
        raise ValueError("PATHSNAP_TRIP_ROOT folder name must match PATHSNAP_PARTITION")
    if partition == "evaluator_test" and os.getenv("PATHSNAP_FINAL_EVALUATION") != "1":
        raise ValueError("Reserved evaluation requires explicit final-evaluation opt-in")

    required_key = os.getenv("PATHSNAP_API_KEY", "")

    def authenticated(x_pathsnap_key: str | None = Header(default=None, alias="X-PathSnap-Key")):
        if required_key and not secrets.compare_digest(x_pathsnap_key or "", required_key):
            raise HTTPException(status_code=401, detail="Invalid backend API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if mode == "demo":
            app.state.ids = list(DEMO_IDS)
            app.state.predict = demo_prediction
        else:
            if not root.is_dir():
                raise RuntimeError(f"Trip directory not found: {root}")
            ids = sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]{8}", p.name)
                         and all((p / f"{p.name}.{s}").is_file() for s in ("track", "nodes", "arcs")))
            if len(ids) != 10:
                raise RuntimeError(f"Expected 10 complete trips in {root}; found {len(ids)}")
            app.state.ids = ids
            if mode == "preview":
                from backend.m2_bridge import m2_preview_predict
                app.state.predict = m2_preview_predict
            else:
                if mode == "model_only":
                    from backend.real_inference import load_model_only_pipeline
                    loader = factory or load_model_only_pipeline
                else:
                    loader = factory or _import_callable(os.environ["PATHSNAP_PIPELINE_FACTORY"])
                app.state.predict = loader()  # Load the model exactly once
                if not callable(app.state.predict):
                    raise TypeError("Pipeline factory must return a callable")
        app.state.evaluate = None
        if mode == "integrated":
            setting = os.getenv("PATHSNAP_EVALUATOR")
            app.state.evaluate = evaluator or (_import_callable(setting) if setting else None)
            if partition == "evaluator_test" and app.state.evaluate is None:
                raise RuntimeError("Evaluator mode requires a separate server-side evaluator")
        log.info("PathSnap ready: mode=%s partition=%s trips=%s", mode, partition, len(app.state.ids))
        yield

    app = FastAPI(title="PathSnap | M4 Route Reconstruction API", version="2.0.0",
                  description="Trip selection, route reconstruction, and server-side results. "
                              "Demo mode uses only synthetic trips. See /docs.", lifespan=lifespan)

    @app.get("/health", tags=["System"])
    def health():
        return {
            "status": "ok", "mode": mode, "partition": partition,
            "trip_count": len(app.state.ids),
            "pipeline_ready": mode == "integrated",
            "model_ready": mode in ("model_only", "integrated"),
            "decoder_ready": mode == "integrated",
            "final_ready": mode == "integrated" and partition == "evaluator_test",
            "notice": "DEVELOPMENT ONLY — synthetic routes" if mode == "demo" else
                      ("DEVELOPMENT ONLY — M2 nearest-candidate baseline" if mode == "preview" else
                       ("REAL M3 ML scores — classifier-only, NO HMM/Viterbi route decoder" if mode == "model_only" else
                        "Integrated ML and HMM route reconstruction")),
        }

    @app.get("/trips", tags=["Trips"], dependencies=[Depends(authenticated)])
    def trips():
        return {"mode": mode, "partition": partition, "trip_ids": app.state.ids}

    @app.post("/reconstruct", response_model=ReconstructionResult, tags=["Trips"],
              dependencies=[Depends(authenticated)], responses={401: {"description": "Authentication required"},
                                                                   404: {"description": "Unknown trip"},
                                                                   422: {"description": "Invalid request"},
                                                                   500: {"description": "Internal prediction failure"}})
    def reconstruct(request: ReconstructionRequest):
        trip_id = request.trip_id
        if trip_id not in app.state.ids:
            raise HTTPException(404, "Trip ID not available in the current mode/partition")
        start = perf_counter()
        try:
            trip = make_demo_trip(trip_id) if mode == "demo" else load_trip(root / trip_id, trip_id)
            prediction = app.state.predict(trip)
            validate_prediction(trip, prediction)
            elapsed_ms = round((perf_counter() - start) * 1000, 3)
            # Only a separately configured, trusted evaluator gets access to labels.
            # It receives no .route data from the prediction pipeline.
            accuracy = None
            if app.state.evaluate is not None:
                accuracy = float(app.state.evaluate(trip_id, prediction))
            route = [int(a) for a in prediction.route_arc_ids]
            observations = [None if a is None else int(a) for a in prediction.observation_arc_ids]
            segments = [RouteSegment(arc_id=a, coordinates=[
                (float(trip.nodes[n][1]), float(trip.nodes[n][0])) for n in trip.arcs[a]
            ]) for a in route]
            # A marked break is never counted as a connected arc transition.
            # This is an arc-adjacency score, not ground-truth route accuracy.
            breaks = prediction.route_break_indices
            continuity = (sum(
                (index + 1 not in breaks)
                and (int(trip.arcs[a, 1]) == int(trip.arcs[b, 0]))
                for index, (a, b) in enumerate(zip(route, route[1:]))
            ) / (len(route) - 1)) if len(route) > 1 else None
            result = ReconstructionResult(
                trip_id=trip_id, mode=mode, partition=partition,
                gps=[GPSPoint(lon=float(lon), lat=float(lat), timestamp=float(time))
                     for lon, lat, time in trip.track],
                observation_arc_ids=observations, route_arc_ids=route, segments=segments,
                route_break_indices=breaks,
                route_segment_count=(len(breaks) + 1 if route else 0),
                observation_count=len(observations), matched_count=sum(a is not None for a in observations),
                route_arc_count=len(route), route_continuity=continuity,
                inference_ms=elapsed_ms, accuracy=accuracy,
                notice=("SYNTHETIC DEMO: no real trip, ML, or HMM" if mode == "demo" else
                        "M2 distance-only preview: not ML/HMM" if mode == "preview" else
                        ("REAL M3 classifier-only candidate selection: HMM/Viterbi NOT connected" if mode == "model_only" else
                         "M3 model + M2 HMM/Viterbi prediction")),
            )
            log.info("reconstruct trip=%s mode=%s observations=%d arcs=%d ms=%.1f",
                     trip_id, mode, len(observations), len(route), elapsed_ms)
            return result
        except Exception:
            log.exception("Reconstruction failed, trip=%s", trip_id)
            raise HTTPException(500, "Route reconstruction failed; see backend logs") from None

    return app


app = create_app()
