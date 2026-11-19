
"""Evaluator-facing frontend server. No models, datasets, or reference labels here.

Laptop 2 contacts Laptop 1 through server-to-server HTTP. Folium maps are
created exclusively from the actual JSON returned by the backend.
"""
from __future__ import annotations

import os
from pathlib import Path

import folium
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from folium.plugins import Fullscreen


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str = Field(pattern=r"^(?:[0-9]{8}|DEMO-[0-9]{3})$")


def build_map(result: dict) -> str:
    gps_values = result.get("gps")

    if not isinstance(gps_values, list) or not gps_values:
        raise ValueError("Backend provided no GPS observations")

    gps = [
        (float(p["lat"]), float(p["lon"]))
        for p in gps_values
    ]

    fmap = folium.Map(
        location=gps[0],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    Fullscreen(position="topright").add_to(fmap)

    observed = folium.FeatureGroup(
        name="Observed GPS",
        show=True
    ).add_to(fmap)

    current = [gps[0]]

    for idx in range(1, len(gps)):
        previous = float(gps_values[idx - 1]["timestamp"])
        now = float(gps_values[idx]["timestamp"])

        if now - previous > 30:
            if len(current) > 1:
                folium.PolyLine(
                    current,
                    color="#3b82f6",
                    weight=3,
                    opacity=0.78,
                    tooltip="Observed GPS"
                ).add_to(observed)

            current = []

        current.append(gps[idx])

    if len(current) > 1:
        folium.PolyLine(
            current,
            color="#3b82f6",
            weight=3,
            opacity=0.78,
            tooltip="Observed GPS"
        ).add_to(observed)

    stride = max(1, (len(gps) + 199) // 200)

    for p in gps[::stride]:
        folium.CircleMarker(
            p,
            radius=2,
            weight=0,
            color="#3b82f6",
            fill=True,
            fill_opacity=0.85
        ).add_to(observed)

    classifier_only = result.get("mode") == "model_only"

    segment_label = (
        "Classifier-selected road segments (NO HMM)"
        if classifier_only
        else "Predicted road route"
    )

    predicted = folium.FeatureGroup(
        name=segment_label,
        show=True
    ).add_to(fmap)

    # Preserve graph discontinuities.
    # Never draw imaginary straight-line connections.
    run = []
    all_points = list(gps)
    breaks = set(result.get("route_break_indices", []))

    for index, segment in enumerate(result.get("segments", [])):
        points = [
            (float(lat), float(lon))
            for lat, lon in segment["coordinates"]
        ]

        if len(points) < 2:
            raise ValueError("Invalid road segment geometry")

        all_points.extend(points)

        if run and index not in breaks and run[-1] == points[0]:
            run.extend(points[1:])
        else:
            if len(run) > 1:
                folium.PolyLine(
                    run,
                    color="#10b981",
                    weight=6,
                    opacity=0.88,
                    tooltip=segment_label
                ).add_to(predicted)

            run = list(points)

    if len(run) > 1:
        folium.PolyLine(
            run,
            color="#10b981",
            weight=6,
            opacity=0.88,
            tooltip=segment_label
        ).add_to(predicted)

    folium.Marker(
        gps[0],
        tooltip="GPS start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(fmap)

    folium.Marker(
        gps[-1],
        tooltip="GPS end",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(fmap)

    fmap.fit_bounds(all_points, padding=(16, 16))
    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap.get_root().render()


def create_app(api_url=None, transport=None):
    api_url = (
        api_url
        or os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")
    ).rstrip("/")

    api_key = os.getenv("PATHSNAP_API_KEY", "")

    app = FastAPI(
        title="PathSnap M4 Frontend",
        version="2.1.0"
    )

    def request_backend(method, path, **kwargs):
        headers = {"X-PathSnap-Key": api_key} if api_key else {}

        try:
            with httpx.Client(
                base_url=api_url,
                timeout=httpx.Timeout(
                    connect=8.0,
                    read=300.0,
                    write=30.0,
                    pool=10.0
                ),
                transport=transport
            ) as client:
                response = client.request(
                    method,
                    path,
                    headers=headers,
                    **kwargs
                )

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            raise HTTPException(
                504,
                "Backend reconstruction timed out; check the backend logs"
            ) from None

        except httpx.HTTPStatusError as error:
            code = error.response.status_code

            if code in (401, 404, 422, 503):
                detail = error.response.json().get(
                    "detail",
                    f"Backend HTTP {code}"
                )
                raise HTTPException(code, str(detail)) from None

            raise HTTPException(
                502,
                f"Backend returned HTTP {code}"
            ) from None

        except (httpx.RequestError, ValueError):
            raise HTTPException(
                502,
                "Cannot contact backend; check its IP, port and firewall"
            ) from None

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(
            Path(__file__).with_name("index.html")
        )

    @app.get("/api/health")
    def health():
        return request_backend("GET", "/health")

    @app.get("/api/trips")
    def trips():
        return request_backend("GET", "/trips")

    @app.post("/api/reconstruct")
    def reconstruct(selection: Selection):
        result = request_backend(
            "POST",
            "/reconstruct",
            json=selection.model_dump()
        )

        try:
            if result["trip_id"] != selection.trip_id:
                raise ValueError(
                    "Response trip ID does not match request"
                )

            rendered_map = build_map(result)

            public_fields = (
                "trip_id",
                "mode",
                "partition",
                "observation_count",
                "matched_count",
                "route_arc_count",
                "route_continuity",
                "inference_ms",
                "accuracy",
                "notice",
                "route_arc_ids"
            )

            return {
                **{
                    key: result[key]
                    for key in public_fields
                },
                "route_break_indices": result.get(
                    "route_break_indices", []
                ),
                "route_segment_count": result.get(
                    "route_segment_count", 0
                ),
                "map_html": rendered_map
            }

        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            OverflowError
        ):
            raise HTTPException(
                502,
                "Backend returned invalid route information"
            ) from None

    return app


app = create_app()
