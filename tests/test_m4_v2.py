"""M4 v2 tests; synthetic fixtures only, no claimed experimental results."""
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.pipeline import Prediction
from frontend.app import create_app as create_frontend


def test_default_demo_runs_without_dataset(monkeypatch):
    monkeypatch.delenv("PATHSNAP_MODE", raising=False)
    monkeypatch.delenv("PATHSNAP_TRIP_ROOT", raising=False)
    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["mode"] == "demo" and health["trip_count"] == 10
        assert health["pipeline_ready"] is False and health["final_ready"] is False
        trip_ids = client.get("/trips").json()["trip_ids"]
        assert len(trip_ids) == 10 and all(t.startswith("DEMO-") for t in trip_ids)
        result = client.post("/reconstruct", json={"trip_id": trip_ids[0]})
        assert result.status_code == 200
        payload = result.json()
        assert payload["mode"] == "demo" and payload["accuracy"] is None
        assert len(payload["gps"]) == payload["observation_count"]
        assert len(payload["observation_arc_ids"]) == payload["observation_count"]
        assert payload["route_arc_count"] == len(payload["route_arc_ids"])
        assert all(set(seg) == {"arc_id", "coordinates"} for seg in payload["segments"])


def test_all_demo_trips_reconstruct():
    with TestClient(create_app(mode="demo", partition="synthetic")) as client:
        for trip_id in client.get("/trips").json()["trip_ids"]:
            response = client.post("/reconstruct", json={"trip_id": trip_id})
            assert response.status_code == 200, trip_id
            assert response.json()["trip_id"] == trip_id


@pytest.mark.parametrize("trip_id,expected", [("abc", 422), ("../etc", 422), (1234, 422),
                                             ("DEMO-999", 404), ("00000000", 404)])
def test_bad_trip_request(trip_id, expected):
    with TestClient(create_app(mode="demo", partition="synthetic")) as client:
        assert client.post("/reconstruct", json={"trip_id": trip_id}).status_code == expected


def test_extra_payload_rejected():
    with TestClient(create_app(mode="demo", partition="synthetic")) as client:
        assert client.post("/reconstruct", json={"trip_id": "DEMO-001", "secret": 4}).status_code == 422


def test_api_key_protects_data_routes(monkeypatch):
    monkeypatch.setenv("PATHSNAP_API_KEY", "test-key")
    with TestClient(create_app(mode="demo", partition="synthetic")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/trips").status_code == 401
        assert client.get("/trips", headers={"X-PathSnap-Key": "test-key"}).status_code == 200
        assert client.post("/reconstruct", json={"trip_id": "DEMO-001"}).status_code == 401


def test_no_evaluator_preview():
    with pytest.raises(ValueError):
        create_app(mode="preview", partition="evaluator_test")


def test_frontend_proxy_and_generated_folium_map(monkeypatch):
    monkeypatch.delenv("PATHSNAP_API_KEY", raising=False)
    with TestClient(create_app(mode="demo", partition="synthetic")) as backend:
        def handler(request):
            response = backend.request(request.method, request.url.path,
                                       content=request.content,
                                       headers={"Content-Type": "application/json"})
            return httpx.Response(response.status_code, content=response.content)
        with TestClient(create_frontend("http://backend", httpx.MockTransport(handler))) as frontend:
            assert "PathSnap" in frontend.get("/").text
            assert frontend.get("/api/health").json()["mode"] == "demo"
            assert len(frontend.get("/api/trips").json()["trip_ids"]) == 10
            data = frontend.post("/api/reconstruct", json={"trip_id": "DEMO-002"}).json()
            assert data["trip_id"] == "DEMO-002"
            assert "L.polyline" in data["map_html"]
            assert "reference_route" not in data


def test_frontend_bad_backend_connection():
    def offline(request):
        raise httpx.ConnectError("offline", request=request)
    with TestClient(create_frontend("http://missing", httpx.MockTransport(offline))) as client:
        assert client.get("/api/trips").status_code == 502


def test_integrated_factory_contract_without_real_model(tmp_path):
    root = tmp_path / "student_test"
    for i in range(10):
        tid = f"{i:08d}"
        tdir = root / tid
        tdir.mkdir(parents=True)
        (tdir / f"{tid}.track").write_text("77 12 0\n77.0004 12 1\n", encoding="utf-8")
        (tdir / f"{tid}.nodes").write_text("77 12\n77.0004 12\n", encoding="utf-8")
        (tdir / f"{tid}.arcs").write_text("0 1\n", encoding="utf-8")
        (tdir / f"{tid}.route").write_text("INVALID REFERENCE — NEVER READ", encoding="utf-8")
    seen = []
    def factory():
        seen.append("factory")
        def inference(trip):
            assert not hasattr(trip, "route")
            seen.append("inference")
            return Prediction([0, 0], [0])
        return inference
    with TestClient(create_app(root=root, mode="integrated", factory=factory)) as client:
        result = client.post("/reconstruct", json={"trip_id": "00000000"})
        assert result.status_code == 200
        assert result.json()["mode"] == "integrated"
        assert result.json()["accuracy"] is None
        assert seen == ["factory", "inference"]


def test_non_demo_missing_dataset_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Trip directory not found"):
        with TestClient(create_app(root=tmp_path / "student_test", mode="preview")):
            pass
