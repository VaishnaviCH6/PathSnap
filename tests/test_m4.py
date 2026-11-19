import httpx
import pytest
from fastapi.testclient import TestClient
from backend.main import create_app
from backend.pipeline import Prediction
from frontend.app import create_app as create_frontend


@pytest.fixture
def root(tmp_path):
    root = tmp_path / 'student_test'
    for i in range(10):
        id = f'{i:08d}'
        p = root / id
        p.mkdir(parents=True)
        (p / f'{id}.track').write_text('77 12 0\n77.001 12 1\n')
        (p / f'{id}.nodes').write_text('77 12\n77.001 12\n77.002 12\n')
        (p / f'{id}.arcs').write_text('0 1\n1 2\n')
        # Invalid reference file proves preview does not need its contents.
        (p / f'{id}.route').write_text('DO NOT READ')
    return root


def test_preview(root):
    with TestClient(create_app(root)) as c:
        assert c.get('/health').json()['final_ready'] is False
        ids = c.get('/trips').json()['trip_ids']
        assert len(ids) == 10
        for id in ids:
            r = c.post('/reconstruct', json={'trip_id': id})
            assert r.status_code == 200
            d = r.json()
            assert d['mode'] == 'preview' and d['accuracy'] is None
            assert d['trip_id'] == id and len(d['observation_arc_ids']) == 2
            assert d['segments'][0]['coordinates'][0] == [12, 77]
            assert 'reference_route' not in d


def test_errors(root):
    with TestClient(create_app(root)) as c:
        for id in ('../escape', '123', 123):
            assert c.post('/reconstruct', json={'trip_id': id}).status_code == 422
        assert c.post('/reconstruct', json={'trip_id':'99999999'}).status_code == 404
        assert c.post('/reconstruct', json={'trip_id':'00000000','extra':True}).status_code == 422


def test_integration_contract(root):
    calls = []
    def factory():
        calls.append('load')
        def predict(trip):
            assert not hasattr(trip, 'route')
            assert not trip.track.flags.writeable
            calls.append('predict')
            return Prediction([0,1], [0,1])
        return predict
    def evaluate(trip_id, prediction):
        calls.append('evaluate')
        return .5  # synthetic test fixture only, never reported as measured accuracy
    with TestClient(create_app(root, mode='integrated', factory=factory, evaluator=evaluate)) as c:
        data = c.post('/reconstruct', json={'trip_id':'00000000'}).json()
        assert data['accuracy'] == .5 and data['route_continuity'] == 1
        assert calls == ['load','predict','evaluate']


def test_invalid_pipeline_output(root):
    with TestClient(create_app(root, mode='integrated', factory=lambda: lambda t: Prediction([100, 0], [0]))) as c:
        assert c.post('/reconstruct',json={'trip_id':'00000000'}).status_code == 500


def test_reserved_preview_blocked(tmp_path):
    with pytest.raises(ValueError):
        create_app(tmp_path/'evaluator_test', partition='evaluator_test')


def test_frontend_proxy(root):
    with TestClient(create_app(root)) as backend:
        def mock(request):
            response = backend.request(request.method, request.url.path, content=request.content,
                                       headers={'Content-Type':'application/json'})
            return httpx.Response(response.status_code, content=response.content)
        with TestClient(create_frontend('http://backend', httpx.MockTransport(mock))) as client:
            assert 'PathSnap' in client.get('/').text
            assert len(client.get('/api/trips').json()['trip_ids']) == 10
            r = client.post('/api/reconstruct',json={'trip_id':'00000000'})
            assert r.status_code == 200
            assert 'L.polyline' in r.json()['map_html']
            assert 'reference_route' not in r.json()


def test_frontend_unavailable():
    def offline(request):
        raise httpx.ConnectError('offline', request=request)
    with TestClient(create_frontend('http://backend', httpx.MockTransport(offline))) as c:
        assert c.get('/api/trips').status_code == 502
