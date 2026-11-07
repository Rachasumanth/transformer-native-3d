import os

from fastapi.testclient import TestClient

from t3d.service import create_app


class FakeRuntime:
    device = 'cpu'
    request_count = 0
    error_count = 0

    def __init__(self):
        self.checkpoint_paths = {'occupancy': 'occ.pt', 'geometry': 'geom.pt', 'shape_vae': 'vae.pt'}
        self.calls = []

    @property
    def checkpoints(self):
        return self.checkpoint_paths

    def readiness(self):
        return {'ready': True, 'device': self.device, 'resolution': 64, 'material': False,
                'checkpoints': self.checkpoints, 'requests': 0, 'errors': 0}

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return b'glTF\x02\x00\x00\x00fake'


def test_health_ready_and_generate():
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime))
    assert client.get('/healthz').status_code == 200
    ready = client.get('/readyz')
    assert ready.status_code == 200 and ready.json()['ready'] is True
    response = client.post('/v1/generate', json={'prompt': 'a bottle', 'steps': 2, 'seed': 7})
    assert response.status_code == 200
    assert response.headers['content-type'] == 'model/gltf-binary'
    assert response.headers['x-t3d-seed'] == '7'
    assert 'a-bottle.glb' in response.headers['content-disposition']
    assert runtime.calls and runtime.calls[0][0] == 'a bottle'


def test_validation_and_metrics():
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime))
    assert client.post('/v1/generate', json={'prompt': ''}).status_code == 422
    assert client.post('/v1/generate', json={'prompt': 'x', 'steps': 0}).status_code == 422
    assert client.post('/v1/generate', json={'prompt': 'x', 'format': 'obj'}).status_code == 422
    client.post('/v1/generate', json={'prompt': 'a cup'})
    metrics = client.get('/metrics').text
    assert 't3d_generation_requests_total 1' in metrics


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv('T3D_API_KEY', 'secret')
    client = TestClient(create_app(FakeRuntime()))
    assert client.post('/v1/generate', json={'prompt': 'a cup'}).status_code == 401
    assert client.post('/v1/generate', json={'prompt': 'a cup'}, headers={'X-API-Key': 'secret'}).status_code == 200


def test_checkpoints_endpoint():
    client = TestClient(create_app(FakeRuntime()))
    response = client.get('/v1/checkpoints')
    assert response.status_code == 200
    assert response.json()['geometry'] == 'geom.pt'


def test_unavailable_runtime_is_not_ready(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('checkpoints missing')

    monkeypatch.setattr('t3d.service.T3DRuntime', fail)
    client = TestClient(create_app(None))
    response = client.get('/readyz')
    assert response.status_code == 503
    assert 'checkpoints missing' in response.json()['detail']
