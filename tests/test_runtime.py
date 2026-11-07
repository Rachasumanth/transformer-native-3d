import threading

import pytest

from t3d.runtime import T3DRuntime, select_device


def test_select_device_cpu_and_auto():
    assert select_device('cpu') == 'cpu'


def test_generate_rejects_invalid_requests_before_model_use():
    runtime = object.__new__(T3DRuntime)
    runtime.lock = threading.RLock()
    runtime.request_count = 0
    runtime.error_count = 0
    runtime.material_flow = None
    runtime.material_vae = None
    with pytest.raises(ValueError):
        runtime.generate('')
    with pytest.raises(ValueError):
        runtime.generate('x', steps=0)
    with pytest.raises(ValueError):
        runtime.generate('x', format='obj')
    with pytest.raises(ValueError):
        runtime.generate('x', guidance=11)
    with pytest.raises(ValueError):
        runtime.generate('x', method='rk4')
    assert runtime.request_count == 0
    assert runtime.error_count == 0
