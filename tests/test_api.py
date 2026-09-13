import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def payload(channels=2, rate=8000, frames=80, subtype="PCM_16"):
    data = np.zeros((frames, channels), dtype=np.float32)
    data[:, 0] = 0.25
    if channels == 2:
        data[:, 1] = -0.5
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV", subtype=subtype)
    return {"audio_base64": base64.b64encode(buffer.getvalue()).decode()}


def test_unintegrated_model_is_not_a_prediction(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pkl"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/detect", json=payload()).status_code == 503


@pytest.mark.parametrize("body", [
    {}, {"audio_base64": "!bad!"},
    {"audio_base64": base64.b64encode(b"not audio").decode()},
    payload(channels=1), payload(rate=16000), payload(frames=0),
    payload(subtype="FLOAT"),
])
def test_invalid_input(body):
    with TestClient(app) as client:
        assert client.post("/detect", json=body).status_code == 422


def test_channel_order_and_prediction_contract():
    class TestDetector:
        ready = True

        def predict(self, customer, agent, sample_rate):
            assert sample_rate == 8000
            np.testing.assert_allclose(customer, 0.25)
            np.testing.assert_allclose(agent, -0.5)
            return {"is_synthetic": True, "confidence": 0.9}

    with TestClient(app) as client:
        app.state.detector = TestDetector()
        result = client.post("/detect", json=payload())
        assert result.status_code == 200
        assert result.json() == {"is_synthetic": True, "confidence": 0.9}
        assert float(result.headers["X-Process-Time-Ms"]) >= 0


def test_full_call_and_optional_confidence():
    # El manifest oficial incluye una llamada de 273 segundos.
    class TestDetector:
        ready = True

        def predict(self, customer, agent, sample_rate):
            assert len(customer) == len(agent) == 273 * sample_rate
            return {"is_synthetic": False}

    with TestClient(app) as client:
        app.state.detector = TestDetector()
        result = client.post("/detect", json=payload(frames=273 * 8000))
        assert result.status_code == 200
        assert result.json() == {"is_synthetic": False}


def test_both_input_names_and_conflicts():
    class TestDetector:
        ready = True

        def predict(self, customer, agent, sample_rate):
            return {"is_synthetic": False}

    with TestClient(app) as client:
        app.state.detector = TestDetector()
        encoded = payload()["audio_base64"]
        for name in ("audio", "audio_base64"):
            assert client.post("/detect", json={name: encoded}).status_code == 200
        assert client.post("/detect", json={"audio": encoded, "audio_base64": "other"}).status_code == 422
