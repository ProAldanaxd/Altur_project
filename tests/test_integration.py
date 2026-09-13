import asyncio
import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.limits import RequestSizeLimit
from app.main import app
from app.model import Detector


@pytest.mark.parametrize("variant", ["baseline", "alfa"])
def test_registered_models_through_api(monkeypatch, variant):
    monkeypatch.setenv("MODEL_VARIANT", variant)
    signal = np.zeros((24000, 2), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000)
    signal[8000:16000, 0] = tone
    signal[:8000, 1] = tone
    buffer = io.BytesIO()
    sf.write(buffer, signal, 8000, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200 and ready.json()["model"] == variant
        assert client.get("/model").json()["variant"] == variant
        for field in ("audio", "audio_base64"):
            response = client.post("/detect", json={field: encoded})
            assert response.status_code == 200
            assert type(response.json()["is_synthetic"]) is bool
            assert response.headers["X-Model-Variant"] == variant


def test_unregistered_weights_rejected(tmp_path):
    path = tmp_path / "invalid.pkl"
    path.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="no coinciden"):
        Detector(path)


def test_unknown_variant_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_VARIANT", "typo")
    with pytest.raises(RuntimeError, match="MODEL_VARIANT"):
        Detector()


@pytest.mark.parametrize("declared", [False, True])
def test_oversized_body_rejected_before_json(declared):
    async def run():
        reached = []

        async def downstream(scope, receive, send):
            reached.append(True)

        middleware = RequestSizeLimit(downstream, max_bytes=4)
        messages = iter([{"type": "http.request", "body": b"abc", "more_body": True},
                         {"type": "http.request", "body": b"def", "more_body": False}])
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "path": "/detect", "method": "POST",
                 "headers": [(b"content-length", b"6")] if declared else []}
        await middleware(scope, receive, send)
        assert not reached
        assert sent[0]["status"] == 413

    asyncio.run(run())
