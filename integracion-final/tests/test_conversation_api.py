import base64
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def wav():
    data = np.zeros((40000, 2), dtype=np.float32)
    tone = .3 * np.sin(2 * np.pi * 300 * np.arange(8000) / 8000)
    data[8000:16000, 1] = tone
    data[12000:20000, 0] = tone
    buffer = io.BytesIO()
    sf.write(buffer, data, 8000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode()


def test_temporal_without_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with TestClient(app) as client:
        before = client.post("/detect", json={"audio": wav()}).json()
        result = client.post("/conversation/analyze", json={"audio": wav()})
        assert result.status_code == 200
        data = result.json()
        assert data["semantic"]["status"] == "not_requested"
        assert data["temporal"]["summary"]["overlap_s"] == .5
        assert data["temporal"]["summary"]["response_latency"]["mean_s"] == -.5
        assert len(data["features"]) == 26
        assert not data["affects_detect"]
        assert client.post("/detect", json={"audio": wav()}).json() == before
        pending = client.post("/conversation/analyze", json={"audio": wav(), "include_semantics": True})
        assert pending.json()["semantic"]["reason"] == "missing_api_key"
        assert client.get("/conversation/status").json()["temporal"] == "ready"


def test_bad_transcript_exceeds_audio():
    with TestClient(app) as client:
        result = client.post("/conversation/analyze", json={"audio": wav(), "include_semantics": True,
            "transcript": [{"channel": 0, "start": 0, "end": 10, "text": "hola"}]})
        assert result.status_code == 422


def test_semantic_input_validation():
    with TestClient(app) as client:
        assert client.post("/conversation/semantic", json={"turns": []}).status_code == 422
        assert client.post("/conversation/analyze", json={"audio": "!!!"}).status_code == 422
