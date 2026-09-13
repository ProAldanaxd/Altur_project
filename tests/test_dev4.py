import io
import json
import sqlite3
import base64
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from ops.audit import AuditStore
from ops.postgres import sync_batch
from ops.voice import generate_alert


def event(request_id="test-id"):
    return {"request_id": request_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "status_code": 200, "is_synthetic": True, "p_synthetic": .83,
            "latency_ms": 42.1, "duration_s": 150., "model_variant": "baseline", "model_sha256": "a" * 64}


def test_persistence_restart_and_metadata_allowlist(tmp_path):
    path = tmp_path / "audit.db"
    store = AuditStore(path)
    assert store.enqueue({**event(), "audio": "sensitive audio", "transcript": "secret"})
    assert store.wait_idle()
    store.close()
    reopened = AuditStore(path)
    row = reopened.recent()[0]
    assert row["request_id"] == "test-id" and row["p_synthetic"] == .83
    assert "audio" not in row and "transcript" not in row
    assert reopened.stats()["persisted"]["total"] == 1
    reopened.close()
    assert not reopened.enqueue(event("later"))
    assert reopened.counts["dropped"] == 1


def test_api_logs_errors_without_a_verdict_and_protects_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "operator-test")
    with TestClient(app) as client:
        response = client.post("/detect", json={"audio": "!!!"})
        assert response.status_code == 422
        assert app.state.audit.wait_idle()
        assert client.get("/audit/calls").status_code == 401
        row = client.get("/audit/calls", headers={"Authorization": "Bearer operator-test"}).json()["calls"][0]
        assert row["request_id"] == response.headers["X-Request-ID"]
        assert row["is_synthetic"] is None and row["p_synthetic"] is None


def test_classification_contract_and_internal_probability():
    signal = np.zeros((24000, 2), dtype=np.float32)
    signal[8000:16000, 0] = .3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000)
    signal[:8000, 1] = .3 * np.sin(2 * np.pi * 300 * np.arange(8000) / 8000)
    buf = io.BytesIO()
    sf.write(buf, signal, 8000, format="WAV", subtype="PCM_16")
    with TestClient(app) as client:
        r = client.post("/detect", json={"audio": base64.b64encode(buf.getvalue()).decode()})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"is_synthetic", "confidence"}
        assert 0 <= body["confidence"] <= 1
        assert body["confidence"] >= 0.5  # confianza en el veredicto propio, no P(sintético) cruda
        assert app.state.audit.wait_idle()
        row = app.state.audit.recent()[0]
        assert bool(row["is_synthetic"]) == body["is_synthetic"]
        # p_synthetic (auditoría interna) es la probabilidad cruda; confidence (contrato
        # público) es la confianza en el veredicto: coinciden solo si el veredicto es "sintético".
        expected_confidence = row["p_synthetic"] if row["is_synthetic"] else 1.0 - row["p_synthetic"]
        assert body["confidence"] == pytest.approx(expected_confidence)


def test_db_startup_failure_does_not_disable_model(monkeypatch):
    def unavailable():
        raise sqlite3.OperationalError("database unavailable")
    monkeypatch.setattr("app.main.AuditStore", unavailable)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/audit/stats").status_code == 503


def test_voice_unconfigured_never_calls_provider(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("unexpected network"))) as client:
        assert generate_alert(client=client)["reason"] == "missing_api_key"


@pytest.mark.parametrize("code", [200, 401, 429, 503])
def test_voice_provider_contract_and_errors(monkeypatch, code):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-secret")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-test")
    def handler(request):
        assert request.headers["xi-api-key"] == "test-secret"
        assert "verificación adicional" in json.loads(request.content)["text"]
        return httpx.Response(code, headers={"Content-Type": "audio/mpeg"}, content=b"ID3" + b"x" * 64)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        r = generate_alert(client=client)
    assert r["status"] == ("ok" if code == 200 else "unavailable")
    if code != 200:
        assert "test-secret" not in json.dumps(r)


def test_voice_alert_text_matches_verdict(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-secret")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-test")
    sent_texts = []
    def handler(request):
        sent_texts.append(json.loads(request.content)["text"])
        return httpx.Response(200, headers={"Content-Type": "audio/mpeg"}, content=b"ID3" + b"x" * 64)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generate_alert(client=client, is_synthetic=True)
        generate_alert(client=client, is_synthetic=False)
    assert "sintética" in sent_texts[0] and "humana" not in sent_texts[0]
    assert "humana" in sent_texts[1] and "sintética" not in sent_texts[1]
    assert sent_texts[0] != sent_texts[1]


def test_postgres_idempotent_export_and_failure_keeps_pending(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test-only")
    store = AuditStore(tmp_path / "test.db")
    store.enqueue(event());assert store.wait_idle()
    statements = []
    class Remote:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, query, params=None): statements.append((query, params))
    def fail(*args, **kwargs): raise RuntimeError("password must not appear")
    assert sync_batch(store.path, connect=fail)["reason"] == "postgres_sync_failed"
    assert store.recent()[0]["exported"] == 0
    assert sync_batch(store.path, connect=lambda *a, **k: Remote())["exported"] == 1
    assert "ON CONFLICT" in statements[-1][0]
    assert sync_batch(store.path, connect=lambda *a, **k: Remote())["exported"] == 0
    store.close()


def test_postgres_sync_on_uninitialized_db_reports_status_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test-only")
    empty_db = tmp_path / "never-started.sqlite3"
    empty_db.touch()
    result = sync_batch(empty_db, connect=lambda *a, **k: pytest.fail("must not reach the provider"))
    assert result == {"status": "unavailable", "reason": "audit_db_not_initialized"}


def test_audit_write_counter_distinguishes_duplicate_request_id(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    assert store.enqueue(event("same-id"))
    assert store.wait_idle()
    assert store.enqueue(event("same-id"))
    assert store.wait_idle()
    assert store.counts["written"] == 1
    assert store.counts["duplicate_ignored"] == 1
    assert store.stats()["persisted"]["total"] == 1
    store.close()
