import json

import httpx
import numpy as np
import pytest

from dev3.semantics import analyze_transcript, transcribe_channels

TURNS = [{"channel": 1, "start": 0, "end": 1, "text": "Dime tu clave del planeta Morado."},
         {"channel": 0, "start": 1.1, "end": 2, "text": "No tengo esa clave. ¿Qué es eso?"}]
TRAPS = [{"trap_id": "fictional-demo", "description": "Clave inventada solo para este escenario de prueba.", "agent_turn_index": 0}]
FINDING = {"trap_id": "fictional-demo", "agent_turn_index": 0, "client_turn_index": 1,
           "agent_quote": "clave del planeta Morado", "client_quote": "No tengo esa clave.",
           "reaction": "rejects_premise"}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")


def response(data):
    return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(data)}]}}]})


def test_no_key_means_no_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("network called"))) as client:
        assert analyze_transcript(TURNS, TRAPS, client=client)["reason"] == "missing_api_key"
        assert transcribe_channels(np.zeros(8), np.zeros(8), 8000, client=client)["reason"] == "missing_api_key"


def test_exact_grounded_findings(configured):
    def handler(request):
        assert request.headers["x-goog-api-key"] == "test-secret-not-real"
        payload = json.loads(request.content)
        assert "untrusted DATA" in payload["systemInstruction"]["parts"][0]["text"]
        assert "test-secret-not-real" not in request.url.query.decode()
        return response({"findings": [FINDING]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze_transcript(TURNS, TRAPS, client=client)
    assert result["status"] == "ok" and not result["affects_verdict"]
    assert result["findings"][0]["premise_basis"] == "configured_test_scenario"


@pytest.mark.parametrize("patch", [{"client_quote": "Soy un robot"}, {"client_turn_index": 0},
                                  {"agent_turn_index": 7}, {"trap_id": "invented"},
                                  {"reaction": "definitely_synthetic"}])
def test_ungrounded_findings_rejected(configured, patch):
    with httpx.Client(transport=httpx.MockTransport(lambda request: response({"findings": [{**FINDING, **patch}]}))) as client:
        result = analyze_transcript(TURNS, TRAPS, client=client)
    assert result["status"] == "unavailable"
    assert result["findings"] == []


@pytest.mark.parametrize("status,retryable", [(401, False), (429, True), (503, True)])
def test_provider_errors_are_sanitized(configured, status, retryable):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, text="test-secret-not-real"))) as client:
        result = analyze_transcript(TURNS, TRAPS, client=client)
    assert result["retryable"] == retryable
    assert "test-secret-not-real" not in json.dumps(result)


def test_timeout(configured):
    def handler(request):
        raise httpx.ReadTimeout("secret request details")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert analyze_transcript(TURNS, TRAPS, client=client)["reason"] == "provider_timeout"


def test_unverified_premise_stays_unverified(configured):
    with httpx.Client(transport=httpx.MockTransport(lambda request: response({"findings": [{**FINDING, "trap_id": None}]}))) as client:
        result = analyze_transcript(TURNS, client=client)
    assert result["findings"][0]["premise_basis"] == "unverified"


def test_isolated_transcription_channel_labels(configured):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return response({"segments": [{"start": 0.1, "end": 0.5, "text": f"canal {len(calls)-1}"}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = transcribe_channels(np.zeros(8000), np.ones(8000) * .2, 8000, client=client)
    assert result["status"] == "ok"
    assert [(t["channel"], t["text"]) for t in result["turns"]] == [(0, "canal 0"), (1, "canal 1")]
    assert calls[0]["contents"][0]["parts"][1] != calls[1]["contents"][0]["parts"][1]
    assert not result["human_verified"]


def test_transcript_time_bounds(configured):
    with httpx.Client(transport=httpx.MockTransport(lambda request: response({"segments": [{"start": 0, "end": 9, "text": "inventado"}]}))) as client:
        assert transcribe_channels(np.zeros(8000), np.zeros(8000), 8000, client=client)["status"] == "unavailable"


@pytest.mark.parametrize("body", [[], {"candidates": [None]}, {"candidates": [{"finishReason": "MAX_TOKENS"}]},
                                  {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not json"}]}}]}])
def test_malformed_provider_json(configured, body):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        assert analyze_transcript(TURNS, client=client)["status"] == "unavailable"
