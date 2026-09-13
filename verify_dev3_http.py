"""Verifica Dev3 con todos los WAV de val; solo informes agregados."""
import argparse
import base64
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd


def check_invariants(data):
    assert data["affects_detect"] is False
    assert data["semantic"]["status"] == "not_requested"
    summary = data["temporal"]["summary"]
    duration = data["temporal"]["duration_s"]
    assert abs(summary["active_speech_s"] + summary["dead_air_s"] - duration) < 1e-6
    assert abs(summary["client_speech_s"] + summary["agent_speech_s"] - summary["overlap_s"] - summary["active_speech_s"]) < 1e-6
    assert all(-1e-9 <= summary[k] <= 1 + 1e-9 for k in ("overlap_ratio", "dead_air_ratio", "client_speech_ratio", "agent_speech_ratio"))
    pairs = data["temporal"]["events"]["response_candidates"]
    assert len({p["client_turn_index"] for p in pairs}) == len(pairs)
    assert np.isfinite(list(data["features"].values())).all()
    assert len(data["features"]) == 26
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8025")
    parser.add_argument("--report", default="verification-dev3-http-new.json",
                        help="No usar verification-dev3-http.json: es evidencia histórica y no debe sobrescribirse")
    args = parser.parse_args()
    root = Path(args.data_root)
    manifest = pd.read_csv(root / "manifest.csv")
    val = manifest[manifest.split == "val"]
    times, overlaps, negative_pairs, errors = [], [], 0, []
    samples = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        status = client.get("/conversation/status").json()
        assert status["temporal"] == "ready"
        for row in val.itertuples():
            payload = {"audio": base64.b64encode((root / "audio" / f"{row.anon_id}.wav").read_bytes()).decode()}
            if len(samples) < 4:
                samples.append(payload)
            t0 = time.perf_counter()
            response = client.post("/conversation/analyze", json=payload)
            times.append((time.perf_counter() - t0) * 1000)
            if response.status_code != 200:
                errors.append(response.status_code)
                continue
            summary = check_invariants(response.json())
            overlaps.append(summary["overlap_s"])
            negative_pairs += sum(p["latency_s"] < 0 for p in response.json()["temporal"]["events"]["response_candidates"])
        def concurrent(payload):
            r = client.post("/conversation/analyze", json=payload)
            if r.status_code == 200:
                check_invariants(r.json())
            return r.status_code
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(concurrent, samples * 2))
        # No contactar a Gemini si el servidor tiene credenciales configuradas: solo
        # verificamos aquí el camino offline (missing_api_key). Si el operador configuró
        # el proveedor, esta prueba se salta en vez de disparar una llamada real y fallar.
        semantic_config = status["semantic"]
        if semantic_config.get("status") == "unavailable" and semantic_config.get("reason") == "missing_api_key":
            unavailable = client.post("/conversation/analyze", json={**samples[0], "include_semantics": True}).json()["semantic"]
            assert unavailable["reason"] == "missing_api_key"
        else:
            unavailable = {"skipped": True, "reason": "gemini_appears_configured_on_server; verificacion offline omitida para no contactar al proveedor",
                          "status_seen": semantic_config}
    report = {"validation_calls": len(val), "passed_invariants": len(overlaps), "http_errors": errors,
              "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)),
              "signed_negative_response_pairs": negative_pairs,
              "mean_overlap_s": float(np.mean(overlaps)), "concurrent_statuses": statuses,
              "semantic_live_status": unavailable, "scope": "WAV de val oficial, localhost; invariantes matemáticas, no exactitud de VAD contra anotación humana"}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors or any(status != 200 for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
