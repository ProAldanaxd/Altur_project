"""Evalúa todas las llamadas val a través de HTTP; errores cuentan como fallos."""
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8025")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report", default="verification-real-http.json")
    args = parser.parse_args()
    root = Path(args.data_root)
    manifest = pd.read_csv(root / "manifest.csv")
    val = manifest[manifest.split == "val"]
    correct, errors, latencies = 0, [], []
    confusion = [[0, 0], [0, 0]]
    samples = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        identity = client.get("/model").json()
        for row in val.itertuples():
            payload = {"audio": base64.b64encode((root / "audio" / f"{row.anon_id}.wav").read_bytes()).decode()}
            if len(samples) < 4:
                samples.append(payload)
            t0 = time.perf_counter()
            response = client.post("/detect", json=payload)
            latencies.append((time.perf_counter() - t0) * 1000)
            if response.status_code != 200:
                errors.append({"status": response.status_code, "detail": response.text[:200]})
                continue
            result = response.json()
            assert type(result.get("is_synthetic")) is bool and "error" not in result
            truth, pred = int(row.label == "synthetic"), int(result["is_synthetic"])
            confusion[truth][pred] += 1
            correct += truth == pred
        def concurrent(payload):
            t0 = time.perf_counter()
            r = client.post("/detect", json=payload)
            return {"status": r.status_code, "ms": (time.perf_counter() - t0) * 1000}
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_results = list(pool.map(concurrent, samples * 2))
        invalid = [{}, {"audio": "!!!"}, {"audio": "aG9sYQ=="}, {"audio": samples[0]["audio"], "audio_base64": "different"}]
        malformed_statuses = [client.post("/detect", json=p).status_code for p in invalid]
        assert malformed_statuses == [422] * len(invalid)
    report = {"validation_calls": len(val), "correct": correct,
              "model_identity": identity,
              "independent_holdout": identity["holdout_preserved"],
              "accuracy_including_errors_as_failures": correct / len(val),
              "http_errors": errors, "confusion_matrix_rows_human_synthetic": confusion,
              "mean_ms": float(np.mean(latencies)), "p50_ms": float(np.percentile(latencies, 50)),
              "p95_ms": float(np.percentile(latencies, 95)), "max_ms": float(max(latencies)),
              "concurrent_requests": concurrent_results, "invalid_input_statuses": malformed_statuses,
              "scope": "HTTP local, val oficial completo, 4 hilos en prueba concurrente; no benchmark oculto ni nube"}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors or any(r["status"] != 200 for r in concurrent_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
