"""Latency benchmark for /detect using synthetic WAVs (no official dataset available).

Does NOT measure accuracy. Measures HTTP round-trip latency, warm-up effect,
p50/p95/p99, throughput and error behavior for varied clip shapes and
concurrency levels, on this machine, this run, this commit's code.
"""
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

WAV_DIR = Path(__file__).parent / "synth_wav"


def load_payloads():
    payloads = {}
    for wav in sorted(WAV_DIR.glob("*.wav")):
        payloads[wav.name] = {"audio": base64.b64encode(wav.read_bytes()).decode()}
    return payloads


def percentile_report(latencies_ms):
    arr = np.asarray(latencies_ms)
    return {"n": len(arr), "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)) if len(arr) >= 20 else None,
            "max_ms": float(arr.max()), "min_ms": float(arr.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8025")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--report", default="work/bench-detect-report.json")
    args = parser.parse_args()

    payloads = load_payloads()
    report = {"note": "SINTÉTICO: WAV generados por seno+ruido, no habla real. Mide solo latencia/robustez, no exactitud.",
               "per_clip": {}}

    with httpx.Client(base_url=args.url, timeout=60) as client:
        # Warm-up: first request after server start is typically slower (imports, caches).
        warm_payload = next(iter(payloads.values()))
        t0 = time.perf_counter()
        client.post("/detect", json=warm_payload)
        cold_start_ms = (time.perf_counter() - t0) * 1000

        for name, payload in payloads.items():
            latencies, statuses = [], []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                r = client.post("/detect", json=payload)
                latencies.append((time.perf_counter() - t0) * 1000)
                statuses.append(r.status_code)
            entry = percentile_report(latencies)
            entry["statuses"] = sorted(set(statuses))
            entry["last_response"] = r.json() if r.status_code == 200 else r.json()
            report["per_clip"][name] = entry

        # Concurrency sweep at fixed clip (the "normal" one), matching historical protocol shape.
        normal_payload = payloads["normal_30s.wav"]

        def one_call(_):
            t0 = time.perf_counter()
            r = client.post("/detect", json=normal_payload)
            return {"status": r.status_code, "ms": (time.perf_counter() - t0) * 1000}

        concurrency_results = {}
        for workers in (1, 2, 4, 8):
            n_requests = max(workers * 4, 8)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(one_call, range(n_requests)))
            wall_s = time.perf_counter() - t0
            ms_values = [r["ms"] for r in results]
            concurrency_results[str(workers)] = {
                "n_requests": n_requests, "wall_s": wall_s,
                "throughput_rps": n_requests / wall_s,
                "statuses": sorted(set(r["status"] for r in results)),
                **percentile_report(ms_values),
            }
        report["cold_start_ms"] = cold_start_ms
        report["concurrency"] = concurrency_results

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
