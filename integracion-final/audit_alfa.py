"""Audita el artefacto Alfa frente al modelo con holdout conservado."""
import argparse
import base64
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.audio import decode_channels
from ml.features import extract_features_from_turns, vad_segments
from train_validate import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    manifest = pd.read_csv(root / "manifest.csv")
    with Path("models/dev2Alfa.pkl").open("rb") as f:
        alfa = pickle.load(f)
    with Path("models/model.pkl").open("rb") as f:
        baseline = pickle.load(f)
    assert alfa["feature_cols"] == baseline["feature_cols"]
    official, audio = [], []
    for i, row in enumerate(manifest.itertuples()):
        turns = json.loads((root / "turns" / f"{row.anon_id}.json").read_text())["turns"]
        official.append(extract_features_from_turns(turns, row.duration_s))
        raw = (root / "audio" / f"{row.anon_id}.wav").read_bytes()
        caller, agent, sr = decode_channels(base64.b64encode(raw).decode())
        turns = vad_segments(caller, sr, 0) + vad_segments(agent, sr, 1)
        audio.append(extract_features_from_turns(turns, len(caller) / sr))
        if (i + 1) % 100 == 0:
            print(f"Auditadas features: {i + 1}/{len(manifest)}", flush=True)
    Xo = pd.DataFrame(official, columns=alfa["feature_cols"])
    Xa = pd.DataFrame(audio, columns=alfa["feature_cols"])
    scaler = alfa["model"].named_estimators_["logreg"]["scale"]
    report = {
        "alfa_sha256": hashlib.sha256(Path("models/dev2Alfa.pkl").read_bytes()).hexdigest(),
        "feature_count": len(alfa["feature_cols"]),
        "scaler_training_samples": int(scaler.n_samples_seen_),
        "scaler_matches_all_official_turns": bool(np.allclose(scaler.mean_, Xo.mean().values, rtol=1e-10, atol=1e-10)),
        "scaler_matches_all_audio_vad": bool(np.allclose(scaler.mean_, Xa.mean().values, rtol=1e-10, atol=1e-10)),
        "has_training_metadata": "metadata" in alfa,
    }
    val = manifest.split == "val"
    y = (manifest.label == "synthetic").astype(int)
    # Solo cambia concurrencia interna, no pesos ni hiperparámetros predictivos.
    alfa["model"].named_estimators_["rf"]["clf"].n_jobs = 1
    with threadpool_limits(limits=1):
        for name, bundle, X in [("alfa_official_turns_diagnostic", alfa, Xo),
                                ("alfa_audio_diagnostic", alfa, Xa),
                                ("baseline_audio_holdout", baseline, Xa)]:
            t0 = time.perf_counter()
            probabilities = bundle["model"].predict_proba(X[val])[:, 1]
            assert np.isfinite(probabilities).all()
            report[name] = metrics(y[val], probabilities)
            report[name]["batch_prediction_ms"] = (time.perf_counter() - t0) * 1000
    report["interpretation"] = "Si Alfa coincide con los 353 turnos, val no es holdout para Alfa. Comparación diagnóstica, no selección de modelo por test independiente."
    Path("audit-alfa.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
