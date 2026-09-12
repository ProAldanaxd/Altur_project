"""Reconstruye Dev 2 con train y evalúa val; usa el mismo VAD que la API."""
import argparse
import base64
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from app.audio import decode_channels
from ml.ensemble import build_model
from ml.features import extract_features_from_turns, vad_segments


def metrics(y, probabilities):
    return {"n": len(y), "accuracy": accuracy_score(y, probabilities >= 0.5),
            "balanced_accuracy": balanced_accuracy_score(y, probabilities >= 0.5),
            "roc_auc": roc_auc_score(y, probabilities), "brier": brier_score_loss(y, probabilities),
            "confusion_matrix_rows_human_synthetic": confusion_matrix(y, probabilities >= 0.5).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="models/model.pkl")
    args = parser.parse_args()
    root = Path(args.data_root)
    manifest = pd.read_csv(root / "manifest.csv")
    assert manifest.anon_id.is_unique
    assert set(manifest.split) == {"train", "val"}
    assert set(manifest.label) == {"human", "synthetic"}
    official_rows, audio_rows, extraction_ms = [], [], []
    for i, row in enumerate(manifest.itertuples()):
        turns = json.loads((root / "turns" / f"{row.anon_id}.json").read_text())["turns"]
        official_rows.append(extract_features_from_turns(turns, row.duration_s))
        raw = (root / "audio" / f"{row.anon_id}.wav").read_bytes()
        start = time.perf_counter()
        caller, agent, sr = decode_channels(base64.b64encode(raw).decode())
        detected = vad_segments(caller, sr, 0) + vad_segments(agent, sr, 1)
        audio_rows.append(extract_features_from_turns(detected, len(caller) / sr))
        extraction_ms.append((time.perf_counter() - start) * 1000)
        if (i + 1) % 50 == 0:
            print(f"Features de audio: {i + 1}/{len(manifest)}", flush=True)
    X_official, X_audio = pd.DataFrame(official_rows), pd.DataFrame(audio_rows)
    assert list(X_audio.columns) == list(X_official.columns)
    assert np.isfinite(X_audio.to_numpy()).all()
    train, val = manifest.split == "train", manifest.split == "val"
    y = (manifest.label == "synthetic").astype(int)
    print("Entrenando baseline de turnos oficiales...", flush=True)
    with threadpool_limits(limits=1):
        baseline = build_model().fit(X_official[train], y[train])
        reference = metrics(y[val], baseline.predict_proba(X_official[val])[:, 1])
        mismatch = metrics(y[val], baseline.predict_proba(X_audio[val])[:, 1])
        print("Baseline:", json.dumps({"official": reference, "audio": mismatch}), flush=True)
        print("Entrenando modelo con VAD de audio en train...", flush=True)
        model = build_model().fit(X_audio[train], y[train])
        end_to_end = metrics(y[val], model.predict_proba(X_audio[val])[:, 1])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"train_n": int(train.sum()), "val_n": int(val.sum()),
              "feature_count": len(X_audio.columns), "sklearn_version": sklearn.__version__,
              "official_turns_validation": reference,
              "official_trained_model_on_audio_validation": mismatch,
              "audio_trained_model_on_audio_validation": end_to_end,
              "feature_extraction_mean_ms": float(np.mean(extraction_ms)),
              "feature_extraction_p95_ms": float(np.percentile(extraction_ms, 95)),
              "training_source": "audio_vad", "trained_split": "train",
              "notes": "Mismo ensemble y VAD de Dev 2; sin ajuste en val. No es evaluación oculta ni medida de red."}
    with output.open("wb") as f:
        pickle.dump({"model": model, "feature_cols": list(X_audio.columns), "metadata": report}, f)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
