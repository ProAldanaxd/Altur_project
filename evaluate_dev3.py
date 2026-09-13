"""Compara Dev2 y Dev2+Dev3 en CV de train; no usa val para elegir features."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from app.audio import decode_channels
from dev3.temporal import analyze_turns, extract_model_features
from ml.features import extract_features_from_turns, vad_segments
from ml.ensemble import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    manifest = pd.read_csv(root / "manifest.csv")
    train = manifest[manifest.split == "train"].reset_index(drop=True)
    original, combined = [], []
    for i, row in enumerate(train.itertuples()):
        raw = (root / "audio" / f"{row.anon_id}.wav").read_bytes()
        caller, agent, sr = decode_channels(base64.b64encode(raw).decode())
        turns = vad_segments(caller, sr, 0) + vad_segments(agent, sr, 1)
        duration = len(caller) / sr
        feats = extract_features_from_turns(turns, duration)
        temporal = extract_model_features(analyze_turns(turns, duration))
        original.append(feats)
        combined.append({**feats, **temporal})
        if (i + 1) % 75 == 0:
            print(f"Features train {i+1}/{len(train)}", flush=True)
    datasets = {"dev2_87": pd.DataFrame(original), "dev2_plus_dev3_113": pd.DataFrame(combined)}
    y = (train.label == "synthetic").astype(int).to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(datasets["dev2_87"], y))
    report = {"train_n": len(train), "val_used": False,
              "protocol": "5 fold stratified CV, seed42, same ensemble hyperparameters, paired splits",
              "limitations": "CV a nivel llamada: no hay ID de hablante para agrupar dentro de train; no es evaluación oculta ni calibración de confidence.",
              "models": {}}
    with threadpool_limits(limits=1):
        for name, X in datasets.items():
            oof = np.zeros(len(y))
            scores = []
            for fold, (fit, test) in enumerate(folds):
                model = build_model().fit(X.iloc[fit], y[fit])
                probabilities = model.predict_proba(X.iloc[test])[:, 1]
                assert np.isfinite(probabilities).all()
                oof[test] = probabilities
                scores.append(float(roc_auc_score(y[test], probabilities)))
                print(f"{name} fold{fold+1}: AUC {scores[-1]:.4f}", flush=True)
            report["models"][name] = {"features": X.shape[1], "fold_auc": scores,
                "mean_fold_auc": float(np.mean(scores)), "oof_auc": float(roc_auc_score(y, oof)),
                "oof_accuracy": float(accuracy_score(y, oof >= .5))}
    b, c = report["models"].values()
    report["mean_auc_delta"] = c["mean_fold_auc"] - b["mean_fold_auc"]
    report["promotion_gate"] = "Mean CV AUC gain >=0.005 and OOF accuracy not reduced; independent validation still required."
    report["eligible_for_further_validation"] = bool(report["mean_auc_delta"] >= .005 and c["oof_accuracy"] >= b["oof_accuracy"])
    report["production_model_changed"] = False
    Path("dev3-training-evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
