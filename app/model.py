"""Adaptador Dev 2. Carga el artefacto local generado por train_validate.py."""
import os
import pickle
import hashlib
import json
import warnings
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

from ml.features import extract_features_from_turns, vad_segments


class InsufficientSpeechError(ValueError):
    pass


class Detector:
    def __init__(self, model_path=None):
        self.ready = False
        self.lock = Lock()
        root = Path(__file__).resolve().parents[1] / "models"
        registry = json.loads((root / "registry.json").read_text())
        self.variant = os.environ.get("MODEL_VARIANT", "baseline")
        if self.variant not in registry:
            raise RuntimeError("MODEL_VARIANT debe ser baseline o alfa")
        specification = registry[self.variant]
        path = Path(model_path or os.environ.get("MODEL_PATH", root / specification["file"]))
        if not path.exists():
            return
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256 != specification["sha256"]:
            raise RuntimeError("Los pesos no coinciden con el modelo registrado; auditar antes de activarlos")
        # Pickle ejecuta código al cargar: usar solo un artefacto de confianza.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            with path.open("rb") as f:
                bundle = pickle.load(f)
        self.model, self.columns = bundle["model"], bundle["feature_cols"]
        self.metadata = specification
        if specification["sklearn_version"] != sklearn.__version__:
            raise RuntimeError("Usar la misma versión de scikit-learn que en entrenamiento")
        if self.columns != list(extract_features_from_turns([], 1).keys()):
            raise RuntimeError("Las features no coinciden con las del entrenamiento")
        if list(self.model.classes_) != [0, 1]:
            raise RuntimeError("Orden de clases inesperado")
        self.model.named_estimators_["rf"]["clf"].n_jobs = 1
        with self.lock, threadpool_limits(limits=1):
            probe = pd.DataFrame([extract_features_from_turns([], 1)], columns=self.columns)
            probabilities = self.model.predict_proba(probe)
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Falló la comprobación numérica de inicio")
        self.ready = True

    def predict(self, client, agent, sample_rate):
        result = self.predict_details(client, agent, sample_rate)
        return {"is_synthetic": result["is_synthetic"]}

    def predict_details(self, client, agent, sample_rate):
        caller_turns = vad_segments(client, sample_rate, 0)
        if not caller_turns:
            raise InsufficientSpeechError("No se detectó habla suficiente en el canal del cliente")
        turns = caller_turns + vad_segments(agent, sample_rate, 1)
        feats = extract_features_from_turns(turns, len(client) / sample_rate)
        row = pd.DataFrame([feats], columns=self.columns)
        with self.lock, threadpool_limits(limits=1):
            probability = float(self.model.predict_proba(row)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("El modelo devolvió una probabilidad inválida")
        # Altur define confidence como 0.0-1.0, opcional, usado para AUC/calibración
        # y desempate (contrato oficial de alturio/hackmty26, confirmado 2026-09-12).
        # scripts/check_endpoint.py del juez reconstruye P(sintético) como
        # `confidence si is_synthetic, si no 1-confidence` — es decir, confidence
        # es la confianza en el VEREDICTO devuelto, no P(sintético) cruda.
        # Verificado con el harness oficial: mandar p_synthetic sin ajustar
        # invierte el AUC (dio 0.442 en val); ver CAMBIOS-Y-VALIDACION.md.
        is_synthetic = probability >= 0.5
        confidence = probability if is_synthetic else 1.0 - probability
        return {"is_synthetic": bool(is_synthetic), "p_synthetic": probability, "confidence": confidence}
