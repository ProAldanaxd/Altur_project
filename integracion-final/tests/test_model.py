import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from app.model import Detector, InsufficientSpeechError
from ml.features import extract_features_from_turns


def test_real_model_rejects_silence():
    detector = Detector()
    assert detector.ready
    with pytest.raises(InsufficientSpeechError):
        detector.predict(np.zeros(8000, dtype=np.float32), np.zeros(8000, dtype=np.float32), 8000)


def test_model_numeric_consistency():
    detector = Detector()
    assert detector.ready
    segments = [{"channel": 1, "start": 0., "end": 1.},
                {"channel": 0, "start": 1.5, "end": 3.},
                {"channel": 1, "start": 2.5, "end": 4.},
                {"channel": 0, "start": 4.6, "end": 6.}]
    row = pd.DataFrame([extract_features_from_turns(segments, 7)], columns=detector.columns)
    pipeline = detector.model.named_estimators_["logreg"]
    scaled = pipeline[:-1].transform(row)
    clf = pipeline[-1]
    assert np.isfinite(scaled).all() and np.isfinite(clf.coef_).all()
    # Compara la multiplicación BLAS con suma explícita independiente.
    reference = expit(np.sum(scaled * clf.coef_, axis=1) + clf.intercept_)
    np.testing.assert_allclose(pipeline.predict_proba(row)[:, 1], reference, atol=1e-10)
    probabilities = detector.model.predict_proba(row)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
