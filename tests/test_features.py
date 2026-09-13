"""Documents and now also fixes (opt-in) a known limitation of the 87-feature
extractor (ml/features.py).

`latency_pairing="legacy"` (the default): only pairs a caller turn with an
agent turn that has ALREADY ENDED, so `latency_frac_negative` is wired to
always be 0.0 by construction. This is the exact definition models/model.pkl
and models/dev2Alfa.pkl were trained against. It must stay the default so
`app/model.py`'s Detector (which calls extract_features_from_turns without
this keyword) is completely unaffected — changing the formula under an
already-deployed pickle without retraining would silently feed it a
distribution shift it never saw during training.

`latency_pairing="signed_v2"`: the corrected pairing (same definition
conversation/temporal.py already uses and tests — see
tests/test_temporal.py::test_response_candidates_are_signed_unique_and_belong_to_latest_agent),
which does produce negative latencies on real overlap. Not wired into any
deployed model yet: it exists so train_validate.py can produce the NEXT
generation of weights the moment the official dataset is available (see
REVISION-TECNICA.md for the exact retraining procedure), without touching
today's production behavior at all.
"""
from ml.features import extract_features_from_turns

OVERLAP_TURNS = [
    {"channel": 1, "start": 0.0, "end": 5.0},   # agent speaking 0-5s
    {"channel": 0, "start": 2.0, "end": 3.0},   # caller starts mid-agent-turn (real overlap)
    {"channel": 1, "start": 6.0, "end": 7.0},
    {"channel": 0, "start": 6.5, "end": 8.0},   # caller answers fast (normal case)
]


def test_default_call_is_legacy_and_latency_frac_negative_is_structurally_always_zero():
    # No latency_pairing keyword passed, exactly like app/model.py's Detector calls it.
    features = extract_features_from_turns(OVERLAP_TURNS, duration_s=10)
    assert features["latency_frac_negative"] == 0.0
    assert features["latency_n"] > 0


def test_explicit_legacy_matches_default():
    default = extract_features_from_turns(OVERLAP_TURNS, duration_s=10)
    explicit = extract_features_from_turns(OVERLAP_TURNS, duration_s=10, latency_pairing="legacy")
    assert default == explicit


def test_signed_v2_produces_negative_latency_on_real_overlap():
    features = extract_features_from_turns(OVERLAP_TURNS, duration_s=10, latency_pairing="signed_v2")
    # Agent turn (0, 5) pairs with caller turn (2, 3): latency = 2 - 5 = -3 (real overlap, signed negative).
    # Agent turn (6, 7) pairs with caller turn (6.5, 8): latency = 6.5 - 7 = -0.5.
    assert features["latency_frac_negative"] == 1.0
    assert features["latency_n"] == 2
    assert features["latency_min"] == -3.0


def test_signed_v2_keeps_the_same_87_column_names_as_legacy():
    # A future retrain must not change the schema, only the values of the
    # existing latency_* columns — this is what keeps Detector's strict
    # column-equality check meaningful after retraining.
    legacy_keys = list(extract_features_from_turns([], 1).keys())
    v2_keys = list(extract_features_from_turns([], 1, latency_pairing="signed_v2").keys())
    assert legacy_keys == v2_keys


def test_invalid_latency_pairing_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        extract_features_from_turns(OVERLAP_TURNS, duration_s=10, latency_pairing="nonsense")
