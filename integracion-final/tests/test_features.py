"""Documents a known limitation of the legacy 87-feature extractor (ml/features.py).

latency_frac_negative is wired to always be 0.0: the pairing loop only keeps
agent turns that already ended before the caller starts (`a["end"] <= ct["start"]`),
so `ct["start"] - max(prev)` can never be negative by construction. This is
different from dev3/temporal.py's signed latency, which does allow negative
values (see tests/test_temporal.py).

This is a confirmed defect in the feature *definition*, not a code bug we can
patch in isolation: models/model.pkl and models/dev2Alfa.pkl were both trained
against this exact (buggy) definition. Changing the formula without retraining
would silently feed the deployed model a distribution shift it never saw. Fixing
it properly requires the official dataset (not included in this delivery) and a
new training run with updated metadata/hashes — see REVISION-TECNICA.md.

This test pins the current behavior so nobody "fixes" the formula in ml/features.py
without also retraining and re-registering the model.
"""
from ml.features import extract_features_from_turns


def test_latency_frac_negative_is_structurally_always_zero():
    # An agent turn that is still ongoing when the caller starts (real overlap,
    # which should intuitively count as a "negative latency" reply) is excluded
    # from the candidate pool entirely, so it can never pull the fraction above 0.
    turns = [
        {"channel": 1, "start": 0.0, "end": 5.0},   # agent speaking 0-5s
        {"channel": 0, "start": 2.0, "end": 3.0},   # caller starts mid-agent-turn (real overlap)
        {"channel": 1, "start": 6.0, "end": 7.0},
        {"channel": 0, "start": 6.5, "end": 8.0},   # caller answers fast (normal case)
    ]
    features = extract_features_from_turns(turns, duration_s=10)
    assert features["latency_frac_negative"] == 0.0
    assert features["latency_n"] > 0
