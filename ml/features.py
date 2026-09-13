"""Funciones temporales de Dev 2; conservadas para compatibilidad de features."""
import numpy as np
FRAME_MS = 20
ENERGY_PERCENTILE = 40
THRESH_SCALE = 0.25
MIN_GAP_MS = 200
MIN_SPEECH_MS = 200


def vad_segments(x, sr, channel, energy_percentile=ENERGY_PERCENTILE,
                 thresh_scale=THRESH_SCALE, min_gap_ms=MIN_GAP_MS,
                 min_speech_ms=MIN_SPEECH_MS):
    """
    Energy VAD with a per-channel adaptive noise floor.

    Rebuilds the turn segments that turns/*.json gives us at training time.
    All knobs exposed so calibrate_vad.py can sweep them.
    """
    frame = int(sr * FRAME_MS / 1000)
    if frame < 1 or len(x) < frame:
        return []

    n = len(x) // frame
    energy = np.sqrt(np.mean(
        x[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)

    floor = np.percentile(energy, energy_percentile)
    peak = np.percentile(energy, 95)
    active = energy > (floor + thresh_scale * (peak - floor))

    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append([start * FRAME_MS / 1000, i * FRAME_MS / 1000])
            start = None
    if start is not None:
        segs.append([start * FRAME_MS / 1000, n * FRAME_MS / 1000])

    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * 1000 < min_gap_ms:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    return [{"channel": channel, "start": round(a, 2), "end": round(b, 2)}
            for a, b in merged if (b - a) * 1000 >= min_speech_ms]

def _entropy(arr, bins=10):
    a = np.asarray(arr, dtype=float)
    if a.size < 3:
        return 0.0
    hist, _ = np.histogram(a, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _stat_block(arr, prefix, out):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    out[f"{prefix}_n"] = float(a.size)
    if a.size == 0:
        for k in ["mean", "std", "min", "max", "median", "iqr", "cv"]:
            out[f"{prefix}_{k}"] = 0.0
        return out
    out[f"{prefix}_mean"] = float(np.mean(a))
    out[f"{prefix}_std"] = float(np.std(a))
    out[f"{prefix}_min"] = float(np.min(a))
    out[f"{prefix}_max"] = float(np.max(a))
    out[f"{prefix}_median"] = float(np.median(a))
    out[f"{prefix}_iqr"] = float(np.percentile(a, 75) - np.percentile(a, 25))
    m = np.mean(a)
    out[f"{prefix}_cv"] = float(np.std(a) / m) if abs(m) > 1e-6 else 0.0
    return out

def _signed_latencies(caller, agent):
    """Latency v2: for each agent turn, pair the first caller onset strictly

    after this agent's onset and before the next agent's onset (same
    definition conversation/temporal.py already uses and tests, see
    tests/test_temporal.py::test_response_candidates_are_signed_unique_and_belong_to_latest_agent).
    latency = caller_start - agent_end, which is negative on real overlap.
    Each caller turn is paired at most once, to the most recent qualifying
    agent turn — unlike legacy pairing, this does not discard the agent
    turn just because it hasn't ended yet.
    """
    latencies = []
    for index, at in enumerate(agent):
        next_agent_start = agent[index + 1]["start"] if index + 1 < len(agent) else float("inf")
        candidates = [ct for ct in caller if at["start"] < ct["start"] < next_agent_start]
        if candidates:
            first = min(candidates, key=lambda ct: ct["start"])
            latencies.append(first["start"] - at["end"])
    return latencies


def extract_features_from_turns(turns, duration_s=None, *, latency_pairing="legacy"):
    """Turn timings -> 87 features. Identical code path to training.

    `latency_pairing` controls only the `latency_*` block:

    - "legacy" (default): the exact formula models/model.pkl and
      models/dev2Alfa.pkl were trained against. Only pairs a caller turn
      with an agent turn that has ALREADY ENDED, so `latency_frac_negative`
      is structurally always 0 (see README.md, seccion 6 (hallazgo #6)) and
      real overlap is silently excluded from the latency stats rather than
      counted as negative. Kept as the default so nothing calling this
      function without the keyword (app/model.py's Detector, in particular)
      changes behavior in any way.
    - "signed_v2": the corrected pairing (see `_signed_latencies`), which
      does produce negative latencies on real overlap. This is NOT wired
      into any deployed model yet: models/model.pkl and models/dev2Alfa.pkl
      were trained with "legacy" and must be retrained before this can be
      used in production. train_validate.py accepts --latency-pairing to
      produce that next generation of weights once the official dataset is
      available; see README.md (seccion 6, hallazgo #6) for the exact procedure.
    """
    if latency_pairing not in ("legacy", "signed_v2"):
        raise ValueError("latency_pairing must be 'legacy' or 'signed_v2'")
    turns = sorted(turns, key=lambda t: t["start"])
    caller = [t for t in turns if t["channel"] == 0]
    agent = [t for t in turns if t["channel"] == 1]

    f = {}
    total = duration_s or max((t["end"] for t in turns), default=1.0)

    f["total_duration"] = float(total)
    f["n_turns_total"] = float(len(turns))
    f["n_turns_caller"] = float(len(caller))
    f["n_turns_agent"] = float(len(agent))
    f["turn_ratio"] = len(caller) / max(len(agent), 1)

    caller_durs = [t["end"] - t["start"] for t in caller]
    agent_durs = [t["end"] - t["start"] for t in agent]
    _stat_block(caller_durs, "caller_dur", f)
    _stat_block(agent_durs, "agent_dur", f)
    f["caller_dur_entropy"] = _entropy(caller_durs)

    caller_talk = float(np.sum(caller_durs)) if caller_durs else 0.0
    agent_talk = float(np.sum(agent_durs)) if agent_durs else 0.0
    f["caller_talk_time"] = caller_talk
    f["caller_talk_ratio"] = caller_talk / max(total, 1e-6)
    f["talk_balance"] = caller_talk / max(caller_talk + agent_talk, 1e-6)

    if latency_pairing == "signed_v2":
        latencies = _signed_latencies(caller, agent)
    else:
        latencies = []
        for ct in caller:
            prev = [a["end"] for a in agent if a["end"] <= ct["start"]]
            if prev:
                latencies.append(ct["start"] - max(prev))
    _stat_block(latencies, "latency", f)
    f["latency_entropy"] = _entropy(latencies)
    f["latency_frac_fast"] = float(np.mean([l < 0.3 for l in latencies])) if latencies else 0.0
    f["latency_frac_slow"] = float(np.mean([l > 2.0 for l in latencies])) if latencies else 0.0
    f["latency_frac_negative"] = float(np.mean([l < 0 for l in latencies])) if latencies else 0.0

    overlaps = []
    for ct in caller:
        for at in agent:
            lo, hi = max(ct["start"], at["start"]), min(ct["end"], at["end"])
            if hi > lo:
                overlaps.append(hi - lo)
    _stat_block(overlaps, "overlap", f)
    f["overlap_total"] = float(np.sum(overlaps)) if overlaps else 0.0
    f["overlap_rate"] = len(overlaps) / max(len(caller), 1)
    f["overlap_ratio_time"] = f["overlap_total"] / max(total, 1e-6)

    rec_gaps, rec_durs = [], []
    for i, ct in enumerate(caller[:-1]):
        if any(a["start"] < ct["end"] and a["end"] > ct["start"] for a in agent):
            nxt = caller[i + 1]
            rec_gaps.append(nxt["start"] - ct["end"])
            rec_durs.append(nxt["end"] - nxt["start"])
    _stat_block(rec_gaps, "recovery_gap", f)
    _stat_block(rec_durs, "recovery_dur", f)
    f["n_interrupted_turns"] = float(len(rec_gaps))

    caller_gaps = [caller[i + 1]["start"] - caller[i]["end"]
                   for i in range(len(caller) - 1)]
    _stat_block(caller_gaps, "caller_gap", f)
    f["caller_gap_entropy"] = _entropy(caller_gaps)

    merged = []
    for t in turns:
        if merged and t["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t["end"])
        else:
            merged.append([t["start"], t["end"]])
    covered = sum(hi - lo for lo, hi in merged)
    f["dead_air_ratio"] = 1.0 - (covered / max(total, 1e-6))

    onsets = [t["start"] for t in caller]
    iti = np.diff(onsets) if len(onsets) > 1 else np.array([])
    _stat_block(iti, "iti", f)
    f["iti_entropy"] = _entropy(iti)
    if iti.size > 2 and np.std(iti) > 1e-6:
        ac = np.corrcoef(iti[:-1], iti[1:])[0, 1]
        f["iti_autocorr"] = float(ac) if np.isfinite(ac) else 0.0
    else:
        f["iti_autocorr"] = 0.0

    seq = [t["channel"] for t in turns]
    f["switch_rate"] = sum(1 for i in range(len(seq) - 1)
                           if seq[i] != seq[i + 1]) / max(len(seq), 1)
    f["caller_consecutive_rate"] = sum(
        1 for i in range(len(seq) - 1)
        if seq[i] == 0 and seq[i + 1] == 0) / max(len(caller), 1)

    return f
