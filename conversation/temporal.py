"""Observable conversation timing, independent of the existing 87-feature model.

Times are seconds and intervals are half-open: [start, end). Channel 0 is the
client and channel 1 the agent. Timing is evidence about audio activity, never
proof of synthesis, deception, interruption intent, or semantic response.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np

from ml.features import vad_segments


SCHEMA_VERSION = "temporal-v1"
FEATURE_NAMES = (
    "dev3_duration_s", "dev3_client_turn_count", "dev3_agent_turn_count",
    "dev3_client_speech_ratio", "dev3_agent_speech_ratio", "dev3_overlap_ratio",
    "dev3_dead_air_ratio", "dev3_client_duration_mean_s", "dev3_client_duration_std_s",
    "dev3_agent_duration_mean_s", "dev3_agent_duration_std_s", "dev3_response_count",
    "dev3_response_mean_s", "dev3_response_std_s", "dev3_response_median_s",
    "dev3_response_negative_fraction", "dev3_response_missing",
    "dev3_client_onset_during_agent_count", "dev3_agent_onset_during_client_count",
    "dev3_simultaneous_onset_count", "dev3_client_restart_count",
    "dev3_client_restart_gap_mean_s", "dev3_client_restart_missing",
    "dev3_client_missing", "dev3_agent_missing", "dev3_zero_duration",
)


def _number(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_s": None, "std_s": None,
                "median_s": None, "min_s": None, "max_s": None}
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean_s": float(data.mean()),
            "std_s": float(data.std()), "median_s": float(np.median(data)),
            "min_s": float(data.min()), "max_s": float(data.max())}


def analyze_turns(turns: list[dict], duration_s: float) -> dict:
    """Normalize activity intervals and report descriptive temporal evidence.

    Invalid timestamps/channels are rejected, not silently clipped. Duplicate,
    intersecting and adjacent same-channel intervals are merged. A response
    candidate is the first client onset strictly after an agent onset and before
    the next agent onset. Simultaneous onsets are excluded. Thus a client onset
    cannot be reused or assigned to an older agent after a newer agent starts.
    A negative latency means the client starts before this agent interval ends.
    This temporal pairing does not establish that speech answers a question.
    """
    duration = _number(duration_s, "duration_s")
    if duration < 0:
        raise ValueError("duration_s must be nonnegative")
    if not isinstance(turns, list):
        raise ValueError("turns must be a list")
    channels: list[list[tuple[float, float]]] = [[], []]
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turns[{index}] must be an object")
        channel = turn.get("channel")
        if (isinstance(channel, (bool, np.bool_)) or
                not isinstance(channel, (int, np.integer)) or channel not in (0, 1)):
            raise ValueError(f"turns[{index}].channel must be integer 0 or 1")
        start = _number(turn.get("start"), f"turns[{index}].start")
        end = _number(turn.get("end"), f"turns[{index}].end")
        if not 0 <= start < end <= duration:
            raise ValueError(f"turns[{index}] must satisfy 0 <= start < end <= duration_s")
        channels[channel].append((start, end))
    client, agent = [_merge(items) for items in channels]
    normalized = sorted(
        ({"channel": channel, "start": start, "end": end}
         for channel, items in enumerate((client, agent)) for start, end in items),
        key=lambda turn: (turn["start"], turn["channel"], turn["end"]),
    )
    union = _merge(client + agent)
    client_time = sum(end - start for start, end in client)
    agent_time = sum(end - start for start, end in agent)
    active_time = sum(end - start for start, end in union)

    overlaps = []
    overlapping_client_indices: set[int] = set()
    i = j = 0
    while i < len(client) and j < len(agent):
        start, end = max(client[i][0], agent[j][0]), min(client[i][1], agent[j][1])
        if end > start:
            overlaps.append({"start": start, "end": end, "duration_s": end - start,
                             "client_turn_index": i, "agent_turn_index": j})
            overlapping_client_indices.add(i)
        if client[i][1] <= agent[j][1]:
            i += 1
        else:
            j += 1
    overlap_time = sum(event["duration_s"] for event in overlaps)
    dead_air = []
    cursor = 0.0
    for start, end in union:
        if start > cursor:
            dead_air.append({"start": cursor, "end": start, "duration_s": start - cursor})
        cursor = end
    if cursor < duration:
        dead_air.append({"start": cursor, "end": duration, "duration_s": duration - cursor})

    client_onsets = [start for start, _ in client]
    agent_onsets = [start for start, _ in agent]
    simultaneous = []
    onset_during_other = []
    agent_onset_indices = {start: index for index, (start, _) in enumerate(agent)}
    for client_index, (start, _) in enumerate(client):
        if start in agent_onset_indices:
            simultaneous.append({"time_s": start, "client_turn_index": client_index,
                                 "agent_turn_index": agent_onset_indices[start]})
    for channel, own, other, other_onsets in (
            (0, client, agent, agent_onsets), (1, agent, client, client_onsets)):
        for index, (start, _) in enumerate(own):
            other_index = bisect_left(other_onsets, start) - 1
            if other_index >= 0 and start < other[other_index][1]:
                onset_during_other.append({"time_s": start, "starting_channel": channel,
                                           "turn_index": index, "active_channel": 1 - channel,
                                           "active_turn_index": other_index})
    onset_during_other.sort(key=lambda event: (event["time_s"], event["starting_channel"]))

    responses = []
    for agent_index, (start, end) in enumerate(agent):
        client_index = bisect_right(client_onsets, start)
        next_agent_start = agent[agent_index + 1][0] if agent_index + 1 < len(agent) else float("inf")
        if client_index < len(client) and client[client_index][0] < next_agent_start:
            responses.append({"agent_turn_index": agent_index, "client_turn_index": client_index,
                              "agent_start_s": start, "agent_end_s": end,
                              "client_start_s": client[client_index][0],
                              "latency_s": client[client_index][0] - end})

    # The next client onset after an overlapping client turn is an observable
    # restart; timing alone cannot establish a recovery, intention, or cause.
    restarts = []
    for index in sorted(overlapping_client_indices):
        if index + 1 < len(client):
            restarts.append({"overlapping_client_turn_index": index,
                             "next_client_turn_index": index + 1,
                             "client_end_s": client[index][1],
                             "next_client_start_s": client[index + 1][0],
                             "gap_s": client[index + 1][0] - client[index][1]})
    latency_values = [event["latency_s"] for event in responses]
    ratio = lambda seconds: seconds / duration if duration else None
    notes = [
        "Intervals use [start,end); touching endpoints do not overlap.",
        "Onsets during other-channel activity are potential interruptions, not established intent.",
        "Response candidates are temporal pairings; timing does not establish a semantic reply.",
        "Temporal observations do not independently establish synthesis or deception.",
    ]
    if len(normalized) < len(turns):
        notes.append("Duplicate, overlapping or adjacent same-channel intervals were merged.")
    if not client or not agent:
        notes.append("One or both channels have no detected speech; interaction statistics may be missing.")
    if not duration:
        notes.append("Zero duration: time ratios are undefined.")
    return {
        "schema_version": SCHEMA_VERSION,
        "duration_s": duration,
        "turns": normalized,
        "summary": {
            "client_turn_count": len(client), "agent_turn_count": len(agent),
            "client_speech_s": client_time, "agent_speech_s": agent_time,
            "active_speech_s": active_time, "overlap_s": overlap_time,
            "dead_air_s": max(0.0, duration - active_time),
            "client_speech_ratio": ratio(client_time), "agent_speech_ratio": ratio(agent_time),
            "overlap_ratio": ratio(overlap_time), "dead_air_ratio": ratio(max(0.0, duration - active_time)),
            "client_turn_duration": _stats([end - start for start, end in client]),
            "agent_turn_duration": _stats([end - start for start, end in agent]),
            "response_latency": _stats(latency_values),
            "response_negative_fraction": (sum(value < 0 for value in latency_values) / len(latency_values)
                                           if latency_values else None),
            "client_onset_during_agent_count": sum(event["starting_channel"] == 0 for event in onset_during_other),
            "agent_onset_during_client_count": sum(event["starting_channel"] == 1 for event in onset_during_other),
            "simultaneous_onset_count": len(simultaneous),
            "client_restart_gap_after_overlap": _stats([event["gap_s"] for event in restarts]),
        },
        "events": {"overlaps": overlaps, "dead_air": dead_air,
                   "onsets_during_other_channel": onset_during_other,
                   "simultaneous_onsets": simultaneous, "response_candidates": responses,
                   "client_restarts_after_overlap": restarts},
        "quality": {"source": "provided_activity_intervals", "input_turn_count": len(turns),
                    "normalized_turn_count": len(normalized),
                    "client_speech_present": bool(client), "agent_speech_present": bool(agent),
                    "notes": notes},
    }


def analyze_channels(client, agent, sample_rate: int) -> dict:
    """Apply the existing energy VAD to finite, equally sized mono waveforms.

    Expected PCM floats are already scaled by the Dev1 decoder. The energy VAD
    is a heuristic (20 ms frames), not a diarization or transcription model.
    """
    if (isinstance(sample_rate, (bool, np.bool_)) or
            not isinstance(sample_rate, (int, np.integer)) or sample_rate < 1):
        raise ValueError("sample_rate must be a positive integer")
    arrays = []
    for name, samples in (("client", client), ("agent", agent)):
        try:
            array = np.asarray(samples, dtype=np.float32)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite mono waveform") from exc
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite mono waveform")
        if array.size and np.max(np.abs(array)) > 1.0:
            raise ValueError(f"{name} PCM samples must be scaled to [-1,1]")
        arrays.append(array)
    if len(arrays[0]) != len(arrays[1]):
        raise ValueError("client and agent must contain the same number of samples")
    turns = vad_segments(arrays[0], sample_rate, 0) + vad_segments(arrays[1], sample_rate, 1)
    analysis = analyze_turns(turns, len(arrays[0]) / sample_rate)
    analysis["quality"].update({"source": "energy_vad", "sample_rate": int(sample_rate),
                                "vad_frame_ms": 20, "vad_min_speech_ms": 200,
                                "vad_merge_gap_ms": 200})
    analysis["quality"]["notes"].append(
        "Energy VAD is approximate: noise, echo, music, backchannels and quiet speech can change events; "
        "sub-200 ms speech may be omitted and gaps below 200 ms merged.")
    return analysis


def extract_model_features(analysis: dict) -> dict[str, float]:
    """26 independent dev3_ features for future fitting, never model87 inputs.

    Missing statistics are zero-filled ONLY in this numeric vector, with explicit
    missing indicators. The human-readable analysis retains None for undefined
    values. No score is invented; use a separately trained/validated calibrator.
    """
    if analysis.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected temporal-v1 analysis")
    summary = analysis["summary"]
    duration = analysis["duration_s"]
    response = summary["response_latency"]
    restart = summary["client_restart_gap_after_overlap"]
    client_duration = summary["client_turn_duration"]
    agent_duration = summary["agent_turn_duration"]
    values = (
        duration, summary["client_turn_count"], summary["agent_turn_count"],
        summary["client_speech_ratio"], summary["agent_speech_ratio"],
        summary["overlap_ratio"], summary["dead_air_ratio"],
        client_duration["mean_s"], client_duration["std_s"],
        agent_duration["mean_s"], agent_duration["std_s"], response["count"],
        response["mean_s"], response["std_s"], response["median_s"],
        summary["response_negative_fraction"], int(response["count"] == 0),
        summary["client_onset_during_agent_count"], summary["agent_onset_during_client_count"],
        summary["simultaneous_onset_count"], restart["count"], restart["mean_s"],
        int(restart["count"] == 0), int(summary["client_turn_count"] == 0),
        int(summary["agent_turn_count"] == 0), int(duration == 0),
    )
    features = {name: 0.0 if value is None else float(value)
                for name, value in zip(FEATURE_NAMES, values)}
    if not all(isfinite(value) for value in features.values()):
        raise ValueError("Temporal model features must be finite")
    return features
