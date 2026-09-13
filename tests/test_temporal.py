import json

import numpy as np
import pytest

from conversation.temporal import FEATURE_NAMES, analyze_channels, analyze_turns, extract_model_features


def turn(channel, start, end):
    return {"channel": channel, "start": start, "end": end}


def test_exact_interval_union_dedup_and_order_independence():
    turns = [turn(0, 1, 4), turn(1, 3, 5), turn(0, 2, 3), turn(0, 1, 4),
             turn(1, 4, 6), turn(0, 4, 5), turn(0, 8, 9)]
    analysis = analyze_turns(turns, 10)
    assert analysis["turns"] == [turn(0, 1, 5), turn(1, 3, 6), turn(0, 8, 9)]
    assert analyze_turns(list(reversed(turns)), 10) == analysis
    summary = analysis["summary"]
    assert summary["client_speech_s"] == 5
    assert summary["agent_speech_s"] == 3
    assert summary["active_speech_s"] == 6
    assert summary["overlap_s"] == 2
    assert summary["dead_air_s"] == 4
    assert summary["active_speech_s"] == summary["client_speech_s"] + summary["agent_speech_s"] - summary["overlap_s"]


def test_response_candidates_are_signed_unique_and_belong_to_latest_agent():
    analysis = analyze_turns([turn(1, 0, 2), turn(0, 1, 1.5), turn(1, 3, 4),
                              turn(1, 5, 6), turn(0, 6.5, 7), turn(0, 8, 9)], 10)
    pairs = analysis["events"]["response_candidates"]
    assert [(p["agent_turn_index"], p["client_turn_index"], p["latency_s"]) for p in pairs] == [(0, 0, -1), (2, 1, .5)]
    assert analysis["summary"]["response_negative_fraction"] == .5
    assert len({p["client_turn_index"] for p in pairs}) == len(pairs)


def test_tied_onsets_are_simultaneous_and_touching_ends_do_not_overlap():
    analysis = analyze_turns([turn(1, 0, 1), turn(0, 0, 1),
                              turn(1, 2, 3), turn(0, 3, 4)], 4)
    summary = analysis["summary"]
    assert summary["simultaneous_onset_count"] == 1
    assert summary["overlap_s"] == 1
    assert summary["client_onset_during_agent_count"] == 0
    assert summary["agent_onset_during_client_count"] == 0
    assert [p["latency_s"] for p in analysis["events"]["response_candidates"]] == [0]


def test_client_onset_tied_with_next_agent_is_not_old_response():
    analysis = analyze_turns([turn(1, 0, 1), turn(1, 2, 3), turn(0, 2, 2.5)], 4)
    assert analysis["events"]["response_candidates"] == []
    assert analysis["summary"]["simultaneous_onset_count"] == 1


def test_onset_direction_and_restart_are_descriptive():
    analysis = analyze_turns([turn(0, 0, 3), turn(1, 1, 2), turn(1, 4, 6),
                              turn(0, 5, 7), turn(0, 8, 9)], 10)
    assert analysis["summary"]["agent_onset_during_client_count"] == 1
    assert analysis["summary"]["client_onset_during_agent_count"] == 1
    restarts = analysis["events"]["client_restarts_after_overlap"]
    assert [event["gap_s"] for event in restarts] == [2, 1]
    assert "synthetic" not in analysis["summary"]


@pytest.mark.parametrize("duration", [0, 10])
def test_silence_has_missing_statistics_and_finite_features(duration):
    analysis = analyze_turns([], duration)
    assert analysis["summary"]["dead_air_s"] == duration
    assert analysis["summary"]["response_latency"]["mean_s"] is None
    vector = extract_model_features(analysis)
    assert tuple(vector) == FEATURE_NAMES
    assert len(vector) == 26
    assert np.isfinite(list(vector.values())).all()
    assert vector["dev3_response_missing"] == 1
    assert vector["dev3_client_missing"] == 1
    assert vector["dev3_agent_missing"] == 1
    assert vector["dev3_zero_duration"] == float(duration == 0)
    json.dumps(analysis, allow_nan=False)


@pytest.mark.parametrize("bad", [turn(0, -1, 2), turn(1, 2, 1), turn(0, 1, 1),
                                 turn(0, 0, 11), turn(2, 0, 1), turn(True, 0, 1),
                                 turn(0, float("nan"), 1), turn(0, 0, float("inf")),
                                 turn(0, "0", 1), {"channel": 0, "start": 0}])
def test_invalid_turns_raise(bad):
    with pytest.raises(ValueError):
        analyze_turns([bad], 10)


@pytest.mark.parametrize("duration", [-1, float("inf"), float("nan"), True, "10"])
def test_invalid_duration_raises(duration):
    with pytest.raises(ValueError):
        analyze_turns([], duration)


def test_channel_vad_and_silence():
    client, agent = np.zeros(40000, dtype=np.float32), np.zeros(40000, dtype=np.float32)
    t = np.arange(8000) / 8000
    client[16000:24000] = .4 * np.sin(2 * np.pi * 300 * t)
    agent[8000:16000] = .4 * np.sin(2 * np.pi * 400 * t)
    analysis = analyze_channels(client, agent, 8000)
    assert analysis["turns"] == [turn(1, 1, 2), turn(0, 2, 3)]
    assert analysis["quality"]["source"] == "energy_vad"
    assert analysis["summary"]["response_latency"]["mean_s"] == 0
    assert analyze_channels(np.zeros(8), np.zeros(8), 8000)["turns"] == []
    assert analyze_channels([], [], 8000)["duration_s"] == 0


@pytest.mark.parametrize("client,agent,sr", [([0], [0, 0], 8000), ([[0]], [[0]], 8000),
                                              ([float("nan")], [0], 8000),
                                              ([2], [0], 8000), ([0], [0], 0),
                                              ([0], [0], 8000.0), ([0], [0], True)])
def test_invalid_channels_raise(client, agent, sr):
    with pytest.raises(ValueError):
        analyze_channels(client, agent, sr)


def test_against_independent_sample_grid_interval_oracle():
    # A discrete oracle independently checks exact set arithmetic for integer boundaries.
    rng = np.random.default_rng(9183)
    for _ in range(40):
        turns = []
        expected = np.zeros((2, 20), dtype=bool)
        for _ in range(20):
            channel = int(rng.integers(0, 2))
            start = int(rng.integers(0, 20))
            end = int(rng.integers(start + 1, 21))
            turns.append(turn(channel, start, end))
            expected[channel, start:end] = True
        result = analyze_turns(turns, 20)["summary"]
        assert result["overlap_s"] == np.logical_and(*expected).sum()
        assert result["active_speech_s"] == np.logical_or(*expected).sum()
        assert result["dead_air_s"] == np.logical_not(np.logical_or(*expected)).sum()
