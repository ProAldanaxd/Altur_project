"""Generates synthetic stereo PCM16/8kHz WAVs for FORMAT/LATENCY/ROBUSTNESS testing only.

These are not real speech and must never be used to claim detection accuracy
against real deepfakes. They exist to exercise /detect and /conversation/analyze
under varied clip shapes (short, long, silent channel, heavy overlap) since the
official dataset is not available in this environment.
"""
import numpy as np
import soundfile as sf
from pathlib import Path

OUT = Path(__file__).parent / "synth_wav"
OUT.mkdir(exist_ok=True)
SR = 8000
RNG = np.random.default_rng(42)


def tone_bursts(duration_s, bursts, freq=220.0, amp=0.3):
    n = int(duration_s * SR)
    x = np.zeros(n, dtype=np.float32)
    for start_s, end_s in bursts:
        a, b = int(start_s * SR), min(int(end_s * SR), n)
        if b > a:
            t = np.arange(b - a) / SR
            x[a:b] = amp * np.sin(2 * np.pi * freq * t) + 0.01 * RNG.standard_normal(b - a)
    return x


def make(name, duration_s, caller_bursts, agent_bursts):
    n = int(duration_s * SR)
    caller = tone_bursts(duration_s, caller_bursts, freq=220)[:n]
    agent = tone_bursts(duration_s, agent_bursts, freq=300)[:n]
    stereo = np.stack([caller, agent], axis=1)
    sf.write(OUT / name, stereo, SR, format="WAV", subtype="PCM_16")


# "Normal" call: alternating turns, ~30s, like a typical full call.
make("normal_30s.wav", 30, [(2, 6), (10, 14), (18, 23), (26, 29)], [(0, 1.5), (6.5, 9.5), (14.5, 17.5), (23.5, 25.5)])

# Short clip: only a few seconds, single caller burst.
make("short_3s.wav", 3, [(0.5, 2.5)], [(0, 0.3)])

# Long clip: ~120s, many alternations.
long_caller = [(i, i + 3) for i in range(2, 115, 8)]
long_agent = [(i, i + 1.5) for i in range(0, 115, 8)]
make("long_120s.wav", 120, long_caller, long_agent)

# Heavy overlap: caller and agent talk simultaneously most of the time.
make("heavy_overlap_20s.wav", 20, [(1, 18)], [(2, 19)])

# Silent client channel: agent talks, caller never does -> should trigger 422 (insufficient client speech).
make("silent_client_15s.wav", 15, [], [(1, 5), (7, 10)])

# Fragment cropped from the middle of a call (simulates a clip with no clear "start"):
# only mid-call activity, no lead-in silence at the very start of caller.
make("cropped_fragment_10s.wav", 10, [(0.2, 4)], [(4.5, 9.8)])

print("Generated:", sorted(p.name for p in OUT.glob("*.wav")))
