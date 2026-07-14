"""Speaking-timeline extraction for VoxSplit Phase 6.

Derives a per-speaker "who spoke when" timeline directly from the separated
tracks with a simple energy VAD, so no gated diarization model (pyannote needs
an accepted licence + HF token) is required. Each separated speaker already IS
one source, so the diarization question reduces to voice-activity detection on
that source.

Run this file directly for a self-test (synthetic, no real data):
  python src/inference/timeline.py
"""
from typing import List, Tuple

import numpy as np

Segment = Tuple[float, float]


def speaking_segments(wav: np.ndarray, sr: int, frame_ms: float = 25.0,
                      hop_ms: float = 10.0, rel_db: float = 20.0,
                      min_speech_ms: float = 200.0,
                      max_gap_ms: float = 300.0) -> List[Segment]:
    """Energy VAD -> list of (start_s, end_s) speaking segments.

    A frame is "active" if its RMS is within rel_db of the track's loudest
    frame. Active frames are joined into segments, gaps shorter than max_gap
    are bridged, and segments shorter than min_speech are dropped.
    """
    wav = np.asarray(wav, dtype=np.float64).flatten()
    if wav.size == 0:
        return []
    frame = max(int(sr * frame_ms / 1000.0), 1)
    hop = max(int(sr * hop_ms / 1000.0), 1)

    rms = []
    for start in range(0, max(len(wav) - frame, 0) + 1, hop):
        seg = wav[start:start + frame]
        rms.append(float(np.sqrt(np.mean(seg * seg) + 1e-12)))
    rms = np.asarray(rms)
    if rms.size == 0:
        return []

    peak = float(rms.max())
    if peak <= 1e-4:   # effectively silent (zeros give ~1e-6 from the eps)
        return []
    floor = peak * (10.0 ** (-rel_db / 20.0))
    active = rms >= floor

    # Frame index -> time (frame centre).
    def t(i: int) -> float:
        return (i * hop + frame / 2.0) / sr

    # Collect raw active runs.
    segments: List[Segment] = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            segments.append((t(i), t(j - 1)))
            i = j
        else:
            i += 1

    # Bridge short gaps.
    max_gap = max_gap_ms / 1000.0
    merged: List[Segment] = []
    for s, e in segments:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Drop too-short segments.
    min_speech = min_speech_ms / 1000.0
    return [(s, e) for s, e in merged if e - s >= min_speech]


def plot_timeline(segments_per_speaker: List[List[Segment]], duration: float):
    """Return a matplotlib Figure of horizontal speaking bars per speaker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(segments_per_speaker)
    fig, ax = plt.subplots(figsize=(8, 0.6 * max(n, 1) + 1))
    for idx, segs in enumerate(segments_per_speaker):
        for s, e in segs:
            ax.barh(idx, e - s, left=s, height=0.6,
                    color=f"C{idx % 10}")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Speaker {i + 1}" for i in range(n)])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, max(duration, 0.1))
    ax.set_title("Speaking timeline")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def _self_test() -> int:
    sr = 8000
    dur = 4
    n = dur * sr
    wav = np.zeros(n, dtype=np.float32)
    # Speech only in the middle 1.5..3.0 s.
    a, b = int(1.5 * sr), int(3.0 * sr)
    t = np.arange(b - a) / sr
    wav[a:b] = 0.3 * np.sin(2 * np.pi * 200 * t)

    segs = speaking_segments(wav, sr)
    print(f"segments: {[(round(s, 2), round(e, 2)) for s, e in segs]}")
    assert len(segs) == 1, segs
    s, e = segs[0]
    assert 1.3 <= s <= 1.7 and 2.8 <= e <= 3.2, (s, e)

    # Silent track -> no segments.
    assert speaking_segments(np.zeros(n, np.float32), sr) == []

    fig = plot_timeline([segs, []], duration=dur)
    assert fig is not None
    print("All timeline self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
