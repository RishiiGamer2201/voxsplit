"""Audio-visual track-to-speaker assignment (VoxSplit AV mode).

The working core of AV mode: given the audio tracks VoxSplit separated and the
per-face mouth-motion signals from the video, decide WHICH track belongs to
WHICH on-screen speaker by correlating each track's loudness envelope with each
face's mouth movement (a talking face's mouth opens when its voice is active).
Assignment is one-to-one via the Hungarian algorithm on negative correlation.

This is a genuine audio-visual feature (the video determines speaker identity)
that runs entirely in our env, without an external AV-separation model. It is
complementary to full AV masking (RTFS-Net etc.), which is documented in the
README as the heavier alternative.

Run this file directly for a self-test (synthetic signals, no data):
  python src/av/av_assign.py
"""
from typing import List, Tuple

import numpy as np


def audio_envelope(wav: np.ndarray, sr: int, num_frames: int,
                   fps: float) -> np.ndarray:
    """Per-video-frame RMS loudness envelope of an audio track.

    Frames the audio into num_frames windows aligned to the video frame rate
    and returns a length-num_frames envelope (zero-mean, unit-variance).
    """
    wav = np.asarray(wav, dtype=np.float64).flatten()
    hop = sr / fps
    env = np.empty(num_frames)
    win = int(round(hop))
    for i in range(num_frames):
        start = int(round(i * hop))
        seg = wav[start:start + win]
        env[i] = np.sqrt(np.mean(seg * seg)) if seg.size else 0.0
    return _standardize(env)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-9)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = _standardize(a[:n])
    b = _standardize(b[:n])
    return float(np.mean(a * b))


def assign_tracks_to_faces(track_envelopes: List[np.ndarray],
                           face_motions: List[np.ndarray]
                           ) -> Tuple[List[int], np.ndarray]:
    """One-to-one assign audio tracks to faces by envelope-motion correlation.

    Returns (assignment, corr_matrix) where assignment[t] is the face index for
    track t (or -1 if there are more tracks than faces). corr_matrix is
    [num_tracks, num_faces] of correlations.
    """
    from scipy.optimize import linear_sum_assignment

    nt, nf = len(track_envelopes), len(face_motions)
    corr = np.zeros((nt, nf))
    for t in range(nt):
        for f in range(nf):
            corr[t, f] = _correlation(track_envelopes[t], face_motions[f])

    assignment = [-1] * nt
    if nt and nf:
        rows, cols = linear_sum_assignment(-corr)  # maximize correlation
        for r, c in zip(rows, cols):
            assignment[r] = int(c)
    return assignment, corr


def _self_test() -> int:
    rng = np.random.default_rng(0)
    n = 200
    # Two distinct activity patterns (speaker A talks early, B talks late).
    motion_a = np.zeros(n); motion_a[20:90] = 1.0
    motion_b = np.zeros(n); motion_b[110:180] = 1.0
    motion_a += 0.05 * rng.standard_normal(n)
    motion_b += 0.05 * rng.standard_normal(n)

    # Audio envelopes matching each (track 0 ~ B, track 1 ~ A) to test that
    # the assignment recovers the cross mapping.
    env_track0 = motion_b + 0.05 * rng.standard_normal(n)
    env_track1 = motion_a + 0.05 * rng.standard_normal(n)

    assign, corr = assign_tracks_to_faces([env_track0, env_track1],
                                          [motion_a, motion_b])
    print(f"assignment (track->face): {assign}")
    print(f"corr matrix:\n{np.round(corr, 2)}")
    assert assign == [1, 0], assign  # track0->faceB(1), track1->faceA(0)

    # audio_envelope: a tone burst in the middle shows up as a mid envelope.
    sr, fps = 8000, 25.0
    dur_frames = 50
    wav = np.zeros(int(dur_frames / fps * sr), dtype=np.float32)
    mid = len(wav) // 2
    wav[mid - sr // 4: mid + sr // 4] = 0.5
    env = audio_envelope(wav, sr, dur_frames, fps)
    assert env.argmax() > dur_frames // 4 and env.argmax() < 3 * dur_frames // 4
    print("All av_assign self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
