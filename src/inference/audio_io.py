"""Input normalization for VoxSplit real-world inputs (Phase 5).

Turns an arbitrary audio file (any sample rate, mono or multichannel, any
loudness) into the mono, model-rate, peak-normalized float32 signal the
separators expect. Centralizes what was previously duplicated as load_mono in
several inference scripts.

Run this file directly for a self-test (no real audio needed):
  python src/inference/audio_io.py
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

MODEL_SR = 8000
DEFAULT_PEAK = 0.9


def to_mono_resampled(signal: np.ndarray, sr: int,
                      target_sr: int = MODEL_SR) -> np.ndarray:
    """Downmix to mono (average channels) and resample to target_sr."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if sr != target_sr:
        tensor = torch.from_numpy(np.ascontiguousarray(signal))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=target_sr)
        signal = tensor.numpy()
    return signal.astype(np.float32)


def peak_normalize(signal: np.ndarray, peak: float = DEFAULT_PEAK
                   ) -> Tuple[np.ndarray, float]:
    """Scale so the max absolute sample equals peak. Returns (signal, scale)."""
    signal = np.asarray(signal, dtype=np.float32)
    m = float(np.max(np.abs(signal))) if signal.size else 0.0
    scale = peak / m if m > 0.0 else 1.0
    return (signal * scale).astype(np.float32), scale


def load_normalized(path: Path, target_sr: int = MODEL_SR,
                    peak: float = DEFAULT_PEAK
                    ) -> Tuple[np.ndarray, int]:
    """Load a file, downmix, resample, peak-normalize. Returns (signal, orig_sr).

    The returned signal is mono float32 at target_sr with peak magnitude ~peak.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = to_mono_resampled(data, sr, target_sr)
    out, _ = peak_normalize(mono, peak)
    return out, sr


def _self_test() -> int:
    rng = np.random.default_rng(0)
    # Stereo at 16 kHz, loud, becomes mono 8 kHz with peak ~0.9.
    stereo = (rng.standard_normal((16000, 2)) * 5.0).astype(np.float32)
    mono = to_mono_resampled(stereo, 16000, MODEL_SR)
    assert mono.ndim == 1, mono.shape
    assert abs(len(mono) - 8000) <= 2, len(mono)  # ~1 s at 8 kHz

    norm, scale = peak_normalize(mono, 0.9)
    assert abs(float(np.max(np.abs(norm))) - 0.9) < 1e-4, np.max(np.abs(norm))
    print(f"stereo16k->mono8k len={len(mono)} peak-scale={scale:.4f}")

    # Silent input must not divide by zero.
    silent, s = peak_normalize(np.zeros(100, np.float32))
    assert s == 1.0 and float(np.max(np.abs(silent))) == 0.0
    print("All audio_io self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
