"""Realize a frozen eval-set manifest into audio for VoxSplit Phase 2.

Reads the manifest written by build_eval_set.py and, for every mixture,
loads each source relative to the LibriSpeech root, applies the recorded
linear gain, sums the sources into an overlapped mixture, peak normalizes
the mixture, and applies the SAME scale to the reference sources. This
reproduces the exact make_mixture.py mixing behaviour, so given the same
manifest and the same LibriSpeech corpus it produces identical audio on
every run.

If the manifest carries a "condition" tag written by make_conditions.py, the
same clean mixing is degraded deterministically:
  noise:  a clean mixture is built (not yet normalized), the recorded noise
          segment is scaled to the recorded SNR and added, then the noisy
          mixture is peak normalized and the same scale is applied to the
          CLEAN references (only the mixture holds noise).
  reverb: each source is convolved with a room impulse response regenerated
          from the recorded shoebox room, gained, and summed; the mixture is
          peak normalized and the same scale is applied to the reverberant
          references.
A manifest with no "condition" behaves exactly as the clean pipeline.

Example:
  python src/data/realize_eval_set.py \
      --manifest data/eval_manifest.json \
      --librispeech-root data/LibriSpeech \
      --out-dir data/eval_set
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
from tqdm import tqdm

# Reuse the Phase 1 mixing helpers via sys.path, the same pattern that
# src/eval/evaluate.py uses to import metrics.
_MIXING_DIR = Path(__file__).resolve().parent.parent / "mixing"
sys.path.insert(0, str(_MIXING_DIR))
from make_mixture import load_mono_resampled  # noqa: E402


def write_wav_float32(path: Path, signal: np.ndarray, sample_rate: int
                      ) -> None:
    """Write a float32 mono WAV deterministically (no timestamped chunks).

    scipy.io.wavfile writes a bare IEEE-float WAV, so the same samples
    always yield byte identical files. soundfile writes a PEAK chunk with a
    wall-clock timestamp for float WAVs, which would break byte-level
    reproducibility of the frozen eval set.
    """
    wavfile.write(str(path), sample_rate, signal.astype(np.float32))


def load_gained_sources(mixture: dict, librispeech_root: Path,
                        sample_rate: int) -> List[np.ndarray]:
    """Load every source and multiply by its recorded gain (no truncation)."""
    signals: List[np.ndarray] = []
    for src in mixture["sources"]:
        path = librispeech_root / src["relpath"]
        sig = load_mono_resampled(path, sample_rate)
        signals.append(sig * float(src["gain"]))
    return signals


def peak_normalize(summed: np.ndarray, refs: List[np.ndarray],
                   normalize_peak: float
                   ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Scale summed to the target peak and apply the same scale to refs."""
    peak = float(np.max(np.abs(summed))) if summed.size else 0.0
    scale = normalize_peak / peak if peak > 0.0 else 1.0
    mix_out = (summed * scale).astype(np.float32)
    refs_out = [(r * scale).astype(np.float32) for r in refs]
    return mix_out, refs_out


def realize_clean(mixture: dict, librispeech_root: Path, sample_rate: int,
                  normalize_peak: float) -> Tuple[np.ndarray,
                                                  List[np.ndarray]]:
    """Clean overlapped mixing, identical to the original pipeline."""
    signals = load_gained_sources(mixture, librispeech_root, sample_rate)
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]
    summed = np.sum(signals, axis=0)
    return peak_normalize(summed, signals, normalize_peak)


def load_noise_segment(noise_path: Path, sample_rate: int, start: int,
                       length: int) -> np.ndarray:
    """Load a noise file, resample to sample_rate, take a looped segment.

    Returns exactly length samples starting at start, tiling/looping the noise
    if it is shorter than needed. Modulo indexing handles both looping and a
    start offset that exceeds the noise length.
    """
    noise = load_mono_resampled(noise_path, sample_rate)
    if noise.size == 0:
        return np.zeros(length, dtype=np.float32)
    idx = (start + np.arange(length)) % noise.size
    return noise[idx].astype(np.float32)


def realize_noise(mixture: dict, librispeech_root: Path, sample_rate: int,
                  normalize_peak: float, noise_dir: Path
                  ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Clean mixing plus additive noise at the recorded SNR.

    References stay clean; only the mixture carries the noise.
    """
    signals = load_gained_sources(mixture, librispeech_root, sample_rate)
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]
    clean_mix = np.sum(signals, axis=0)

    noise_meta = mixture["noise"]
    noise_path = noise_dir / noise_meta["relpath"]
    noise = load_noise_segment(noise_path, sample_rate,
                               int(noise_meta["start"]), min_len)

    p_mix = float(np.mean(clean_mix ** 2))
    p_noise = float(np.mean(noise ** 2))
    snr_db = float(noise_meta["snr_db"])
    if p_noise > 0.0 and p_mix > 0.0:
        g = np.sqrt(p_mix / (p_noise * (10.0 ** (snr_db / 10.0))))
    else:
        g = 0.0
    noisy = clean_mix + g * noise

    # References remain the clean gained sources; only the mixture is noisy.
    return peak_normalize(noisy, signals, normalize_peak)


def build_rir(room_dim: List[float], rt60: float, mic_pos: List[float],
              source_pos: List[float], sample_rate: int) -> np.ndarray:
    """Regenerate a single room impulse response with no RNG.

    Uses pra.inverse_sabine to turn the recorded rt60 and room geometry into
    a wall absorption and image-source order, then the deterministic image
    source method. Given the same recorded parameters this is byte
    reproducible.
    """
    import pyroomacoustics as pra
    absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=sample_rate,
                       materials=pra.Material(absorption),
                       max_order=max_order)
    room.add_source(list(source_pos))
    room.add_microphone(list(mic_pos))
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=np.float32)


def realize_reverb(mixture: dict, librispeech_root: Path, sample_rate: int,
                   normalize_peak: float) -> Tuple[np.ndarray,
                                                   List[np.ndarray]]:
    """Convolve each source with its room impulse response, then mix.

    The reverberant, gained source is the reference for its speaker.
    """
    reverb = mixture["reverb"]
    room_dim = reverb["room_dim"]
    rt60 = float(reverb["rt60"])
    mic_pos = reverb["mic_pos"]
    source_positions = reverb["source_pos"]

    reverberant: List[np.ndarray] = []
    for src, spos in zip(mixture["sources"], source_positions):
        path = librispeech_root / src["relpath"]
        sig = load_mono_resampled(path, sample_rate)
        rir = build_rir(room_dim, rt60, mic_pos, spos, sample_rate)
        wet = fftconvolve(sig, rir, mode="full")
        reverberant.append((wet * float(src["gain"])).astype(np.float32))

    # Keep the full reverberant tails, truncate all to a common min length.
    min_len = min(len(s) for s in reverberant)
    reverberant = [s[:min_len] for s in reverberant]
    summed = np.sum(reverberant, axis=0)
    return peak_normalize(summed, reverberant, normalize_peak)


def realize_one(mixture: dict, librispeech_root: Path, sample_rate: int,
                normalize_peak: float, out_dir: Path, condition: str,
                noise_dir: Path) -> None:
    """Build and write one mixture folder from a manifest entry."""
    if condition == "reverb":
        mix_out, refs_out = realize_reverb(
            mixture, librispeech_root, sample_rate, normalize_peak)
    elif condition == "noise":
        mix_out, refs_out = realize_noise(
            mixture, librispeech_root, sample_rate, normalize_peak, noise_dir)
    else:
        mix_out, refs_out = realize_clean(
            mixture, librispeech_root, sample_rate, normalize_peak)

    mix_dir = out_dir / mixture["id"]
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav_float32(mix_dir / "mixture.wav", mix_out, sample_rate)
    for i, ref in enumerate(refs_out, start=1):
        write_wav_float32(mix_dir / f"source{i}.wav", ref, sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Realize an eval-set manifest into mixture and source "
                    "wav files. Supports clean, noise, and reverb manifests.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/eval_manifest.json"),
                        help="Manifest JSON produced by build_eval_set.py or "
                             "make_conditions.py.")
    parser.add_argument("--librispeech-root", type=Path,
                        default=Path("data/LibriSpeech"),
                        help="Base directory the manifest relpaths resolve "
                             "against.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval_set"),
                        help="Output directory for realized mixtures.")
    parser.add_argument("--noise-dir", type=Path, default=None,
                        help="Override the noise root for a noise manifest. "
                             "Defaults to the manifest's top-level noise_dir.")
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}")
        return 1

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    sample_rate = int(manifest["sample_rate"])
    normalize_peak = float(manifest["normalize_peak"])
    mixtures = manifest["mixtures"]
    condition = manifest.get("condition") or "clean"

    noise_dir: Path = Path("")
    if condition == "noise":
        if args.noise_dir is not None:
            noise_dir = args.noise_dir
        elif manifest.get("noise_dir"):
            noise_dir = Path(manifest["noise_dir"])
        else:
            print("Noise manifest has no noise_dir and --noise-dir not given.")
            return 1
        if not noise_dir.is_dir():
            print(f"Noise directory not found: {noise_dir}")
            return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Realizing {len(mixtures)} '{condition}' mixtures at "
          f"{sample_rate} Hz into {out_dir} ...")
    if condition == "noise":
        print(f"Noise dir: {noise_dir}")
    for mixture in tqdm(mixtures, desc="realize", unit="mix"):
        realize_one(mixture, args.librispeech_root, sample_rate,
                    normalize_peak, out_dir, condition, noise_dir)

    print(f"Done. Wrote {len(mixtures)} mixture folders to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
