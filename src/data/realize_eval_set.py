"""Realize a frozen eval-set manifest into audio for VoxSplit Phase 2.

Reads the manifest written by build_eval_set.py and, for every mixture,
loads each source relative to the LibriSpeech root, applies the recorded
linear gain, sums the sources into an overlapped mixture, peak normalizes
the mixture, and applies the SAME scale to the reference sources. This
reproduces the exact make_mixture.py mixing behaviour, so given the same
manifest and the same LibriSpeech corpus it produces identical audio on
every run.

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
from typing import List

import numpy as np
from scipy.io import wavfile
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


def realize_one(mixture: dict, librispeech_root: Path, sample_rate: int,
                normalize_peak: float, out_dir: Path) -> None:
    """Build and write one mixture folder from a manifest entry."""
    signals: List[np.ndarray] = []
    for src in mixture["sources"]:
        path = librispeech_root / src["relpath"]
        sig = load_mono_resampled(path, sample_rate)
        signals.append(sig * float(src["gain"]))

    # Truncate every source to the shortest length so they fully overlap.
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    summed = np.sum(signals, axis=0)
    peak = float(np.max(np.abs(summed)))
    if peak > 0.0:
        scale = normalize_peak / peak
    else:
        scale = 1.0

    mix_out = (summed * scale).astype(np.float32)
    refs_out = [(s * scale).astype(np.float32) for s in signals]

    mix_dir = out_dir / mixture["id"]
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav_float32(mix_dir / "mixture.wav", mix_out, sample_rate)
    for i, ref in enumerate(refs_out, start=1):
        write_wav_float32(mix_dir / f"source{i}.wav", ref, sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Realize an eval-set manifest into mixture and source "
                    "wav files.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/eval_manifest.json"),
                        help="Manifest JSON produced by build_eval_set.py.")
    parser.add_argument("--librispeech-root", type=Path,
                        default=Path("data/LibriSpeech"),
                        help="Base directory the manifest relpaths resolve "
                             "against.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval_set"),
                        help="Output directory for realized mixtures.")
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}")
        return 1

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    sample_rate = int(manifest["sample_rate"])
    normalize_peak = float(manifest["normalize_peak"])
    mixtures = manifest["mixtures"]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Realizing {len(mixtures)} mixtures at {sample_rate} Hz "
          f"into {out_dir} ...")
    for mixture in tqdm(mixtures, desc="realize", unit="mix"):
        realize_one(mixture, args.librispeech_root, sample_rate,
                    normalize_peak, out_dir)

    print(f"Done. Wrote {len(mixtures)} mixture folders to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
