"""Synthetic overlapping-mixture generator for VoxSplit Phase 1.

Builds fully overlapped multi-speaker mixtures from a directory of single
speaker source utterances (for example LibriSpeech). Each mixture is written
with its clean reference sources and a manifest, so it doubles as the
test-data generator for separation and scoring.

Example:
  python src/mixing/make_mixture.py --source-dir data/LibriSpeech/dev-clean \
      --num-speakers 3 --out-dir data/mixtures --num-mixtures 5 --seed 0
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


AUDIO_EXTS = (".wav", ".flac")


def find_sources(source_dir: Path, min_seconds: float,
                 sample_rate: int) -> Dict[str, List[Path]]:
    """Scan source_dir recursively and group usable files by speaker id.

    Speaker id is the filename prefix before the first '-' (LibriSpeech
    style, for example 84-121123-0000.flac -> speaker 84). If the filename
    has no '-', the top level parent folder name is used instead. Files
    shorter than min_seconds are skipped.
    """
    by_speaker: Dict[str, List[Path]] = {}
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        try:
            info = sf.info(str(path))
        except Exception:
            continue
        if info.frames <= 0 or info.samplerate <= 0:
            continue
        duration = info.frames / float(info.samplerate)
        if duration < min_seconds:
            continue
        stem = path.stem
        if "-" in stem:
            speaker = stem.split("-", 1)[0]
        else:
            speaker = path.parent.name
        by_speaker.setdefault(speaker, []).append(path)
    return by_speaker


def load_mono_resampled(path: Path, sample_rate: int) -> np.ndarray:
    """Load an audio file, downmix to mono, resample to sample_rate.

    Returns a float32 numpy array of shape [time].
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # data is [time, channels]; downmix to mono by averaging channels.
    mono = data.mean(axis=1)
    if sr != sample_rate:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=sample_rate)
        mono = tensor.numpy()
    return mono.astype(np.float32)


def rms(signal: np.ndarray) -> float:
    """Root mean square level of a signal, with a small floor."""
    return float(np.sqrt(np.mean(signal ** 2) + 1e-12))


def build_mixture(sources: List[np.ndarray], rng: random.Random
                  ) -> Tuple[np.ndarray, List[np.ndarray], List[float], float]:
    """Combine equal-length sources into one overlapped mixture.

    Each source after the first gets a random loudness offset drawn
    uniformly in [-5, 5] dB relative to the first source. The summed
    mixture is then scaled so its peak magnitude is about 0.9. The same
    mixture scale is applied to every stored reference source so that
    references and mixture stay consistent for SI-SDR scoring.

    Returns (mixture, scaled_sources, applied_gains, mixture_scale).
    """
    ref_rms = rms(sources[0])
    gains: List[float] = []
    scaled: List[np.ndarray] = []
    for i, src in enumerate(sources):
        if i == 0:
            gain = 1.0
        else:
            offset_db = rng.uniform(-5.0, 5.0)
            target_rms = ref_rms * (10.0 ** (offset_db / 20.0))
            gain = target_rms / rms(src)
        gains.append(float(gain))
        scaled.append(src * gain)

    mixture = np.sum(scaled, axis=0)
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.0:
        mixture_scale = 0.9 / peak
    else:
        mixture_scale = 1.0

    mixture = (mixture * mixture_scale).astype(np.float32)
    scaled = [(s * mixture_scale).astype(np.float32) for s in scaled]
    return mixture, scaled, gains, float(mixture_scale)


def make_one_mixture(by_speaker: Dict[str, List[Path]], num_speakers: int,
                     sample_rate: int, rng: random.Random,
                     ) -> Tuple[np.ndarray, List[np.ndarray], List[Path],
                                List[str], List[float], float]:
    """Sample K distinct speakers, load and align them, build a mixture."""
    speakers = list(by_speaker.keys())
    if len(speakers) < num_speakers:
        raise ValueError(
            f"Need {num_speakers} distinct speakers but only found "
            f"{len(speakers)} in the source directory.")
    chosen_speakers = rng.sample(speakers, num_speakers)

    picked_paths: List[Path] = []
    speaker_ids: List[str] = []
    sources: List[np.ndarray] = []
    for spk in chosen_speakers:
        path = rng.choice(by_speaker[spk])
        picked_paths.append(path)
        speaker_ids.append(spk)
        sources.append(load_mono_resampled(path, sample_rate))

    # Truncate every source to the shortest length so they fully overlap.
    min_len = min(len(s) for s in sources)
    sources = [s[:min_len] for s in sources]

    mixture, scaled, gains, mixture_scale = build_mixture(sources, rng)
    return (mixture, scaled, picked_paths, speaker_ids, gains, mixture_scale)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic overlapping multi-speaker mixtures.")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Directory of source speech (searched "
                             "recursively for .wav and .flac).")
    parser.add_argument("--num-speakers", "-k", type=int, default=3,
                        help="Number of overlapping speakers per mixture.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output directory for mixtures.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Target sample rate in Hz.")
    parser.add_argument("--num-mixtures", type=int, default=1,
                        help="How many mixtures to generate.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for full reproducibility.")
    parser.add_argument("--min-seconds", type=float, default=2.0,
                        help="Skip source files shorter than this.")
    args = parser.parse_args()

    if args.num_speakers < 1:
        print("num-speakers must be at least 1.")
        return 1

    source_dir: Path = args.source_dir
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}")
        return 1

    print(f"Scanning sources under {source_dir} ...")
    by_speaker = find_sources(source_dir, args.min_seconds, args.sample_rate)
    total_files = sum(len(v) for v in by_speaker.values())
    print(f"Found {total_files} usable files across "
          f"{len(by_speaker)} speakers.")
    if len(by_speaker) < args.num_speakers:
        print(f"Not enough distinct speakers ({len(by_speaker)}) for "
              f"{args.num_speakers}-speaker mixtures.")
        return 1

    rng = random.Random(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_mixtures):
        (mixture, scaled, picked_paths, speaker_ids, gains,
         mixture_scale) = make_one_mixture(
            by_speaker, args.num_speakers, args.sample_rate, rng)

        mix_dir = out_dir / f"mixture_{idx:03d}"
        mix_dir.mkdir(parents=True, exist_ok=True)

        sf.write(str(mix_dir / "mixture.wav"), mixture, args.sample_rate,
                 subtype="FLOAT")
        for i, src in enumerate(scaled, start=1):
            sf.write(str(mix_dir / f"source{i}.wav"), src, args.sample_rate,
                     subtype="FLOAT")

        manifest = {
            "mixture": "mixture.wav",
            "num_speakers": args.num_speakers,
            "sample_rate": args.sample_rate,
            "seed": args.seed,
            "mixture_index": idx,
            "mixture_scale": mixture_scale,
            "num_samples": int(len(mixture)),
            "sources": [
                {
                    "file": f"source{i + 1}.wav",
                    "source_path": str(picked_paths[i]),
                    "speaker_id": speaker_ids[i],
                    "applied_gain": gains[i],
                }
                for i in range(args.num_speakers)
            ],
        }
        with open(mix_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        speakers_str = ", ".join(speaker_ids)
        print(f"Wrote {mix_dir}  ({args.num_speakers} speakers: "
              f"{speakers_str}, {len(mixture)} samples)")

    print(f"Done. Generated {args.num_mixtures} mixture(s) in {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
