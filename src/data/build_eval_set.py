"""Build a frozen evaluation-set manifest for VoxSplit Phase 2.

Deterministically selects, for each requested speaker count K, a fixed set
of K-speaker mixtures from a LibriSpeech subset. For every source it loads
the audio only long enough to compute the same rms-based linear gain that
make_mixture.py would apply, then records that gain, the source path
relative to the LibriSpeech root, and the speaker id. No audio is written:
the manifest plus the source corpus fully determine the eval set, and
realize_eval_set.py turns the manifest into wav files.

Example:
  python src/data/build_eval_set.py \
      --source-dir data/LibriSpeech/test-clean \
      --librispeech-root data/LibriSpeech \
      --out data/eval_manifest.json \
      --speaker-counts "2,3,4,5" --per-count 20 --seed 0
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Reuse the Phase 1 mixing helpers. Insert the mixing dir on sys.path the
# same way src/eval/evaluate.py inserts its own dir to import metrics.
_MIXING_DIR = Path(__file__).resolve().parent.parent / "mixing"
sys.path.insert(0, str(_MIXING_DIR))
from make_mixture import (  # noqa: E402
    find_sources,
    load_mono_resampled,
    rms,
)


def parse_speaker_counts(text: str) -> List[int]:
    """Parse a comma separated list such as '2,3,4,5' into sorted ints."""
    counts: List[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        counts.append(int(piece))
    return counts


def to_relpath(path: Path, root: Path) -> str:
    """Path of a source file relative to root, using forward slashes."""
    rel = path.resolve().relative_to(root.resolve())
    return rel.as_posix()


def build_mixture_entry(by_speaker: Dict[str, List[Path]], num_speakers: int,
                        sample_rate: int, librispeech_root: Path,
                        rng: random.Random, mixture_id: str) -> Dict:
    """Sample K distinct speakers and record their gains and relpaths.

    Mirrors make_mixture.make_one_mixture and build_mixture: source 0 has
    gain 1.0, and each later source draws a loudness offset in [-5, 5] dB
    relative to source 0, converted to a linear gain via the rms rule. The
    same rng call ordering (sample speakers, choose a file per speaker,
    then draw offsets for sources 1..K-1) is used so the manifest is
    reproducible from the seed alone.
    """
    speakers = list(by_speaker.keys())
    chosen_speakers = rng.sample(speakers, num_speakers)

    picked_paths: List[Path] = []
    speaker_ids: List[str] = []
    signals = []
    for spk in chosen_speakers:
        path = rng.choice(by_speaker[spk])
        picked_paths.append(path)
        speaker_ids.append(spk)
        signals.append(load_mono_resampled(path, sample_rate))

    ref_rms = rms(signals[0])
    sources: List[Dict] = []
    for i, sig in enumerate(signals):
        if i == 0:
            gain = 1.0
        else:
            offset_db = rng.uniform(-5.0, 5.0)
            target_rms = ref_rms * (10.0 ** (offset_db / 20.0))
            gain = target_rms / rms(sig)
        sources.append({
            "relpath": to_relpath(picked_paths[i], librispeech_root),
            "speaker_id": speaker_ids[i],
            "gain": float(gain),
        })

    return {
        "id": mixture_id,
        "num_speakers": num_speakers,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a frozen eval-set manifest (JSON only, no audio).")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="LibriSpeech subset root, for example "
                             "data/LibriSpeech/test-clean.")
    parser.add_argument("--librispeech-root", type=Path,
                        default=Path("data/LibriSpeech"),
                        help="Base directory that stored relpaths are "
                             "recorded against.")
    parser.add_argument("--out", type=Path,
                        default=Path("data/eval_manifest.json"),
                        help="Output manifest JSON path.")
    parser.add_argument("--speaker-counts", default="2,3,4,5",
                        help="Comma separated speaker counts, for example "
                             "'2,3,4,5'.")
    parser.add_argument("--per-count", type=int, default=20,
                        help="Number of mixtures per speaker count.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Target sample rate in Hz for gain computation.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for full reproducibility.")
    parser.add_argument("--min-seconds", type=float, default=4.0,
                        help="Skip source files shorter than this.")
    args = parser.parse_args()

    source_dir: Path = args.source_dir
    if not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}")
        return 1

    speaker_counts = parse_speaker_counts(args.speaker_counts)
    if not speaker_counts:
        print("No valid speaker counts parsed from --speaker-counts.")
        return 1

    print(f"Scanning sources under {source_dir} ...")
    by_speaker = find_sources(source_dir, args.min_seconds, args.sample_rate)
    total_files = sum(len(v) for v in by_speaker.values())
    num_speakers_avail = len(by_speaker)
    print(f"Found {total_files} usable files across "
          f"{num_speakers_avail} speakers "
          f"(min-seconds={args.min_seconds}).")

    rng = random.Random(args.seed)
    mixtures: List[Dict] = []
    per_k_written: Dict[int, int] = {}

    for k in speaker_counts:
        if num_speakers_avail < k:
            print(f"Skipping K={k}: only {num_speakers_avail} distinct "
                  f"speakers available, need at least {k}.")
            continue
        for idx in range(args.per_count):
            mixture_id = f"spk{k}_{idx:03d}"
            entry = build_mixture_entry(
                by_speaker, k, args.sample_rate,
                args.librispeech_root, rng, mixture_id)
            mixtures.append(entry)
        per_k_written[k] = args.per_count

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_subset": source_dir.name,
        "librispeech_root": args.librispeech_root.as_posix(),
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "normalize_peak": 0.9,
        "speaker_counts": speaker_counts,
        "per_count": args.per_count,
        "mixtures": mixtures,
    }

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("")
    print(f"Wrote manifest to {out_path}")
    print(f"Total mixtures: {len(mixtures)}")
    for k in speaker_counts:
        if k in per_k_written:
            print(f"  K={k}: {per_k_written[k]} mixtures")
        else:
            print(f"  K={k}: skipped (not enough speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
