"""Augment a clean VoxSplit eval-set manifest with a NOISE or REVERB condition.

Reads a clean manifest produced by build_eval_set.py and writes a NEW manifest
that copies every top-level field, adds a "condition" tag, and records, per
mixture, the exact parameters needed to regenerate the degraded audio. The
input manifest is never modified. All random draws come from a single seeded
random.Random in a fixed order, so the same seed and the same corpus always
produce an identical output manifest.

NOISE: each mixture gets a chosen noise file (relative to --noise-dir), a
sample start offset, and a target SNR in dB. Only noise file lengths are read
(via soundfile.info); no full audio is loaded here.

REVERB: each mixture gets an explicit shoebox room (dimensions, rt60, mic
position, one source position per speaker) so realize_eval_set.py can
regenerate the exact room impulse responses with no RNG at realize time.

Examples:
  python src/data/make_conditions.py \
      --manifest data/eval_manifest.json --condition noise \
      --noise-dir data/wham_noise/tr \
      --out data/eval_manifest_noise.json --seed 0

  python src/data/make_conditions.py \
      --manifest data/eval_manifest.json --condition reverb \
      --out data/eval_manifest_reverb.json --seed 0
"""
import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import soundfile as sf

# Assumed typical mixture duration used only to choose a valid random start
# offset into a noise file. It matches the build_eval_set min-seconds default,
# so a typical WHAM noise file (usually several seconds long) leaves a valid
# range for the random start. If a noise file is shorter than this, the start
# is 0 and realize_eval_set.py tiles/loops the noise to the mixture length.
NOISE_SEGMENT_SECONDS = 4.0

# Geometry limits for the reverb shoebox rooms, in metres.
ROOM_XY_MIN = 3.0
ROOM_XY_MAX = 8.0
ROOM_Z_MIN = 2.5
ROOM_Z_MAX = 4.0
MIC_JITTER = 0.3          # +/- jitter of the mic about the room centre
WALL_MARGIN = 0.5         # minimum distance of a source from every wall
MIC_MARGIN = 0.5          # minimum distance of a source from the mic
MAX_SOURCE_TRIES = 1000   # bounded retry budget when placing a source


def find_noise_files(noise_dir: Path) -> List[Path]:
    """Return all .wav files under noise_dir, recursively, sorted.

    Sorting makes the candidate list deterministic so a seeded choice over it
    is reproducible.
    """
    files = [p for p in noise_dir.rglob("*")
             if p.is_file() and p.suffix.lower() == ".wav"]
    return sorted(files)


def noise_len_in_target_sr(path: Path, sample_rate: int) -> int:
    """Length of a noise file in samples after resampling to sample_rate.

    Reads only the header via soundfile.info, never the audio samples.
    """
    info = sf.info(str(path))
    if info.samplerate == sample_rate:
        return int(info.frames)
    return int(round(info.frames * sample_rate / float(info.samplerate)))


def to_relpath(path: Path, root: Path) -> str:
    """Path of path relative to root, using forward slashes."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def assign_noise(mixture: Dict, noise_files: List[Path], noise_dir: Path,
                 sample_rate: int, snr_min: float, snr_max: float,
                 rng: random.Random) -> Dict:
    """Draw a noise file, start offset, and SNR for one mixture.

    Fixed RNG draw order per mixture: noise file, then start offset, then
    snr_db. Keeping this order stable is what makes the manifest reproducible.
    """
    chosen = rng.choice(noise_files)
    needed = int(round(NOISE_SEGMENT_SECONDS * sample_rate))
    noise_len = noise_len_in_target_sr(chosen, sample_rate)
    max_start = noise_len - needed
    if max_start > 0:
        start = rng.randint(0, max_start)
    else:
        start = 0
    snr_db = rng.uniform(snr_min, snr_max)

    entry = dict(mixture)
    entry["noise"] = {
        "relpath": to_relpath(chosen, noise_dir),
        "start": int(start),
        "snr_db": float(snr_db),
    }
    return entry


def place_source(room_dim: List[float], mic_pos: List[float],
                 rng: random.Random) -> List[float]:
    """Sample one source position inside the room away from walls and mic.

    Draws x, y, z uniformly within the wall margin and retries (still using
    the seeded rng, so deterministic) until the point is at least MIC_MARGIN
    from the mic.
    """
    x_dim, y_dim, z_dim = room_dim
    for _ in range(MAX_SOURCE_TRIES):
        x = rng.uniform(WALL_MARGIN, x_dim - WALL_MARGIN)
        y = rng.uniform(WALL_MARGIN, y_dim - WALL_MARGIN)
        z = rng.uniform(WALL_MARGIN, z_dim - WALL_MARGIN)
        dist = math.sqrt((x - mic_pos[0]) ** 2 + (y - mic_pos[1]) ** 2
                         + (z - mic_pos[2]) ** 2)
        if dist >= MIC_MARGIN:
            return [float(x), float(y), float(z)]
    # Fallback: return the last draw even if slightly close to the mic. With
    # the configured margins this branch is effectively unreachable.
    return [float(x), float(y), float(z)]


def assign_reverb(mixture: Dict, rt60_min: float, rt60_max: float,
                  rng: random.Random) -> Dict:
    """Draw an explicit shoebox room for one mixture.

    Fixed RNG draw order per mixture: room x, room y, room z, rt60, mic
    jitter x/y/z, then each source position (which itself draws x/y/z, with
    bounded retries). Everything is recorded so realize regenerates the exact
    RIRs with no RNG.
    """
    x_dim = rng.uniform(ROOM_XY_MIN, ROOM_XY_MAX)
    y_dim = rng.uniform(ROOM_XY_MIN, ROOM_XY_MAX)
    z_dim = rng.uniform(ROOM_Z_MIN, ROOM_Z_MAX)
    room_dim = [float(x_dim), float(y_dim), float(z_dim)]

    rt60 = rng.uniform(rt60_min, rt60_max)

    mic_pos = [
        float(x_dim / 2.0 + rng.uniform(-MIC_JITTER, MIC_JITTER)),
        float(y_dim / 2.0 + rng.uniform(-MIC_JITTER, MIC_JITTER)),
        float(z_dim / 2.0 + rng.uniform(-MIC_JITTER, MIC_JITTER)),
    ]

    num_sources = len(mixture["sources"])
    source_pos = [place_source(room_dim, mic_pos, rng)
                  for _ in range(num_sources)]

    entry = dict(mixture)
    entry["reverb"] = {
        "room_dim": room_dim,
        "rt60": float(rt60),
        "mic_pos": mic_pos,
        "source_pos": source_pos,
    }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a condition-augmented (noise or reverb) copy of a "
                    "clean eval-set manifest.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/eval_manifest.json"),
                        help="Input clean manifest from build_eval_set.py.")
    parser.add_argument("--condition", required=True,
                        choices=["noise", "reverb"],
                        help="Degradation condition to add.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output condition-augmented manifest path.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for full reproducibility.")
    # Noise options.
    parser.add_argument("--noise-dir", type=Path, default=None,
                        help="Root of noise wav files (searched recursively "
                             "for .wav). Required for --condition noise.")
    parser.add_argument("--snr-db-min", type=float, default=0.0,
                        help="Minimum SNR in dB for the noise condition.")
    parser.add_argument("--snr-db-max", type=float, default=10.0,
                        help="Maximum SNR in dB for the noise condition.")
    # Reverb options.
    parser.add_argument("--rt60-min", type=float, default=0.2,
                        help="Minimum RT60 in seconds for the reverb "
                             "condition.")
    parser.add_argument("--rt60-max", type=float, default=0.6,
                        help="Maximum RT60 in seconds for the reverb "
                             "condition.")
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}")
        return 1

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("condition"):
        print(f"Input manifest already has condition "
              f"'{manifest['condition']}'; expected a clean manifest.")
        return 1

    sample_rate = int(manifest["sample_rate"])
    mixtures = manifest["mixtures"]
    rng = random.Random(args.seed)

    # Copy every top-level field, then override created/seed and add condition.
    out_manifest = dict(manifest)
    out_manifest["created"] = datetime.now().isoformat(timespec="seconds")
    out_manifest["condition"] = args.condition
    out_manifest["condition_seed"] = int(args.seed)

    if args.condition == "noise":
        if args.noise_dir is None:
            print("--noise-dir is required for --condition noise.")
            return 1
        if not args.noise_dir.is_dir():
            print(f"Noise directory not found: {args.noise_dir}")
            return 1
        if args.snr_db_min > args.snr_db_max:
            print("--snr-db-min must not exceed --snr-db-max.")
            return 1
        noise_files = find_noise_files(args.noise_dir)
        if not noise_files:
            print(f"No .wav noise files found under {args.noise_dir}")
            return 1
        print(f"Found {len(noise_files)} noise .wav files under "
              f"{args.noise_dir}")
        out_manifest["noise_dir"] = str(args.noise_dir)
        out_manifest["snr_db_range"] = [float(args.snr_db_min),
                                        float(args.snr_db_max)]
        new_mixtures = [
            assign_noise(mix, noise_files, args.noise_dir, sample_rate,
                         args.snr_db_min, args.snr_db_max, rng)
            for mix in mixtures
        ]
    else:  # reverb
        if args.rt60_min > args.rt60_max:
            print("--rt60-min must not exceed --rt60-max.")
            return 1
        out_manifest["rt60_range"] = [float(args.rt60_min),
                                      float(args.rt60_max)]
        new_mixtures = [
            assign_reverb(mix, args.rt60_min, args.rt60_max, rng)
            for mix in mixtures
        ]

    out_manifest["mixtures"] = new_mixtures

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_manifest, fh, indent=2)

    print("")
    print(f"Wrote {args.condition} manifest to {out_path}")
    print(f"Total mixtures: {len(new_mixtures)}")
    if args.condition == "noise":
        snrs = [m["noise"]["snr_db"] for m in new_mixtures]
        print(f"Noise dir: {args.noise_dir}")
        print(f"SNR dB range requested: "
              f"[{args.snr_db_min}, {args.snr_db_max}]")
        print(f"SNR dB assigned: min={min(snrs):.2f} max={max(snrs):.2f}")
    else:
        rt60s = [m["reverb"]["rt60"] for m in new_mixtures]
        print(f"RT60 range requested: [{args.rt60_min}, {args.rt60_max}]")
        print(f"RT60 assigned: min={min(rt60s):.2f} max={max(rt60s):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
