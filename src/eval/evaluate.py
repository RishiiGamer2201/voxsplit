"""Evaluate separation quality for one mixture: SI-SDR, SI-SDRi, PESQ, STOI.

Loads the mixture, reference sources, and estimated sources, aligns them to
a common sample rate and length, finds the best estimate-to-reference
permutation by SI-SDR, and reports per-speaker and mean metrics. One
summary row is appended to a CSV for tracking across experiments.

Example:
  python src/eval/evaluate.py \
      --mixture data/mixtures/mixture_000/mixture.wav \
      --refs "data/mixtures/mixture_000/source*.wav" \
      --ests "data/estimates/mixture_000/est*.wav"
"""
import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# Allow "import metrics" whether run as a script or from elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
    best_permutation,
    compute_pesq,
    compute_stoi,
    si_sdr,
)


def gather_files(pattern_or_dir: str, exts=(".wav", ".flac")) -> List[Path]:
    """Resolve a directory or a glob pattern to a sorted list of files."""
    p = Path(pattern_or_dir)
    if p.is_dir():
        files = [f for f in sorted(p.iterdir())
                 if f.suffix.lower() in exts]
    else:
        files = [Path(f) for f in sorted(glob.glob(pattern_or_dir))]
    return files


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Load audio, downmix to mono, resample to sample_rate, return float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=sample_rate)
        mono = tensor.numpy()
    return mono.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score separated sources against references.")
    parser.add_argument("--mixture", required=True, type=Path,
                        help="Mixture wav file.")
    parser.add_argument("--refs", required=True,
                        help="Directory or glob of reference source wavs.")
    parser.add_argument("--ests", required=True,
                        help="Directory or glob of estimated source wavs.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Sample rate for scoring in Hz.")
    parser.add_argument("--csv", type=Path,
                        default=Path("experiments/phase1_results.csv"),
                        help="CSV file to append the summary row to.")
    args = parser.parse_args()

    if not args.mixture.is_file():
        print(f"Mixture file not found: {args.mixture}")
        return 1

    ref_paths = gather_files(args.refs)
    est_paths = gather_files(args.ests)
    if not ref_paths:
        print(f"No reference files matched: {args.refs}")
        return 1
    if not est_paths:
        print(f"No estimate files matched: {args.ests}")
        return 1
    if len(ref_paths) != len(est_paths):
        print(f"Count mismatch: {len(ref_paths)} references vs "
              f"{len(est_paths)} estimates. They must be equal.")
        return 1

    sr = args.sample_rate
    mixture = load_mono(args.mixture, sr)
    references = [load_mono(p, sr) for p in ref_paths]
    estimates = [load_mono(p, sr) for p in est_paths]

    # Truncate everything to a common minimum length.
    min_len = min([len(mixture)]
                  + [len(x) for x in references]
                  + [len(x) for x in estimates])
    mixture = mixture[:min_len]
    references = [x[:min_len] for x in references]
    estimates = [x[:min_len] for x in estimates]

    perm, pairs, mean_matched = best_permutation(estimates, references)

    k = len(references)
    per_si_sdr: List[float] = []
    per_si_sdri: List[float] = []
    per_pesq: List[float] = []
    per_stoi: List[float] = []

    for est_idx, ref_idx in pairs:
        ref = references[ref_idx]
        est = estimates[est_idx]
        matched = si_sdr(est, ref)
        baseline = si_sdr(mixture, ref)
        per_si_sdr.append(matched)
        per_si_sdri.append(matched - baseline)
        per_pesq.append(compute_pesq(ref, est, sr))
        per_stoi.append(compute_stoi(ref, est, sr))

    mean_si_sdr = float(np.mean(per_si_sdr))
    mean_si_sdri = float(np.mean(per_si_sdri))
    mean_pesq = float(np.nanmean(per_pesq))
    mean_stoi = float(np.nanmean(per_stoi))

    print("")
    print(f"Mixture: {args.mixture}")
    print(f"Speakers: {k}   Sample rate: {sr} Hz   "
          f"Length: {min_len} samples")
    print("")
    header = (f"{'spk':>3}  {'est':>3}  {'ref':>3}  {'SI-SDR':>8}  "
              f"{'SI-SDRi':>8}  {'PESQ':>6}  {'STOI':>6}")
    print(header)
    print("-" * len(header))
    for j, (est_idx, ref_idx) in enumerate(pairs):
        print(f"{j:>3}  {est_idx:>3}  {ref_idx:>3}  "
              f"{per_si_sdr[j]:>8.2f}  {per_si_sdri[j]:>8.2f}  "
              f"{per_pesq[j]:>6.3f}  {per_stoi[j]:>6.3f}")
    print("-" * len(header))
    print(f"{'mean':>3}  {'':>3}  {'':>3}  "
          f"{mean_si_sdr:>8.2f}  {mean_si_sdri:>8.2f}  "
          f"{mean_pesq:>6.3f}  {mean_stoi:>6.3f}")
    print("")

    # Append a summary row to the CSV, creating it with a header if missing.
    csv_path: Path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["timestamp", "mixture", "num_speakers",
                             "mean_si_sdri", "mean_pesq", "mean_stoi"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            str(args.mixture),
            k,
            f"{mean_si_sdri:.4f}",
            f"{mean_pesq:.4f}",
            f"{mean_stoi:.4f}",
        ])
    print(f"Appended summary row to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
