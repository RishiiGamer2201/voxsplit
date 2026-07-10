"""Batch scorer for a VoxSplit Phase 2 eval set, aggregated by speaker count.

For every mixture id present in both the eval set (references and mixture)
and the estimates directory, and whose source and estimate counts match,
this loads the mixture, references, and estimates, finds the best
estimate-to-reference permutation by SI-SDR, and computes per-speaker
SI-SDR, SI-SDRi (matched minus mixture-vs-reference baseline), PESQ, and
STOI. Results are aggregated by number of speakers K, printed as a per-K
table, and appended one row per (tag, K) to a CSV.

Example:
  python src/eval/evaluate_set.py \
      --eval-dir data/eval_set \
      --ests-root data/eval_estimates/sepformer-libri3mix \
      --tag sepformer-libri3mix
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# Allow "import metrics" whether run as a script or from elsewhere, matching
# the sys.path pattern in src/eval/evaluate.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
    best_permutation,
    compute_pesq,
    compute_stoi,
    si_sdr,
)


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Load audio, downmix to mono, resample to sample_rate, return float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=sample_rate)
        mono = tensor.numpy()
    return mono.astype(np.float32)


def score_mixture(mix_dir: Path, est_dir: Path, sample_rate: int
                  ) -> "tuple[int, float, float, float, float]":
    """Score one mixture. Returns (K, si_sdr, si_sdri, pesq, stoi) means."""
    ref_paths = sorted(mix_dir.glob("source*.wav"))
    est_paths = sorted(est_dir.glob("est*.wav"))

    mixture = load_mono(mix_dir / "mixture.wav", sample_rate)
    references = [load_mono(p, sample_rate) for p in ref_paths]
    estimates = [load_mono(p, sample_rate) for p in est_paths]

    min_len = min([len(mixture)]
                  + [len(x) for x in references]
                  + [len(x) for x in estimates])
    mixture = mixture[:min_len]
    references = [x[:min_len] for x in references]
    estimates = [x[:min_len] for x in estimates]

    _, pairs, _ = best_permutation(estimates, references)

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
        per_pesq.append(compute_pesq(ref, est, sample_rate))
        per_stoi.append(compute_stoi(ref, est, sample_rate))

    return (
        len(references),
        float(np.mean(per_si_sdr)),
        float(np.mean(per_si_sdri)),
        float(np.nanmean(per_pesq)),
        float(np.nanmean(per_stoi)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score an eval set and aggregate metrics by speaker "
                    "count.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"),
                        help="Directory holding <id>/source*.wav and "
                             "<id>/mixture.wav.")
    parser.add_argument("--ests-root", required=True, type=Path,
                        help="Directory holding <id>/est*.wav.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Sample rate for scoring in Hz.")
    parser.add_argument("--csv", type=Path,
                        default=Path("experiments/eval_set_results.csv"),
                        help="CSV file to append per-K summary rows to.")
    parser.add_argument("--tag", default="",
                        help="Free-text label for this run recorded in the "
                             "CSV.")
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    ests_root: Path = args.ests_root
    if not eval_dir.is_dir():
        print(f"Eval directory not found: {eval_dir}")
        return 1
    if not ests_root.is_dir():
        print(f"Estimates directory not found: {ests_root}")
        return 1

    sr = args.sample_rate

    # Ids present in both the eval set and the estimates.
    eval_ids = {d.name for d in eval_dir.iterdir()
                if d.is_dir() and (d / "mixture.wav").is_file()}
    est_ids = {d.name for d in ests_root.iterdir() if d.is_dir()}
    common_ids = sorted(eval_ids & est_ids)

    # Accumulate per-K lists of per-mixture means.
    by_k: Dict[int, Dict[str, List[float]]] = {}
    skipped = 0
    scored = 0
    for mixture_id in common_ids:
        mix_dir = eval_dir / mixture_id
        est_dir = ests_root / mixture_id
        n_ref = len(sorted(mix_dir.glob("source*.wav")))
        n_est = len(sorted(est_dir.glob("est*.wav")))
        if n_ref == 0 or n_est == 0 or n_ref != n_est:
            skipped += 1
            continue

        k, m_si_sdr, m_si_sdri, m_pesq, m_stoi = score_mixture(
            mix_dir, est_dir, sr)
        bucket = by_k.setdefault(k, {"si_sdr": [], "si_sdri": [],
                                     "pesq": [], "stoi": []})
        bucket["si_sdr"].append(m_si_sdr)
        bucket["si_sdri"].append(m_si_sdri)
        bucket["pesq"].append(m_pesq)
        bucket["stoi"].append(m_stoi)
        scored += 1

    print("")
    print(f"Scored {scored} mixtures, skipped {skipped} "
          f"(source/estimate count mismatch or empty).")
    print("")
    header = (f"{'K':>3}  {'mixtures':>8}  {'SI-SDR':>8}  {'SI-SDRi':>8}  "
              f"{'PESQ':>6}  {'STOI':>6}")
    print(header)
    print("-" * len(header))

    csv_rows = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for k in sorted(by_k.keys()):
        bucket = by_k[k]
        n = len(bucket["si_sdr"])
        mean_si_sdr = float(np.nanmean(bucket["si_sdr"]))
        mean_si_sdri = float(np.nanmean(bucket["si_sdri"]))
        mean_pesq = float(np.nanmean(bucket["pesq"]))
        mean_stoi = float(np.nanmean(bucket["stoi"]))
        print(f"{k:>3}  {n:>8}  {mean_si_sdr:>8.2f}  {mean_si_sdri:>8.2f}  "
              f"{mean_pesq:>6.3f}  {mean_stoi:>6.3f}")
        csv_rows.append([
            timestamp,
            args.tag,
            k,
            n,
            f"{mean_si_sdr:.4f}",
            f"{mean_si_sdri:.4f}",
            f"{mean_pesq:.4f}",
            f"{mean_stoi:.4f}",
        ])
    print("-" * len(header))

    if not csv_rows:
        print("No mixtures scored; nothing appended to the CSV.")
        return 0

    csv_path: Path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["timestamp", "tag", "num_speakers",
                             "num_mixtures", "mean_si_sdr", "mean_si_sdri",
                             "mean_pesq", "mean_stoi"])
        for row in csv_rows:
            writer.writerow(row)
    print("")
    print(f"Appended {len(csv_rows)} row(s) to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
