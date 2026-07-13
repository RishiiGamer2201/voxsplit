"""Blind recursive OR-PIT separation for UNKNOWN speaker count (Phase 4).

Unlike separate_orpit_recursive_set.py (which uses the references as an oracle
to pick heads and a known count for depth), this uses NO references. The Phase
4 count/stop classifier (count_classifier.py) drives every decision:

  1. If P(>=2 speakers) on the input mixture is below the threshold, the input
     is a single speaker: emit it unchanged.
  2. Otherwise recurse: run the 2-head OR-PIT model, and of the two heads
     recurse on the one with the higher P(>=2 speakers) (the residual "rest")
     while keeping the other (the extracted single speaker).
  3. Stop when the residual's P(>=2 speakers) drops below the threshold (it is
     the last single speaker) or a max-speaker cap is hit.

The predicted speaker count is simply the number of emitted estimates. The
script separates every mixture folder under --eval-dir with no knowledge of
its true count, writes est1..estK.wav, and logs a per-mixture prediction row
(true vs predicted count) plus a confusion summary for count accuracy.

Run this file directly for a self-test that needs no real data or model:
  python src/inference/separate_recursive_blind.py --self-test

Example:
  python src/inference/separate_recursive_blind.py \
      --eval-dir data/eval_set \
      --orpit-ckpt checkpoints/orpit/ckpt_step20000.pt \
      --clf-ckpt checkpoints/count_clf/ckpt_step8000.pt \
      --out-root data/eval_estimates/orpit_blind \
      --pred-csv experiments/phase4_count_predictions.csv --tag orpit20k_clf8k
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from train_orpit import separate_forward, neutralize_lazy_modules  # noqa: E402
from count_classifier import SpeakerCountCNN, MAX_SPEAKERS  # noqa: E402

MODEL_SR = 8000


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_mono(path: Path, target_sr: int = MODEL_SR) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=target_sr)
        mono = tensor.numpy()
    return mono.astype(np.float32)


def blind_recursive_separate(
        mixture: np.ndarray,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        prob_multi_fn: Callable[[np.ndarray], float],
        threshold: float = 0.5,
        max_speakers: int = MAX_SPEAKERS) -> List[np.ndarray]:
    """Recurse with a classifier deciding head selection and stopping.

    forward_fn: [T] -> [2, T2] the two OR-PIT heads.
    prob_multi_fn: [T] -> P(>=2 speakers) for a waveform.
    Returns a list of 1..max_speakers estimate arrays; its length is the
    predicted speaker count.
    """
    mixture = np.asarray(mixture, dtype=np.float32)
    if prob_multi_fn(mixture) < threshold:
        return [mixture]

    residual = mixture
    estimates: List[np.ndarray] = []
    for _ in range(max_speakers - 1):
        heads = forward_fn(residual)                 # [2, T2]
        p0 = prob_multi_fn(heads[0])
        p1 = prob_multi_fn(heads[1])
        # Recurse on the head more likely to hold >=2 speakers; keep the other.
        if p0 <= p1:
            keep, rest, p_rest = heads[0], heads[1], p1
        else:
            keep, rest, p_rest = heads[1], heads[0], p0
        estimates.append(np.asarray(keep, dtype=np.float32))
        if p_rest < threshold:
            estimates.append(np.asarray(rest, dtype=np.float32))
            return estimates
        residual = np.asarray(rest, dtype=np.float32)

    # Hit the cap: the final residual is the last speaker.
    estimates.append(residual)
    return estimates


def count_sources(mix_dir: Path) -> int:
    return len(sorted(mix_dir.glob("source*.wav")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Blind recursive OR-PIT separation (unknown count).")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"))
    parser.add_argument("--orpit-ckpt", type=Path,
                        help="OR-PIT checkpoint from train_orpit.py.")
    parser.add_argument("--clf-ckpt", type=Path,
                        help="Count classifier checkpoint.")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-wsj02mix")
    parser.add_argument("--out-root", type=Path,
                        help="Output root for estimates.")
    parser.add_argument("--pred-csv", type=Path,
                        default=Path("experiments/phase4_count_predictions.csv"),
                        help="CSV to append per-mixture count predictions to.")
    parser.add_argument("--tag", default="",
                        help="Free-text label recorded in the prediction CSV.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(>=2 speakers) decision threshold.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if args.orpit_ckpt is None or args.clf_ckpt is None or args.out_root is None:
        print("--orpit-ckpt, --clf-ckpt and --out-root are required.")
        return 1
    if not args.eval_dir.is_dir():
        print(f"Eval directory not found: {args.eval_dir}")
        return 1

    mix_dirs = [d for d in sorted(args.eval_dir.iterdir())
                if d.is_dir() and (d / "mixture.wav").is_file()]
    if not mix_dirs:
        print(f"No mixtures under {args.eval_dir}.")
        return 1
    print(f"Found {len(mix_dirs)} mixtures (count unknown to the model).")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # OR-PIT model.
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    savedir = Path("pretrained_models") / args.init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=args.init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()
    oc = torch.load(str(args.orpit_ckpt), map_location=device)
    encoder, masknet, decoder = (sep.mods.encoder, sep.mods.masknet,
                                 sep.mods.decoder)
    encoder.load_state_dict(oc["encoder"])
    masknet.load_state_dict(oc["masknet"])
    decoder.load_state_dict(oc["decoder"])
    for m in (encoder, masknet, decoder):
        m.eval()

    # Count classifier.
    cc = torch.load(str(args.clf_ckpt), map_location=device)
    clf = SpeakerCountCNN(num_classes=cc.get("num_classes", MAX_SPEAKERS))
    clf.load_state_dict(cc["model"])
    clf.to(device).eval()
    print(f"Loaded OR-PIT step {oc.get('step','?')} and classifier step "
          f"{cc.get('step','?')}.")

    def forward_fn(signal: np.ndarray) -> np.ndarray:
        mix = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
        mix = mix.to(device)
        with torch.no_grad():
            est = separate_forward(encoder, masknet, decoder, mix)  # [1,T2,2]
        return est.cpu().numpy()[0].T  # [2, T2]

    def prob_multi_fn(signal: np.ndarray) -> float:
        wav = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
        return float(clf.prob_multi(wav.to(device))[0])

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, int, int]] = []
    for i, mix_dir in enumerate(mix_dirs, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        estimates = blind_recursive_separate(
            mono, forward_fn, prob_multi_fn, threshold=args.threshold)
        pred_k = len(estimates)
        true_k = count_sources(mix_dir)
        est_dir = args.out_root / mix_dir.name
        est_dir.mkdir(parents=True, exist_ok=True)
        for j, est in enumerate(estimates, start=1):
            sf.write(str(est_dir / f"est{j}.wav"),
                     est.astype(np.float32), MODEL_SR, subtype="FLOAT")
        rows.append((mix_dir.name, true_k, pred_k))
        print(f"[{i}/{len(mix_dirs)}] {mix_dir.name}: true={true_k} "
              f"pred={pred_k}")

    _report_and_log(rows, args.pred_csv, args.tag)
    print(f"Done. Estimates under {args.out_root}.")
    return 0


def _report_and_log(rows, pred_csv: Path, tag: str) -> None:
    """Print a count-accuracy summary and append per-mixture rows to a CSV."""
    total = len(rows)
    correct = sum(1 for _, tk, pk in rows if tk == pk)
    print("")
    print(f"Count accuracy: {correct}/{total} = {correct / max(total,1):.3f}")
    # Per-true-K accuracy and mean absolute error.
    by_k = {}
    for _, tk, pk in rows:
        b = by_k.setdefault(tk, [0, 0, 0])
        b[0] += 1
        b[1] += int(tk == pk)
        b[2] += abs(tk - pk)
    print(f"{'trueK':>5}  {'n':>3}  {'acc':>5}  {'MAE':>5}")
    for k in sorted(by_k):
        n, ok, ae = by_k[k]
        print(f"{k:>5}  {n:>3}  {ok / n:>5.2f}  {ae / n:>5.2f}")

    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not pred_csv.exists() or pred_csv.stat().st_size == 0
    with open(pred_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["tag", "mixture_id", "true_k", "pred_k"])
        for mid, tk, pk in rows:
            w.writerow([tag, mid, tk, pk])
    print(f"Appended {total} prediction rows to {pred_csv}")


def _self_test() -> int:
    """Blind recursion returns the right count with an ideal model+classifier.

    Ideal forward: given a residual that is the sum of some tones, extract the
    lowest-frequency present tone and return (one, rest). Ideal classifier:
    P(>=2) = 1 if two or more tones are present, else 0.
    """
    sr = MODEL_SR
    n = 2 * sr
    t = np.arange(n) / float(sr)
    freqs = [200.0, 350.0, 500.0, 650.0, 800.0]

    def amp_at(sig, f):
        return float(2.0 * np.dot(sig, np.sin(2 * np.pi * f * t)) / n)

    def present(sig):
        return [i for i in range(len(freqs)) if abs(amp_at(sig, freqs[i])) > 0.3]

    def prob_multi_fn(sig):
        return 1.0 if len(present(sig)) >= 2 else 0.0

    for true_k in (1, 2, 3, 4, 5):
        sources = [np.sin(2 * np.pi * freqs[i] * t).astype(np.float32)
                   for i in range(true_k)]
        mixture = np.sum(sources, axis=0).astype(np.float32)

        def forward_fn(residual):
            p = present(residual)
            one_idx = min(p)  # deterministic pick
            one = sources[one_idx]
            rest = (np.sum([sources[i] for i in p if i != one_idx], axis=0)
                    if len(p) > 1 else np.zeros(n, np.float32))
            return np.stack([one, rest]).astype(np.float32)

        ests = blind_recursive_separate(mixture, forward_fn, prob_multi_fn)
        assert len(ests) == true_k, (true_k, len(ests))
        print(f"true_k={true_k}: predicted {len(ests)} speakers")

    print("All blind_recursive_separate self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
