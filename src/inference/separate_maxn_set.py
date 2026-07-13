"""Max-N plus silence detection for unknown count (VoxSplit Phase 4, secondary).

The alternative to recursive OR-PIT: always run the largest fixed-N model (the
5-speaker uPIT SepFormer from train_pit.py), which emits N=5 output tracks, then
DROP the channels that carry no real speaker. On a mixture of K<5 speakers the
extra 5-K heads come out near-silent or as low-energy residue, so gating by
per-channel energy recovers the count. Kept-channel count is the prediction.

Channel gating: keep a channel if its RMS is within --drop-db of the loudest
channel's RMS (relative), so the threshold adapts to overall level. This is the
silence/activity detector; a relative rule beats an absolute floor because
mixtures are peak-normalized to different absolute levels.

Runs over every eval-set mixture with no knowledge of the true count, writes the
kept channels as est1..estK.wav, and reports count accuracy plus a per-mixture
prediction CSV, mirroring separate_recursive_blind.py so the two unknown-count
routes are directly comparable.

Run this file directly for a self-test (gating logic, no data or model):
  python src/inference/separate_maxn_set.py --self-test

Example:
  python src/inference/separate_maxn_set.py \
      --eval-dir data/eval_set --ckpt checkpoints/pit5_libri/ckpt_step8000.pt \
      --num-heads 5 --drop-db 15 \
      --out-root data/eval_estimates/maxn5 \
      --pred-csv experiments/phase4_count_predictions.csv --tag maxn5_drop15
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_orpit import neutralize_lazy_modules  # noqa: E402
from train_pit import sepformer_forward, expand_masknet_heads  # noqa: E402

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


def active_channels(estimates: np.ndarray, drop_db: float) -> List[int]:
    """Indices of channels kept as active by relative-energy gating.

    estimates: [T, N]. Keep channel i if its RMS is at most drop_db below the
    loudest channel's RMS. The loudest channel is always kept, so at least one
    speaker is returned.
    """
    rms = np.sqrt(np.mean(estimates.astype(np.float64) ** 2, axis=0) + 1e-12)
    peak = float(rms.max())
    floor = peak * (10.0 ** (-drop_db / 20.0))
    keep = [i for i in range(estimates.shape[1]) if rms[i] >= floor]
    return keep or [int(np.argmax(rms))]


def count_sources(mix_dir: Path) -> int:
    return len(sorted(mix_dir.glob("source*.wav")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Max-N + silence detection for unknown speaker count.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"))
    parser.add_argument("--ckpt", type=Path,
                        help="Fixed-N uPIT SepFormer checkpoint (train_pit.py).")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-libri3mix",
                        help="Architecture to load and expand to --num-heads.")
    parser.add_argument("--num-heads", type=int, default=5,
                        help="Output head count of the max-N model.")
    parser.add_argument("--drop-db", type=float, default=15.0,
                        help="Drop channels more than this many dB below the "
                             "loudest channel.")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--pred-csv", type=Path,
                        default=Path("experiments/phase4_count_predictions.csv"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if args.ckpt is None or args.out_root is None:
        print("--ckpt and --out-root are required (or use --self-test).")
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

    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    savedir = Path("pretrained_models") / args.init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=args.init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()
    encoder, masknet, decoder = (sep.mods.encoder, sep.mods.masknet,
                                 sep.mods.decoder)
    expand_masknet_heads(masknet, args.num_heads)
    ckpt = torch.load(str(args.ckpt), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    masknet.load_state_dict(ckpt["masknet"])
    decoder.load_state_dict(ckpt["decoder"])
    for m in (encoder, masknet, decoder):
        m.eval()
    print(f"Loaded {args.num_heads}-head model {args.ckpt} "
          f"(step {ckpt.get('step','?')}).")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, mix_dir in enumerate(mix_dirs, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        mix = torch.from_numpy(mono).unsqueeze(0).to(device)
        with torch.no_grad():
            est = sepformer_forward(encoder, masknet, decoder, mix,
                                    args.num_heads)
        est = est.cpu().numpy()[0]                    # [T, num_heads]
        keep = active_channels(est, args.drop_db)
        pred_k = len(keep)
        true_k = count_sources(mix_dir)
        est_dir = args.out_root / mix_dir.name
        est_dir.mkdir(parents=True, exist_ok=True)
        for j, ch in enumerate(keep, start=1):
            sf.write(str(est_dir / f"est{j}.wav"),
                     est[:, ch].astype(np.float32), MODEL_SR, subtype="FLOAT")
        rows.append((mix_dir.name, true_k, pred_k))
        print(f"[{i}/{len(mix_dirs)}] {mix_dir.name}: true={true_k} "
              f"pred={pred_k}")

    _report_and_log(rows, args.pred_csv, args.tag)
    print(f"Done. Estimates under {args.out_root}.")
    return 0


def _report_and_log(rows, pred_csv: Path, tag: str) -> None:
    total = len(rows)
    correct = sum(1 for _, tk, pk in rows if tk == pk)
    print("")
    print(f"Count accuracy: {correct}/{total} = {correct / max(total,1):.3f}")
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
    """Energy gating keeps loud channels and drops near-silent ones."""
    rng = np.random.default_rng(0)
    t = 8000
    # 5 channels: 3 active (unit-ish RMS), 2 near-silent.
    chans = [rng.standard_normal(t) for _ in range(3)]
    chans += [1e-3 * rng.standard_normal(t), 1e-4 * rng.standard_normal(t)]
    est = np.stack(chans, axis=1).astype(np.float32)   # [T, 5]

    keep = active_channels(est, drop_db=15.0)
    print(f"kept {len(keep)} of 5 channels: {keep}")
    assert keep == [0, 1, 2], keep

    # A very permissive threshold keeps everything.
    assert len(active_channels(est, drop_db=200.0)) == 5
    # A very strict threshold keeps only the loudest.
    assert len(active_channels(est, drop_db=0.0)) == 1
    # All-equal channels are all kept.
    equal = np.ones((t, 4), dtype=np.float32)
    assert len(active_channels(equal, drop_db=15.0)) == 4
    print("All max-N gating self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
