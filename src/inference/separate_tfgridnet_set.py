"""Batch inference over an eval-set level with a trained TF-GridNet checkpoint.

Rebuilds the TF-GridNet from the config stored in the checkpoint, loads its
weights, and separates every mixture folder whose source count equals
--num-speakers, writing est1..estN.wav at 8 kHz. Mirrors
separate_finetuned_set.py's convtasnet path.

Example:
  python src/inference/separate_tfgridnet_set.py \
      --eval-dir data/eval_set --ckpt checkpoints/tfgridnet/ckpt_step8000.pt \
      --num-speakers 3 --out-root data/eval_estimates/tfgridnet_step8000
"""
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from tfgridnet import TFGridNet  # noqa: E402

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


def count_sources(mix_dir: Path) -> int:
    return len(sorted(mix_dir.glob("source*.wav")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate an eval-set level with a trained TF-GridNet.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"))
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Checkpoint .pt from train_tfgridnet.py.")
    parser.add_argument("--num-speakers", required=True, type=int,
                        help="Only process mixtures with this many sources.")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        print(f"Eval directory not found: {args.eval_dir}")
        return 1
    if not args.ckpt.is_file():
        print(f"Checkpoint not found: {args.ckpt}")
        return 1

    mix_dirs = [d for d in sorted(args.eval_dir.iterdir()) if d.is_dir()]
    matching = [d for d in mix_dirs
                if count_sources(d) == args.num_speakers
                and (d / "mixture.wav").is_file()]
    if not matching:
        print(f"No mixtures with {args.num_speakers} sources under "
              f"{args.eval_dir}.")
        return 1
    print(f"Found {len(matching)} mixtures with {args.num_speakers} sources.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    ckpt = torch.load(str(args.ckpt), map_location=device)
    cfg = ckpt.get("config", {"num_spks": args.num_speakers})
    model = TFGridNet(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded TF-GridNet checkpoint {args.ckpt} "
          f"(step {ckpt.get('step', '?')}).")

    args.out_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for i, mix_dir in enumerate(matching, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        mix = torch.from_numpy(mono).unsqueeze(0).to(device)
        with torch.no_grad():
            est = model(mix)              # [1, N, T]
        est = est.cpu().numpy()[0]        # [N, T]
        est_dir = args.out_root / mix_dir.name
        est_dir.mkdir(parents=True, exist_ok=True)
        for j in range(est.shape[0]):
            sf.write(str(est_dir / f"est{j + 1}.wav"),
                     est[j].astype(np.float32), MODEL_SR, subtype="FLOAT")
        written.append(est_dir)
        print(f"[{i}/{len(matching)}] {mix_dir.name}: wrote {est.shape[0]} "
              f"estimates")

    print(f"Done. Wrote estimates for {len(written)} mixtures under "
          f"{args.out_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
