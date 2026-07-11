"""Batch inference with a fine-tuned OR-PIT SepFormer checkpoint.

Loads the wsj02mix architecture, overwrites its weights with a checkpoint from
train_orpit.py, and separates every mixture in an eval-set level whose source
count matches the model output count. For a 2-output OR-PIT model this is the
2-speaker level (the two heads are one speaker and the "rest", which for two
speakers is just the other speaker), so it can be scored directly against the
2-speaker references. Higher speaker counts need the Phase 4 recursion.

Example:
  python src/inference/separate_orpit_set.py \
      --eval-dir data/eval_set --ckpt checkpoints/orpit/ckpt_step6000.pt \
      --num-speakers 2 --out-root data/eval_estimates/orpit_step6000
"""
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# Reuse the exact forward and the lazy-module fix from the trainer.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_orpit import separate_forward, neutralize_lazy_modules  # noqa: E402

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
        description="Separate an eval-set level with a fine-tuned OR-PIT "
                    "SepFormer checkpoint.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"),
                        help="Directory holding <id>/mixture.wav and "
                             "<id>/source*.wav.")
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Checkpoint .pt from train_orpit.py.")
    parser.add_argument("--init-model", default="speechbrain/sepformer-wsj02mix",
                        help="Architecture to load the checkpoint into.")
    parser.add_argument("--num-speakers", required=True, type=int,
                        help="Only process mixtures with this many sources "
                             "(use 2 for a 2-output OR-PIT model).")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Output root for estimates.")
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

    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy

    savedir = Path("pretrained_models") / args.init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=args.init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()

    ckpt = torch.load(str(args.ckpt), map_location=device)
    encoder, masknet, decoder = sep.mods.encoder, sep.mods.masknet, sep.mods.decoder
    encoder.load_state_dict(ckpt["encoder"])
    masknet.load_state_dict(ckpt["masknet"])
    decoder.load_state_dict(ckpt["decoder"])
    for m in (encoder, masknet, decoder):
        m.eval()
    print(f"Loaded checkpoint {args.ckpt} (step {ckpt.get('step', '?')}).")

    args.out_root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for i, mix_dir in enumerate(matching, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        mix = torch.from_numpy(mono).unsqueeze(0).to(device)
        with torch.no_grad():
            est = separate_forward(encoder, masknet, decoder, mix)  # [1,T2,2]
        est = est.cpu().numpy()[0]  # [T2, 2]
        est_dir = args.out_root / mix_dir.name
        est_dir.mkdir(parents=True, exist_ok=True)
        for j in range(est.shape[1]):
            sf.write(str(est_dir / f"est{j + 1}.wav"),
                     est[:, j].astype(np.float32), MODEL_SR, subtype="FLOAT")
        written.append(est_dir)
        print(f"[{i}/{len(matching)}] {mix_dir.name}: wrote {est.shape[1]} "
              f"estimates")

    print(f"Done. Wrote estimates for {len(written)} mixtures under "
          f"{args.out_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
