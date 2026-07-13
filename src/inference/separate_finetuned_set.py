"""Batch inference over an eval-set level from a fine-tuned checkpoint.

Supports two architectures:
  - sepformer:   load the N-output SepFormer architecture from --init-model,
                 overwrite encoder/masknet/decoder with a train_pit.py
                 checkpoint, and run the generalized sepformer_forward.
  - convtasnet:  build a torchaudio ConvTasNet(num_sources=N) and load a
                 train_convtasnet.py checkpoint.

For either arch, every mixture folder whose source count equals --num-speakers
is separated and est1..estN.wav are written at 8 kHz float32. Mirrors
separate_orpit_set.py for loading mixtures and writing outputs.

Examples:
  python src/inference/separate_finetuned_set.py \
      --eval-dir data/eval_set --arch sepformer \
      --ckpt checkpoints/pit_libri3mix/ckpt_step1000.pt \
      --init-model speechbrain/sepformer-libri3mix --num-speakers 3 \
      --out-root data/eval_estimates/pit_libri3mix_step1000

  python src/inference/separate_finetuned_set.py \
      --eval-dir data/eval_set --arch convtasnet \
      --ckpt checkpoints/convtasnet/ckpt_step1000.pt --num-speakers 2 \
      --out-root data/eval_estimates/convtasnet_step1000
"""
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# Reuse the generalized forward and the lazy-module fix from the trainers.
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


def count_sources(mix_dir: Path) -> int:
    return len(sorted(mix_dir.glob("source*.wav")))


def separate_sepformer(matching, args, device) -> int:
    """Load the SepFormer arch, overwrite from ckpt, and separate mixtures."""
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy

    savedir = Path("pretrained_models") / args.init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=args.init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()

    ckpt = torch.load(str(args.ckpt), map_location=device)
    encoder = sep.mods.encoder
    masknet = sep.mods.masknet
    decoder = sep.mods.decoder
    # If the checkpoint has more heads than the init arch (4/5-spk models
    # warm-started from 3-head libri3mix), expand the masknet to match.
    expand_masknet_heads(masknet, args.num_speakers)
    encoder.load_state_dict(ckpt["encoder"])
    masknet.load_state_dict(ckpt["masknet"])
    decoder.load_state_dict(ckpt["decoder"])
    for m in (encoder, masknet, decoder):
        m.eval()
    print(f"Loaded sepformer checkpoint {args.ckpt} "
          f"(step {ckpt.get('step', '?')}).")

    written: List[Path] = []
    for i, mix_dir in enumerate(matching, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        mix = torch.from_numpy(mono).unsqueeze(0).to(device)
        with torch.no_grad():
            est = sepformer_forward(
                encoder, masknet, decoder, mix, args.num_speakers)
        est = est.cpu().numpy()[0]  # [T2, N]
        _write_estimates(args.out_root, mix_dir, est)
        written.append(args.out_root / mix_dir.name)
        print(f"[{i}/{len(matching)}] {mix_dir.name}: wrote "
              f"{est.shape[1]} estimates")

    print(f"Done. Wrote estimates for {len(written)} mixtures under "
          f"{args.out_root}.")
    return 0


def separate_convtasnet(matching, args, device) -> int:
    """Build ConvTasNet, load its state_dict, and separate mixtures."""
    from torchaudio.models import ConvTasNet

    model = ConvTasNet(num_sources=args.num_speakers).to(device)
    ckpt = torch.load(str(args.ckpt), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded convtasnet checkpoint {args.ckpt} "
          f"(step {ckpt.get('step', '?')}).")

    written: List[Path] = []
    for i, mix_dir in enumerate(matching, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        mix = torch.from_numpy(mono).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            est = model(mix)               # [1, N, T]
        est = est.cpu().numpy()[0].T       # [T, N]
        _write_estimates(args.out_root, mix_dir, est)
        written.append(args.out_root / mix_dir.name)
        print(f"[{i}/{len(matching)}] {mix_dir.name}: wrote "
              f"{est.shape[1]} estimates")

    print(f"Done. Wrote estimates for {len(written)} mixtures under "
          f"{args.out_root}.")
    return 0


def _write_estimates(out_root: Path, mix_dir: Path, est: np.ndarray) -> None:
    """Write est[:, j] as est{j+1}.wav under out_root/<mix id>/."""
    est_dir = out_root / mix_dir.name
    est_dir.mkdir(parents=True, exist_ok=True)
    for j in range(est.shape[1]):
        sf.write(str(est_dir / f"est{j + 1}.wav"),
                 est[:, j].astype(np.float32), MODEL_SR, subtype="FLOAT")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate an eval-set level with a fine-tuned checkpoint "
                    "(sepformer or convtasnet).")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"),
                        help="Directory holding <id>/mixture.wav and "
                             "<id>/source*.wav.")
    parser.add_argument("--arch", required=True,
                        choices=["sepformer", "convtasnet"],
                        help="Checkpoint architecture.")
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Checkpoint .pt from train_pit.py or "
                             "train_convtasnet.py.")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-libri3mix",
                        help="SepFormer architecture to load the checkpoint "
                             "into (sepformer arch only).")
    parser.add_argument("--num-speakers", required=True, type=int,
                        help="Only process mixtures with this many sources; "
                             "also the model head count.")
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

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.arch == "sepformer":
        return separate_sepformer(matching, args, device)
    return separate_convtasnet(matching, args, device)


if __name__ == "__main__":
    raise SystemExit(main())
