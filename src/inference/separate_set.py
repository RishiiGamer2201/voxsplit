"""Batch SepFormer separation over a realized eval set for VoxSplit Phase 2.

Loads the pretrained SepFormer model ONCE and runs it over every mixture
folder whose source count matches --num-speakers, so the model output count
lines up with the references. Estimates are written as est1.wav ... estN.wav
per mixture. This is the batch counterpart to src/inference/separate.py and
uses the same speechbrain 1.1.0 API, including local_strategy=COPY which is
required for fetching on this Windows machine.

Example:
  python src/inference/separate_set.py \
      --eval-dir data/eval_set \
      --model speechbrain/sepformer-libri3mix \
      --num-speakers 3
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

MODEL_SR = 8000


def resolve_device(device: str) -> str:
    """Turn the requested device string into a concrete torch device."""
    if device == "auto":
        # SpeechBrain expects "<type>:<index>", so use "cuda:0" not "cuda".
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_mono(path: Path, target_sr: int = MODEL_SR) -> np.ndarray:
    """Load audio, downmix to mono, resample to target_sr, return float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=target_sr)
        mono = tensor.numpy()
    return mono.astype(np.float32)


def count_sources(mix_dir: Path) -> int:
    """Number of source*.wav reference files in a mixture folder."""
    return len(sorted(mix_dir.glob("source*.wav")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch separate an eval set with pretrained SepFormer.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"),
                        help="Directory holding <id>/mixture.wav and "
                             "<id>/source*.wav.")
    parser.add_argument("--model", default="speechbrain/sepformer-libri3mix",
                        help="Pretrained SepFormer model source.")
    parser.add_argument("--num-speakers", required=True, type=int,
                        help="Only process mixtures with exactly this many "
                             "source*.wav files.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Output root. Defaults to "
                             "data/eval_estimates/<model-basename>.")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda:0, or cpu.")
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    if not eval_dir.is_dir():
        print(f"Eval directory not found: {eval_dir}")
        return 1

    model_basename = args.model.split("/")[-1]
    if args.out_root is not None:
        out_root: Path = args.out_root
    else:
        out_root = Path("data/eval_estimates") / model_basename

    # Select mixture folders with a matching source count before loading the
    # model, so we can bail out early if there is nothing to do.
    mix_dirs = [d for d in sorted(eval_dir.iterdir()) if d.is_dir()]
    matching = [d for d in mix_dirs
                if count_sources(d) == args.num_speakers
                and (d / "mixture.wav").is_file()]
    if not matching:
        print(f"No mixtures with exactly {args.num_speakers} sources found "
              f"under {eval_dir}.")
        return 1
    print(f"Found {len(matching)} mixtures with {args.num_speakers} "
          f"sources to process.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Imported here so --help works without loading heavy modules.
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy

    savedir = Path("pretrained_models") / model_basename
    print(f"Loading model {args.model} (savedir={savedir}) ...")
    # COPY instead of the default SYMLINK: on Windows, symlinking the fetched
    # files needs admin or Developer Mode, which we do not assume.
    model = SepformerSeparation.from_hparams(
        source=args.model,
        savedir=str(savedir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    written_dirs: List[Path] = []
    for i, mix_dir in enumerate(matching, start=1):
        mixture_id = mix_dir.name
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        mix = torch.from_numpy(mono).unsqueeze(0).to(device)  # [1, time]
        with torch.no_grad():
            est = model.separate_batch(mix)  # [1, time, n_sources]
        est = est.cpu().numpy()[0]  # [time, n_sources]
        n_sources = est.shape[1]

        est_dir = out_root / mixture_id
        est_dir.mkdir(parents=True, exist_ok=True)
        for j in range(n_sources):
            sf.write(str(est_dir / f"est{j + 1}.wav"),
                     est[:, j].astype(np.float32), MODEL_SR, subtype="FLOAT")
        written_dirs.append(est_dir)
        print(f"[{i}/{len(matching)}] {mixture_id}: wrote {n_sources} "
              f"estimates to {est_dir}")

    print(f"Done. Wrote estimates for {len(written_dirs)} mixtures under "
          f"{out_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
