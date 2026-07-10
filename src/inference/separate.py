"""Run pretrained SepFormer separation on a single audio file.

Loads the input with soundfile, downmixes to mono, resamples to 8 kHz
(the rate the pretrained models expect), and writes one estimated source
track per detected speaker.

Example:
  python src/inference/separate.py data/mixtures/mixture_000/mixture.wav \
      --out-dir data/estimates/mixture_000
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

MODEL_SR = 8000


def resolve_device(device: str) -> str:
    """Turn the requested device string into a concrete torch device."""
    if device == "auto":
        # SpeechBrain parses run_opts device as "<type>:<index>", so use the
        # explicit "cuda:0" form rather than a bare "cuda".
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def load_mono_8k(path: Path, target_sr: int = MODEL_SR
                 ) -> "tuple[np.ndarray, int]":
    """Load audio, downmix to mono, resample to target_sr.

    Returns (mono_signal, original_sample_rate).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=target_sr)
        mono = tensor.numpy()
    return mono.astype(np.float32), sr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate speakers with pretrained SepFormer.")
    parser.add_argument("input", type=Path,
                        help="Input mixture wav file.")
    parser.add_argument("--model", default="speechbrain/sepformer-wsj03mix",
                        help="Pretrained SepFormer model source, for example "
                             "speechbrain/sepformer-wsj03mix, "
                             "speechbrain/sepformer-libri3mix, or "
                             "speechbrain/sepformer-wsj02mix (2 speakers).")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory to write estimated sources.")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, or cpu.")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}")
        return 1

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Imported here so that --help works without loading heavy modules.
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy

    savedir = Path("pretrained_models") / args.model.split("/")[-1]
    print(f"Loading model {args.model} (savedir={savedir}) ...")
    # COPY instead of the default SYMLINK: on Windows, symlinking the fetched
    # files needs admin or Developer Mode, which we do not assume.
    model = SepformerSeparation.from_hparams(
        source=args.model,
        savedir=str(savedir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )

    mono, orig_sr = load_mono_8k(args.input, MODEL_SR)
    print(f"Loaded {args.input} (orig sr={orig_sr}, "
          f"{len(mono)} samples at {MODEL_SR} Hz).")

    mix = torch.from_numpy(mono).unsqueeze(0).to(device)  # [1, time]
    with torch.no_grad():
        est = model.separate_batch(mix)  # [1, time, n_sources]
    est = est.cpu().numpy()[0]  # [time, n_sources]
    n_sources = est.shape[1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n_sources):
        out_path = args.out_dir / f"est{i + 1}.wav"
        sf.write(str(out_path), est[:, i].astype(np.float32),
                 MODEL_SR, subtype="FLOAT")
        written.append(out_path)

    print(f"Wrote {n_sources} estimated source(s):")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
