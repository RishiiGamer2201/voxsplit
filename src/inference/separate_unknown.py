"""VoxSplit Phase 4 entry point: separate one file with UNKNOWN speaker count.

Given a single audio file with an unknown number of overlapping speakers, this
loads the converged OR-PIT SepFormer and the Phase 4 count/stop classifier,
runs blind recursive separation (no reference, no count supplied), and writes
one estimated track per detected speaker. The number of output files is the
model's own estimate of the speaker count.

Example:
  python src/inference/separate_unknown.py meeting.wav \
      --orpit-ckpt checkpoints/orpit/ckpt_step20000.pt \
      --clf-ckpt checkpoints/count_clf/ckpt_step8000.pt \
      --out-dir out/meeting
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from separate_recursive_blind import (  # noqa: E402
    blind_recursive_separate, load_mono, resolve_device, MODEL_SR)
from train_orpit import separate_forward, neutralize_lazy_modules  # noqa: E402
from count_classifier import SpeakerCountCNN, MAX_SPEAKERS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate one file with unknown speaker count (Phase 4).")
    parser.add_argument("input", type=Path, help="Input mixture wav file.")
    parser.add_argument("--orpit-ckpt", required=True, type=Path)
    parser.add_argument("--clf-ckpt", required=True, type=Path)
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-wsj02mix")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(>=2 speakers) decision threshold.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}")
        return 1

    device = resolve_device(args.device)
    print(f"Using device: {device}")

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

    cc = torch.load(str(args.clf_ckpt), map_location=device)
    clf = SpeakerCountCNN(num_classes=cc.get("num_classes", MAX_SPEAKERS))
    clf.load_state_dict(cc["model"])
    clf.to(device).eval()

    def forward_fn(signal: np.ndarray) -> np.ndarray:
        mix = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
        with torch.no_grad():
            est = separate_forward(encoder, masknet, decoder, mix.to(device))
        return est.cpu().numpy()[0].T

    def prob_multi_fn(signal: np.ndarray) -> float:
        wav = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
        return float(clf.prob_multi(wav.to(device))[0])

    mono = load_mono(args.input, MODEL_SR)
    estimates = blind_recursive_separate(
        mono, forward_fn, prob_multi_fn, threshold=args.threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for j, est in enumerate(estimates, start=1):
        sf.write(str(args.out_dir / f"speaker{j}.wav"),
                 est.astype(np.float32), MODEL_SR, subtype="FLOAT")
    print(f"Detected {len(estimates)} speaker(s); wrote "
          f"{len(estimates)} track(s) to {args.out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
