"""Speech-enhancement post-filter for separated streams (VoxSplit Phase 5).

Optional polish: pass each separated track through a single-channel speech
enhancer to clean residual bleed and artifacts left by separation. Uses
SpeechBrain's MetricGAN+ (SpectralMaskEnhancement), which is pure PyTorch and
runs in the MAIN voxsplit env.

NOTE: the ClearerVoice FRCRN / MossFormer2-SE models named in the plan live in
the separate `clearvoice` env, whose numba/llvmlite.dll is blocked by a Windows
Application Control policy on this machine, so they cannot run here. MetricGAN+
is an equivalent modern SE post-filter with no numba dependency.

The model runs at 16 kHz; 8 kHz tracks are upsampled, enhanced, and written back
at 8 kHz so evaluate_set.py can score a before/after ablation.

Pre-fetch the weights ONCE (the SpeechBrain HF fetch can hang on Windows):
  python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('speechbrain/metricgan-plus-voicebank', \
    local_dir='pretrained_models/metricgan-plus')"

Run this file directly for a self-test (no real data; identity-ish check):
  python src/inference/enhance.py --self-test

Example:
  python src/inference/enhance.py \
      --ests-root data/eval_estimates/orpit_step20000 \
      --out-root  data/eval_estimates/orpit_step20000_enh
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

OUT_SR = 8000
MODEL_SR = 16000


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def enhance_signal(enhancer, signal_8k: np.ndarray, device: str) -> np.ndarray:
    """Enhance an 8 kHz mono signal, returning 8 kHz mono float32.

    Upsamples to the model's 16 kHz, runs the spectral-mask enhancer, and
    downsamples back so the result stays comparable to the 8 kHz references.
    """
    x = torch.from_numpy(np.ascontiguousarray(signal_8k)).float()
    x16 = AF.resample(x, OUT_SR, MODEL_SR).unsqueeze(0).to(device)  # [1, T]
    lengths = torch.ones(1, device=device)
    with torch.no_grad():
        y16 = enhancer.enhance_batch(x16, lengths=lengths)         # [1, T]
    y8 = AF.resample(y16.squeeze(0).cpu(), MODEL_SR, OUT_SR)
    return y8.numpy().astype(np.float32)


def load_enhancer(model_dir: Path, device: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from train_orpit import neutralize_lazy_modules  # noqa: E402
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    from speechbrain.utils.fetching import LocalStrategy
    # SpeechBrain registers lazy-import proxies (e.g. k2_fsa) that raise on
    # access and break from_hparams; drop them, same fix as the trainer.
    neutralize_lazy_modules()
    source = str(model_dir) if model_dir.is_dir() else \
        "speechbrain/metricgan-plus-voicebank"
    enhancer = SpectralMaskEnhancement.from_hparams(
        source=source, savedir=str(model_dir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()
    return enhancer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhance separated tracks with MetricGAN+.")
    parser.add_argument("--ests-root", type=Path,
                        help="Root of <id>/est*.wav (or speaker*.wav).")
    parser.add_argument("--out-root", type=Path,
                        help="Output root, same structure, enhanced.")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("pretrained_models/metricgan-plus"),
                        help="Local MetricGAN+ dir (pre-fetched).")
    parser.add_argument("--pattern", default="est*.wav",
                        help="Glob for track files inside each mixture dir.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not (args.ests_root and args.out_root):
        print("--ests-root and --out-root are required (or --self-test).")
        return 1
    if not args.ests_root.is_dir():
        print(f"Estimates root not found: {args.ests_root}")
        return 1

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    enhancer = load_enhancer(args.model_dir, device)

    mix_dirs = [d for d in sorted(args.ests_root.iterdir()) if d.is_dir()]
    total = 0
    for mix_dir in mix_dirs:
        tracks = sorted(mix_dir.glob(args.pattern))
        if not tracks:
            continue
        out_dir = args.out_root / mix_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for track in tracks:
            data, sr = sf.read(str(track), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if sr != OUT_SR:
                mono = AF.resample(torch.from_numpy(mono), sr,
                                   OUT_SR).numpy()
            enhanced = enhance_signal(enhancer, mono, device)
            sf.write(str(out_dir / track.name), enhanced, OUT_SR,
                     subtype="FLOAT")
            total += 1
        print(f"{mix_dir.name}: enhanced {len(tracks)} track(s)")

    print(f"Done. Enhanced {total} tracks under {args.out_root}.")
    return 0


def _self_test() -> int:
    """Enhancer runs and returns a same-length 8 kHz signal on a clean tone."""
    device = resolve_device("auto")
    enhancer = load_enhancer(Path("pretrained_models/metricgan-plus"), device)
    sr = OUT_SR
    t = np.arange(2 * sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = enhance_signal(enhancer, tone, device)
    assert out.ndim == 1 and abs(len(out) - len(tone)) <= sr // 10, len(out)
    assert np.isfinite(out).all()
    print(f"enhanced tone: in {len(tone)} -> out {len(out)} samples, finite ok")
    print("Enhancer self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
