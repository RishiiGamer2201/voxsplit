"""Recursive OR-PIT separation for 3+ speakers (VoxSplit Phase 3 -> Phase 4).

The OR-PIT SepFormer has two output heads: one speaker and "the rest" (the sum
of all remaining speakers). For K > 2 a single forward pass cannot return K
tracks, so this recurses (Takahashi et al. 2019, arXiv:1904.03065): extract one
speaker, feed the residual "rest" head back into the same network, and repeat
until every speaker is out. With a known speaker count K it runs exactly K-1
passes and emits K estimates.

Head selection at each step (which head is the extracted speaker vs the residual
to recurse on) uses an ORACLE here: the head that best matches a remaining
reference is kept, the other is recursed. This measures the separation-quality
CEILING of the recursion, isolating it from the blind stop/selection classifier,
which is the Phase 4 deliverable. Labelled clearly in the eval CSV via --tag.

Estimates are written as est1..estK.wav at 8 kHz; score them with
src/eval/evaluate_set.py (which does its own best-permutation matching).

Run this file directly for a self-test that needs no real data or model:
  python src/inference/separate_orpit_recursive_set.py --self-test

Example:
  python src/inference/separate_orpit_recursive_set.py \
      --eval-dir data/eval_set --ckpt checkpoints/orpit/ckpt_step6000.pt \
      --speaker-counts 3,4,5 \
      --out-root data/eval_estimates/orpit_step6000_recursive
"""
import argparse
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# Reuse the exact forward and lazy-module fix from the trainer, and SI-SDR from
# the metrics module (same sys.path pattern used across the repo).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from train_orpit import separate_forward, neutralize_lazy_modules  # noqa: E402
from metrics import si_sdr  # noqa: E402

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


def _best_ref_score(head: np.ndarray, references: List[np.ndarray],
                    remaining: List[int]) -> "tuple[float, int]":
    """Best SI-SDR of a head over the still-unassigned references.

    Returns (best_score, ref_index). Truncates to the shorter length so a
    head whose length differs slightly from the reference still scores.
    """
    best_score = -np.inf
    best_ref = remaining[0]
    for ref_idx in remaining:
        score = si_sdr(head, references[ref_idx])
        if score > best_score:
            best_score = score
            best_ref = ref_idx
    return float(best_score), best_ref


def recursive_separate(mixture: np.ndarray, references: List[np.ndarray],
                       forward_fn: Callable[[np.ndarray], np.ndarray],
                       ) -> List[np.ndarray]:
    """Recursively extract K speakers with oracle head selection.

    mixture: [T]. references: K clean sources (used ONLY to pick which head is
    the extracted speaker vs the residual). forward_fn: [T] -> [2, T2] the two
    OR-PIT heads. Runs K-1 passes and returns K estimate arrays.
    """
    k = len(references)
    if k < 2:
        # One speaker: the mixture is already the estimate.
        return [np.asarray(mixture, dtype=np.float32)]

    residual = np.asarray(mixture, dtype=np.float32)
    remaining = list(range(k))
    estimates: List[np.ndarray] = []

    for step in range(k - 1):
        heads = forward_fn(residual)          # [2, T2]
        if step < k - 2:
            # Keep whichever head best matches a remaining reference (the clean
            # single speaker); recurse on the other head (the "rest").
            score0, ref0 = _best_ref_score(heads[0], references, remaining)
            score1, ref1 = _best_ref_score(heads[1], references, remaining)
            if score0 >= score1:
                keep, rest, matched_ref = heads[0], heads[1], ref0
            else:
                keep, rest, matched_ref = heads[1], heads[0], ref1
            estimates.append(np.asarray(keep, dtype=np.float32))
            remaining.remove(matched_ref)
            residual = np.asarray(rest, dtype=np.float32)
        else:
            # Last pass: two speakers remain, both heads are final estimates.
            estimates.append(np.asarray(heads[0], dtype=np.float32))
            estimates.append(np.asarray(heads[1], dtype=np.float32))

    return estimates


def count_sources(mix_dir: Path) -> int:
    return len(sorted(mix_dir.glob("source*.wav")))


def parse_counts(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursive OR-PIT separation for 3+ speakers.")
    parser.add_argument("--eval-dir", type=Path, default=Path("data/eval_set"),
                        help="Directory holding <id>/mixture.wav and "
                             "<id>/source*.wav.")
    parser.add_argument("--ckpt", type=Path,
                        help="OR-PIT checkpoint .pt from train_orpit.py.")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-wsj02mix",
                        help="Architecture to load the checkpoint into.")
    parser.add_argument("--speaker-counts", default="3,4,5",
                        help="Comma-separated speaker counts to process.")
    parser.add_argument("--out-root", type=Path,
                        help="Output root for estimates.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic self-test (no data or model).")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if args.ckpt is None or args.out_root is None:
        print("--ckpt and --out-root are required (or use --self-test).")
        return 1
    if not args.eval_dir.is_dir():
        print(f"Eval directory not found: {args.eval_dir}")
        return 1
    if not args.ckpt.is_file():
        print(f"Checkpoint not found: {args.ckpt}")
        return 1

    counts = set(parse_counts(args.speaker_counts))
    mix_dirs = [d for d in sorted(args.eval_dir.iterdir()) if d.is_dir()]
    matching = [d for d in mix_dirs
                if count_sources(d) in counts
                and (d / "mixture.wav").is_file()]
    if not matching:
        print(f"No mixtures with speaker counts {sorted(counts)} under "
              f"{args.eval_dir}.")
        return 1
    print(f"Found {len(matching)} mixtures with counts {sorted(counts)}.")

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
    encoder, masknet, decoder = (sep.mods.encoder, sep.mods.masknet,
                                 sep.mods.decoder)
    encoder.load_state_dict(ckpt["encoder"])
    masknet.load_state_dict(ckpt["masknet"])
    decoder.load_state_dict(ckpt["decoder"])
    for m in (encoder, masknet, decoder):
        m.eval()
    print(f"Loaded checkpoint {args.ckpt} (step {ckpt.get('step', '?')}).")

    def model_forward(signal: np.ndarray) -> np.ndarray:
        mix = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
        mix = mix.to(device)
        with torch.no_grad():
            est = separate_forward(encoder, masknet, decoder, mix)  # [1,T2,2]
        return est.cpu().numpy()[0].T  # [2, T2]

    args.out_root.mkdir(parents=True, exist_ok=True)
    for i, mix_dir in enumerate(matching, start=1):
        mono = load_mono(mix_dir / "mixture.wav", MODEL_SR)
        refs = [load_mono(p, MODEL_SR)
                for p in sorted(mix_dir.glob("source*.wav"))]
        estimates = recursive_separate(mono, refs, model_forward)
        est_dir = args.out_root / mix_dir.name
        est_dir.mkdir(parents=True, exist_ok=True)
        for j, est in enumerate(estimates, start=1):
            sf.write(str(est_dir / f"est{j}.wav"),
                     est.astype(np.float32), MODEL_SR, subtype="FLOAT")
        print(f"[{i}/{len(matching)}] {mix_dir.name}: K={len(refs)}, wrote "
              f"{len(estimates)} estimates")

    print(f"Done. Wrote estimates for {len(matching)} mixtures under "
          f"{args.out_root}.")
    return 0


def _self_test() -> int:
    """Recursion recovers all K sources given an ideal one-and-rest forward.

    Uses distinct pure tones as sources so an oracle forward can detect which
    tones are present in a residual, extract one, and return (one, rest).
    """
    sr = MODEL_SR
    n = 2 * sr
    t = np.arange(n) / float(sr)
    freqs = [200.0, 350.0, 500.0, 650.0, 800.0]

    def make_sources(k: int) -> List[np.ndarray]:
        return [np.sin(2.0 * np.pi * freqs[i] * t).astype(np.float32)
                for i in range(k)]

    def amp_at(signal: np.ndarray, freq: float) -> float:
        # Correlate with the unit tone to read its amplitude in the signal.
        tone = np.sin(2.0 * np.pi * freq * t)
        return float(2.0 * np.dot(signal, tone) / n)

    for k in (2, 3, 4, 5):
        sources = make_sources(k)
        mixture = np.sum(sources, axis=0).astype(np.float32)

        def ideal_forward(residual: np.ndarray) -> np.ndarray:
            present = [i for i in range(k)
                       if abs(amp_at(residual, freqs[i])) > 0.3]
            # Extract the present tone with the largest residual amplitude.
            one_idx = max(present, key=lambda i: abs(amp_at(residual, freqs[i])))
            one = sources[one_idx]
            rest = np.sum([sources[i] for i in present if i != one_idx],
                          axis=0) if len(present) > 1 else np.zeros(n, np.float32)
            heads = np.stack([one, rest]).astype(np.float32)
            # Shuffle head order so oracle selection is actually exercised.
            if one_idx % 2 == 1:
                heads = heads[::-1].copy()
            return heads

        estimates = recursive_separate(mixture, sources, ideal_forward)
        assert len(estimates) == k, (k, len(estimates))
        # Every source should be matched by some estimate at high SI-SDR.
        for src in sources:
            best = max(si_sdr(est, src) for est in estimates)
            assert best > 30.0, (k, best)
        print(f"K={k}: recovered {k} sources, min-best SI-SDR ok")

    print("All recursive_separate self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
