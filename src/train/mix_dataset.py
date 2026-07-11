"""On-the-fly training-mixture dataset for OR-PIT fine-tuning (Phase 3).

Builds fully overlapped 2- or 3-speaker mixtures on the fly from a directory
of single-speaker source utterances (for example LibriSpeech). Speakers are
grouped by the same rule used in make_mixture.find_sources, so this reuses the
Phase 1 scanning logic. Each item samples K speakers, loads one utterance
each, random-crops (or pads) to a fixed segment length, applies per-source
random rms-relative gains, and sums to the mixture.

Because K varies per item, sources cannot be stacked across a batch. The
collate function keeps sources as a list of per-item tensors.

Run this file directly for a self-test on synthetic audio (no real data):
  python src/train/mix_dataset.py
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

# Reuse find_sources from the Phase 1 mixer (cross-folder import like
# src/eval/evaluate.py does).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mixing"))
from make_mixture import find_sources  # noqa: E402


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Load audio, downmix to mono, resample to sample_rate, return float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        tensor = torch.from_numpy(np.ascontiguousarray(mono))
        tensor = AF.resample(tensor, orig_freq=sr, new_freq=sample_rate)
        mono = tensor.numpy()
    return mono.astype(np.float32)


def rms(signal: np.ndarray) -> float:
    """Root mean square level of a signal, with a small floor."""
    return float(np.sqrt(np.mean(signal ** 2) + 1e-12))


def crop_or_pad(signal: np.ndarray, length: int,
                rng: np.random.Generator) -> np.ndarray:
    """Random-crop signal to length, or right-pad with zeros if too short."""
    n = len(signal)
    if n >= length:
        start = int(rng.integers(0, n - length + 1))
        return signal[start:start + length]
    out = np.zeros(length, dtype=np.float32)
    out[:n] = signal
    return out


class MixDataset(Dataset):
    """On-the-fly 2- or 3-speaker overlapped mixtures for OR-PIT training.

    Each __getitem__ is deterministic given (seed, index): the same index
    always yields the same mixture, so a fixed index sequence is reproducible.
    """

    def __init__(self, source_dir: Path, sample_rate: int = 8000,
                 segment_seconds: float = 3.0,
                 speaker_counts=(2, 3), seed: int = 0,
                 length: int = 100000, min_seconds: float = 2.0) -> None:
        self.source_dir = Path(source_dir)
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_len = int(round(self.segment_seconds * self.sample_rate))
        self.speaker_counts = tuple(int(k) for k in speaker_counts)
        self.seed = int(seed)
        self.length = int(length)

        self.by_speaker: Dict[str, List[Path]] = find_sources(
            self.source_dir, min_seconds, self.sample_rate)
        self.speakers: List[str] = sorted(self.by_speaker.keys())
        max_k = max(self.speaker_counts)
        if len(self.speakers) < max_k:
            raise ValueError(
                f"Need at least {max_k} distinct speakers but only found "
                f"{len(self.speakers)} in {self.source_dir}.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, object]:
        # Per-item RNG seeded from (seed, index) for reproducibility.
        rng = np.random.default_rng((self.seed, int(index)))

        k = int(rng.choice(self.speaker_counts))
        spk_idx = rng.choice(len(self.speakers), size=k, replace=False)
        chosen = [self.speakers[i] for i in spk_idx]

        raw: List[np.ndarray] = []
        for spk in chosen:
            files = self.by_speaker[spk]
            path = files[int(rng.integers(0, len(files)))]
            sig = load_mono(path, self.sample_rate)
            raw.append(crop_or_pad(sig, self.segment_len, rng))

        # Per-source gains: source 0 is the reference at gain 1.0; each other
        # source gets a [-5, 5] dB rms-relative offset (matches make_mixture).
        ref_rms = rms(raw[0])
        scaled: List[np.ndarray] = []
        for i, src in enumerate(raw):
            if i == 0:
                gain = 1.0
            else:
                offset_db = float(rng.uniform(-5.0, 5.0))
                target_rms = ref_rms * (10.0 ** (offset_db / 20.0))
                gain = target_rms / rms(src)
            scaled.append((src * gain).astype(np.float32))

        sources = np.stack(scaled, axis=0)  # [K, T]
        mixture = sources.sum(axis=0).astype(np.float32)  # [T]

        return {
            "mixture": torch.from_numpy(mixture),          # [T]
            "sources": torch.from_numpy(sources),          # [K, T]
            "num_speakers": k,
        }


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate items whose per-item source counts differ.

    Mixtures all share the fixed segment length, so they stack into [B, T].
    Sources stay a list of B tensors, each [K_b, T], because K_b varies.
    """
    mixtures = torch.stack([item["mixture"] for item in batch], dim=0)
    sources = [item["sources"] for item in batch]
    num_speakers = [int(item["num_speakers"]) for item in batch]
    return {
        "mixture": mixtures,        # [B, T]
        "sources": sources,         # list of B tensors, each [K_b, T]
        "num_speakers": num_speakers,
    }


def _self_test() -> int:
    """Self-test on a tiny synthetic source tree (no real data needed)."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="voxsplit_mixds_"))
    sr = 8000
    try:
        rng = np.random.default_rng(0)
        n = int(4.0 * sr)
        t = np.arange(n) / float(sr)
        # Four speakers, LibriSpeech-style filenames so find_sources groups
        # them by the prefix before the first '-'.
        for s in range(4):
            speaker = str(100 + s)
            spk_dir = tmp / speaker
            spk_dir.mkdir(parents=True, exist_ok=True)
            for f in range(2):
                freq = 150.0 + 40.0 * s + 7.0 * f
                sig = 0.3 * np.sin(2.0 * np.pi * freq * t)
                sig = sig + 0.01 * rng.standard_normal(n)
                name = f"{speaker}-0-{f:04d}.flac"
                sf.write(str(spk_dir / name), sig.astype(np.float32), sr,
                         subtype="PCM_16")

        ds = MixDataset(tmp, sample_rate=sr, segment_seconds=3.0,
                        speaker_counts=(2, 3), seed=0, length=50)
        seg = int(3.0 * sr)

        item = ds[0]
        assert item["mixture"].shape == (seg,), item["mixture"].shape
        k = item["num_speakers"]
        assert item["sources"].shape == (k, seg), item["sources"].shape
        assert k in (2, 3), k
        # Mixture must equal the sum of its sources.
        recon = item["sources"].sum(dim=0)
        max_err = float((recon - item["mixture"]).abs().max())
        print(f"item0 K={k} seg={seg} sum-vs-mix max abs err={max_err:.2e}")
        assert max_err < 1e-5, max_err

        # Determinism: same index yields identical output.
        again = ds[0]
        assert torch.equal(item["mixture"], again["mixture"])
        assert torch.equal(item["sources"], again["sources"])
        print("determinism check passed (index 0 reproducible).")

        # Collate over a mixed-K batch.
        batch = collate([ds[0], ds[1], ds[2], ds[3]])
        assert batch["mixture"].shape == (4, seg), batch["mixture"].shape
        assert len(batch["sources"]) == 4
        assert len(batch["num_speakers"]) == 4
        ks = batch["num_speakers"]
        print(f"batch Ks={ks} mixture={tuple(batch['mixture'].shape)}")
        for src, kb in zip(batch["sources"], ks):
            assert src.shape == (kb, seg), (src.shape, kb)
        print("collate check passed.")

        print("All mix_dataset self-tests passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-test the OR-PIT on-the-fly mixture dataset.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic self-test (default action).")
    parser.parse_args()
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
