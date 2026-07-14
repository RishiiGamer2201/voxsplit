"""On-the-fly noise/reverb augmentation for robustness fine-tuning (Phase 5).

Wraps a MixDataset so each item is randomly left clean or degraded with additive
WHAM noise and/or room reverberation, matching the Phase 2 eval conventions:

  noise:  mixture = clean_mix + noise scaled to a random SNR; references stay
          CLEAN (the model learns to separate through noise, not to denoise a
          target).
  reverb: every source is convolved with a room impulse response, the mixture
          is their sum, and the references become the REVERBERANT sources (the
          model separates reverberant speech, as the Phase 2 reverb set scores).

Room impulse responses are expensive to synthesize, so a fixed pool is built
once at construction and sampled per item, which keeps augmentation cheap enough
for training. Noise is streamed from a directory of .wav files (e.g. WHAM).

Run this file directly for a self-test on synthetic sources (no real data):
  python src/train/augment.py
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_dataset import MixDataset  # noqa: E402


def build_rir_pool(size: int, sample_rate: int, rng: np.random.Generator,
                   rt60_range=(0.2, 0.6)) -> List[np.ndarray]:
    """Pre-synthesize a pool of single-channel room impulse responses.

    Uses pyroomacoustics shoebox rooms with random geometry and RT60, mirroring
    make_conditions.py so training reverb matches the eval reverb distribution.
    """
    import pyroomacoustics as pra
    pool: List[np.ndarray] = []
    for _ in range(size):
        x = float(rng.uniform(3.0, 8.0))
        y = float(rng.uniform(3.0, 8.0))
        z = float(rng.uniform(2.5, 4.0))
        rt60 = float(rng.uniform(*rt60_range))
        try:
            absorption, max_order = pra.inverse_sabine(rt60, [x, y, z])
            room = pra.ShoeBox([x, y, z], fs=sample_rate,
                               materials=pra.Material(absorption),
                               max_order=max_order)
            mic = [x / 2, y / 2, z / 2]
            src = [float(rng.uniform(0.5, x - 0.5)),
                   float(rng.uniform(0.5, y - 0.5)),
                   float(rng.uniform(0.5, z - 0.5))]
            room.add_source(src)
            room.add_microphone(mic)
            room.compute_rir()
            pool.append(np.asarray(room.rir[0][0], dtype=np.float32))
        except Exception:
            continue
    if not pool:
        pool.append(np.array([1.0], dtype=np.float32))  # identity fallback
    return pool


def find_noise_files(noise_dir: Path) -> List[Path]:
    return sorted(p for p in noise_dir.rglob("*") if p.suffix.lower() == ".wav")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


class AugMixDataset(Dataset):
    """Randomly clean/noise/reverb-degraded wrapper over a MixDataset."""

    def __init__(self, base: MixDataset, noise_dir: Optional[Path] = None,
                 reverb: bool = False, snr_range=(0.0, 10.0),
                 rt60_range=(0.2, 0.6), rir_pool_size: int = 50,
                 p_clean: float = 0.34, seed: int = 0) -> None:
        self.base = base
        self.sr = base.sample_rate
        self.noise_files = (find_noise_files(Path(noise_dir))
                            if noise_dir is not None else [])
        self.reverb = reverb
        self.snr_range = snr_range
        self.p_clean = p_clean
        self.seed = seed
        pool_rng = np.random.default_rng((seed, 999))
        self.rir_pool = (build_rir_pool(rir_pool_size, self.sr, pool_rng,
                                        rt60_range) if reverb else [])

    def __len__(self) -> int:
        return len(self.base)

    def _add_noise(self, mixture: np.ndarray, rng) -> np.ndarray:
        if not self.noise_files:
            return mixture
        path = self.noise_files[int(rng.integers(0, len(self.noise_files)))]
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        noise = data.mean(axis=1)
        if len(noise) < len(mixture):
            reps = int(np.ceil(len(mixture) / max(len(noise), 1)))
            noise = np.tile(noise, reps)
        start = int(rng.integers(0, max(len(noise) - len(mixture), 0) + 1))
        noise = noise[start:start + len(mixture)]
        snr = float(rng.uniform(*self.snr_range))
        p_mix = float(np.mean(mixture ** 2))
        p_noise = float(np.mean(noise ** 2))
        if p_noise <= 0 or p_mix <= 0:
            return mixture
        g = np.sqrt(p_mix / (p_noise * (10.0 ** (snr / 10.0))))
        return (mixture + g * noise).astype(np.float32)

    def _apply_reverb(self, sources: np.ndarray, rng):
        length = sources.shape[1]
        wet = []
        for k in range(sources.shape[0]):
            rir = self.rir_pool[int(rng.integers(0, len(self.rir_pool)))]
            conv = fftconvolve(sources[k], rir)[:length]
            wet.append(conv.astype(np.float32))
        wet = np.stack(wet, axis=0)
        return wet, wet.sum(axis=0).astype(np.float32)

    def __getitem__(self, index: int):
        item = self.base[index]
        sources = item["sources"].numpy()          # [K, T]
        mixture = item["mixture"].numpy()           # [T]
        rng = np.random.default_rng((self.seed, 7, int(index)))

        roll = rng.random()
        do_reverb = self.reverb and roll > self.p_clean and rng.random() < 0.5
        do_noise = (len(self.noise_files) > 0 and roll > self.p_clean
                    and not do_reverb) or (do_reverb and rng.random() < 0.5)

        if do_reverb:
            sources, mixture = self._apply_reverb(sources, rng)
        if do_noise:
            mixture = self._add_noise(mixture, rng)

        return {
            "mixture": torch.from_numpy(mixture.astype(np.float32)),
            "sources": torch.from_numpy(sources.astype(np.float32)),
            "num_speakers": item["num_speakers"],
        }


def _self_test() -> int:
    """Wrap a tiny synthetic MixDataset and check shapes and degradation."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="voxsplit_aug_"))
    sr = 8000
    try:
        rng = np.random.default_rng(0)
        n = int(4.0 * sr)
        t = np.arange(n) / sr
        for s in range(3):
            d = tmp / f"{100 + s}"
            d.mkdir()
            for f in range(2):
                sig = 0.3 * np.sin(2 * np.pi * (150 + 40 * s) * t)
                sf.write(str(d / f"{100+s}-0-{f:04d}.flac"),
                         sig.astype(np.float32), sr, subtype="PCM_16")
        # A short noise file.
        noise_dir = tmp / "noise"
        noise_dir.mkdir()
        sf.write(str(noise_dir / "n.wav"),
                 (0.1 * rng.standard_normal(n)).astype(np.float32), sr)

        base = MixDataset(tmp, sample_rate=sr, segment_seconds=2.0,
                          speaker_counts=(2,), seed=0, length=20)
        aug = AugMixDataset(base, noise_dir=noise_dir, reverb=True,
                            rir_pool_size=3, seed=0)
        seg = int(2.0 * sr)
        changed = 0
        for i in range(12):
            it = aug[i]
            assert it["mixture"].shape == (seg,), it["mixture"].shape
            k = it["num_speakers"]
            assert it["sources"].shape == (k, seg), it["sources"].shape
            base_mix = base[i]["mixture"].numpy()
            if not np.allclose(base_mix, it["mixture"].numpy(), atol=1e-4):
                changed += 1
        print(f"{changed}/12 items were degraded (noise/reverb applied)")
        assert changed >= 4, changed
        print(f"RIR pool size: {len(aug.rir_pool)}")
        print("All augment self-tests passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(_self_test())
