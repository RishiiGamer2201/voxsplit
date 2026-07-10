"""Separation metrics for VoxSplit: SI-SDR, PESQ, STOI, and PIT matching.

All functions operate on 1-D numpy arrays of equal length at a known
sample rate. SI-SDR is scale invariant, so estimate and reference do not
need matched gains. Permutation matching handles the fact that a separator
outputs sources in an arbitrary order.

Run this file directly for a self-test that needs no real data or model:
  python src/eval/metrics.py
"""
import itertools
from typing import List, Tuple

import numpy as np

# A large but finite dB value returned when signals are effectively
# identical (zero error), to avoid returning +inf.
SI_SDR_MAX_DB = 200.0


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal to distortion ratio in dB.

    Projects the estimate onto the reference to remove a global scale,
    then compares signal energy to residual energy. Returns a large
    positive number (SI_SDR_MAX_DB) when the estimate matches the
    reference up to scale (zero residual energy).
    """
    estimate = np.asarray(estimate, dtype=np.float64).flatten()
    reference = np.asarray(reference, dtype=np.float64).flatten()
    n = min(len(estimate), len(reference))
    estimate = estimate[:n]
    reference = reference[:n]

    ref_energy = float(np.dot(reference, reference)) + 1e-12
    scale = float(np.dot(estimate, reference)) / ref_energy
    projection = scale * reference
    noise = estimate - projection

    proj_energy = float(np.dot(projection, projection))
    noise_energy = float(np.dot(noise, noise))
    if noise_energy <= 0.0:
        return SI_SDR_MAX_DB
    value = 10.0 * np.log10((proj_energy + 1e-12) / (noise_energy + 1e-12))
    return float(value)


def best_permutation(estimates: List[np.ndarray],
                     references: List[np.ndarray],
                     ) -> Tuple[List[int], List[Tuple[int, int]], float]:
    """Find the estimate ordering that maximizes mean SI-SDR vs references.

    K is small (up to about 6), so an exhaustive search over
    itertools.permutations is used. Returns:
      perm:  perm[j] is the estimate index matched to reference j.
      pairs: list of (estimate_index, reference_index) matched tuples.
      mean_si_sdr: mean SI-SDR over the matched pairs, in dB.
    """
    k = len(references)
    if len(estimates) != k:
        raise ValueError(
            f"Number of estimates ({len(estimates)}) must equal number of "
            f"references ({k}).")

    best_perm: Tuple[int, ...] = tuple(range(k))
    best_score = -np.inf
    for perm in itertools.permutations(range(k)):
        scores = [si_sdr(estimates[perm[j]], references[j])
                  for j in range(k)]
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_perm = perm

    perm_list = list(best_perm)
    pairs = [(best_perm[j], j) for j in range(k)]
    return perm_list, pairs, float(best_score)


def compute_pesq(reference: np.ndarray, estimate: np.ndarray,
                 sample_rate: int = 8000) -> float:
    """Narrowband PESQ at 8 kHz. Returns NaN on failure.

    PESQ can raise on degenerate or silent signals, so it is wrapped in a
    try/except and NaN is recorded on any failure.
    """
    try:
        from pesq import pesq as pesq_fn
        n = min(len(reference), len(estimate))
        ref = np.asarray(reference, dtype=np.float64).flatten()[:n]
        est = np.asarray(estimate, dtype=np.float64).flatten()[:n]
        return float(pesq_fn(sample_rate, ref, est, "nb"))
    except Exception:
        return float("nan")


def compute_stoi(reference: np.ndarray, estimate: np.ndarray,
                 sample_rate: int = 8000) -> float:
    """Short-Time Objective Intelligibility. Returns NaN on failure."""
    try:
        from pystoi import stoi as stoi_fn
        n = min(len(reference), len(estimate))
        ref = np.asarray(reference, dtype=np.float64).flatten()[:n]
        est = np.asarray(estimate, dtype=np.float64).flatten()[:n]
        return float(stoi_fn(ref, est, sample_rate))
    except Exception:
        return float("nan")


def _self_test() -> int:
    """Sanity checks that need no real data or model."""
    rng = np.random.default_rng(0)
    n = 8000

    # A sine and a filtered-noise signal as two distinct sources.
    t = np.arange(n) / 8000.0
    sine = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float64)
    noise = rng.standard_normal(n)
    # Simple low-pass by cumulative smoothing so it differs from the sine.
    filtered = np.convolve(noise, np.ones(16) / 16.0, mode="same")

    # 1) SI-SDR of a signal against itself is very high.
    self_score = si_sdr(sine, sine)
    print(f"si_sdr(sine, sine)         = {self_score:.2f} dB")
    assert self_score >= 100.0, "self SI-SDR should be very large"

    # Scaled copy should also score very high (scale invariance).
    scaled_score = si_sdr(3.7 * sine, sine)
    print(f"si_sdr(3.7*sine, sine)     = {scaled_score:.2f} dB")
    assert scaled_score >= 100.0, "SI-SDR must be scale invariant"

    # A different signal should score much lower.
    cross_score = si_sdr(filtered, sine)
    print(f"si_sdr(filtered, sine)     = {cross_score:.2f} dB")
    assert cross_score < self_score, "different signal should score lower"

    # 2) best_permutation recovers a known shuffle.
    references = [sine, filtered]
    # Present estimates in swapped order; matcher should undo the swap.
    estimates = [filtered * 0.5, sine * 2.0]
    perm, pairs, mean_score = best_permutation(estimates, references)
    print(f"best_permutation perm      = {perm}")
    print(f"best_permutation pairs     = {pairs}")
    print(f"best_permutation mean SISDR= {mean_score:.2f} dB")
    # reference 0 (sine) should match estimate index 1, reference 1 index 0.
    assert perm == [1, 0], f"expected [1, 0], got {perm}"
    assert mean_score >= 100.0, "matched pairs should score very high"

    print("All metrics self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
