"""OR-PIT loss for VoxSplit Phase 3 (Takahashi et al. 2019, arXiv:1904.03065).

OR-PIT (one-and-rest permutation invariant training) drives a 2-output
separator to place ONE speaker on one head and the SUM OF ALL REMAINING
speakers ("the rest") on the other head. For each item we try every choice of
which source is the "one", pair its "rest" against the two heads in both
assignments, and keep the best. The loss is the negative mean SI-SNR of that
best match, so minimizing it improves separation.

Run this file directly for a self-test that needs no real data or model:
  python src/train/orpit_loss.py
"""
import argparse
from typing import List, Tuple

import torch


def si_snr(est: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SNR in dB for a single [T] pair (higher is better).

    The target is zero-mean centered, the estimate is projected onto it to
    remove a global scale, then signal energy is compared to residual energy.
    """
    est = est.flatten()
    target = target.flatten()
    n = min(est.shape[0], target.shape[0])
    est = est[:n]
    target = target[:n]

    est = est - est.mean()
    target = target - target.mean()

    ref_energy = torch.sum(target * target) + eps
    scale = torch.sum(est * target) / ref_energy
    proj = scale * target
    noise = est - proj

    ratio = (torch.sum(proj * proj) + eps) / (torch.sum(noise * noise) + eps)
    return 10.0 * torch.log10(ratio)


def orpit_loss(outputs: torch.Tensor, sources: torch.Tensor
               ) -> Tuple[torch.Tensor, int]:
    """OR-PIT loss for one item.

    outputs: [2, T] the two head estimates.
    sources: [K, T] the K clean references (K >= 2).

    For each candidate "one" speaker j: one = sources[j], rest = sum of the
    others. Score both head assignments and take the better. Choose the j with
    the highest score. Returns (loss, best_j) where loss = -0.5 * best_value
    (negative mean SI-SNR over the two heads).
    """
    k = sources.shape[0]
    n = min(outputs.shape[-1], sources.shape[-1])
    outputs = outputs[:, :n]
    sources = sources[:, :n]

    total = sources.sum(dim=0)  # [T]

    best_value = None       # tensor carrying the grad for the winning j
    best_scalar = None      # detached float, only for the argmax comparison
    best_j = 0
    for j in range(k):
        one = sources[j]
        rest = total - one  # sum of all remaining sources
        # Assignment A: head0 -> one, head1 -> rest.
        a = si_snr(outputs[0], one) + si_snr(outputs[1], rest)
        # Assignment B: head1 -> one, head0 -> rest.
        b = si_snr(outputs[1], one) + si_snr(outputs[0], rest)
        value = torch.maximum(a, b)
        scalar = float(value.detach())
        if best_scalar is None or scalar > best_scalar:
            best_value = value
            best_scalar = scalar
            best_j = j

    loss = -0.5 * best_value
    return loss, best_j


def batch_orpit_loss(batch_outputs: torch.Tensor,
                     batch_sources: List[torch.Tensor]
                     ) -> Tuple[torch.Tensor, List[int]]:
    """Average orpit_loss over a batch.

    batch_outputs: [B, 2, T] the two heads per item.
    batch_sources: list of B tensors, each [K_b, T].
    Returns (mean_loss, best_js) with one best_j per item.
    """
    losses: List[torch.Tensor] = []
    best_js: List[int] = []
    for i in range(len(batch_sources)):
        loss, best_j = orpit_loss(batch_outputs[i], batch_sources[i])
        losses.append(loss)
        best_js.append(best_j)
    mean_loss = torch.stack(losses).mean()
    return mean_loss, best_js


def _self_test() -> int:
    """Self-test on synthetic sources (no real data or model needed)."""
    torch.manual_seed(0)
    t = 8000

    # ---- K = 3, plant a known "one" ----
    sources = torch.randn(3, t)
    planted_j = 1
    one = sources[planted_j]
    rest = sources.sum(dim=0) - one
    # Outputs set exactly to (one, rest) for the planted j.
    outputs = torch.stack([one, rest], dim=0)

    loss, best_j = orpit_loss(outputs, sources)
    print(f"K=3 planted j={planted_j}: loss={float(loss):.2f} "
          f"best_j={best_j}")
    assert best_j == planted_j, (best_j, planted_j)
    # Exact match: SI-SNR is huge, so loss is very negative.
    assert float(loss) < -50.0, float(loss)

    # A wrong assignment (heads swapped away from any good match) scores worse.
    bad_outputs = torch.stack([rest, torch.randn(t)], dim=0)
    bad_loss, _ = orpit_loss(bad_outputs, sources)
    print(f"K=3 wrong assignment:   loss={float(bad_loss):.2f}")
    assert float(bad_loss) > float(loss), (float(bad_loss), float(loss))

    # ---- K = 2 special case: rest is just the other speaker ----
    src2 = torch.randn(2, t)
    out2 = torch.stack([src2[0], src2[1]], dim=0)
    loss2, best_j2 = orpit_loss(out2, src2)
    print(f"K=2 exact match:        loss={float(loss2):.2f} "
          f"best_j={best_j2}")
    assert float(loss2) < -50.0, float(loss2)

    # ---- batch helper ----
    batch_outputs = torch.stack([outputs, out2[:, :t]], dim=0)
    # batch_sources holds mixed-K items.
    batch_sources = [sources, src2]
    mean_loss, best_js = batch_orpit_loss(batch_outputs, batch_sources)
    print(f"batch mean loss={float(mean_loss):.2f} best_js={best_js}")
    assert best_js[0] == planted_j, best_js
    assert float(mean_loss) < -50.0, float(mean_loss)

    print("All orpit_loss self-tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-test the OR-PIT one-and-rest loss.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic self-test (default action).")
    parser.parse_args()
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
