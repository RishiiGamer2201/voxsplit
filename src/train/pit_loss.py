"""Standard utterance-level PIT loss for a FIXED number of outputs N.

Classic permutation invariant training (Yu et al. 2017, arXiv:1607.00325):
a separator emits exactly N heads, and each head must match one distinct
clean reference. We try every permutation of references onto heads, sum the
per-pair SI-SNR, keep the best permutation, and return the negative mean
SI-SNR of that permutation as the loss (so minimizing it improves separation).

This differs from orpit_loss.py: OR-PIT places one speaker on one head and
the sum of the rest on the other head (2 heads, variable speaker count). Here
we have exactly N heads and N references and match them one to one.

Reuses si_snr from orpit_loss.py for the per-pair score.

Run this file directly for a self-test that needs no real data or model:
  python src/train/pit_loss.py
"""
import argparse
import itertools
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# Cross-folder import of si_snr, matching the sys.path pattern in
# train_orpit.py and mix_dataset.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from orpit_loss import si_snr  # noqa: E402


def pit_loss(outputs: torch.Tensor, targets: torch.Tensor
             ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Utterance-level PIT loss for one item with a fixed N.

    outputs: [N, T] the N head estimates.
    targets: [N, T] the N clean references.

    Over every permutation p of range(N), compute
    sum_i si_snr(outputs[i], targets[p[i]]); pick the permutation with the
    highest sum. Returns (loss, best_perm) where loss is the negative mean
    SI-SNR of the winning permutation, so loss = -(max_sum) / N.

    The grad-carrying tensor for the winning permutation is kept; a detached
    float is used only for the argmax comparison (same trick as orpit_loss).
    """
    n = outputs.shape[0]
    if targets.shape[0] != n:
        raise ValueError(
            f"outputs has {n} heads but targets has {targets.shape[0]}.")
    length = min(outputs.shape[-1], targets.shape[-1])
    outputs = outputs[:, :length]
    targets = targets[:, :length]

    best_value = None       # tensor carrying the grad for the winning perm
    best_scalar = None      # detached float, only for the argmax comparison
    best_perm: Tuple[int, ...] = tuple(range(n))
    for perm in itertools.permutations(range(n)):
        value = None
        for i in range(n):
            pair = si_snr(outputs[i], targets[perm[i]])
            value = pair if value is None else value + pair
        scalar = float(value.detach())
        if best_scalar is None or scalar > best_scalar:
            best_value = value
            best_scalar = scalar
            best_perm = perm

    loss = -(best_value) / n
    return loss, best_perm


def batch_pit_loss(batch_outputs: torch.Tensor,
                   batch_targets: torch.Tensor
                   ) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
    """Average pit_loss over a batch.

    batch_outputs: [B, N, T] the N heads per item.
    batch_targets: [B, N, T] the N references per item.
    Returns (mean_loss, best_perms) with one best_perm per item.
    """
    losses: List[torch.Tensor] = []
    best_perms: List[Tuple[int, ...]] = []
    for i in range(batch_outputs.shape[0]):
        loss, best_perm = pit_loss(batch_outputs[i], batch_targets[i])
        losses.append(loss)
        best_perms.append(best_perm)
    mean_loss = torch.stack(losses).mean()
    return mean_loss, best_perms


def _self_test() -> int:
    """Self-test on synthetic tensors (no real data or model needed)."""
    torch.manual_seed(0)
    t = 8000

    # ---- N = 3, plant a known shuffle ----
    targets = torch.randn(3, t)
    planted = (2, 0, 1)  # outputs[i] == targets[planted[i]]
    outputs = torch.stack([targets[planted[i]] for i in range(3)], dim=0)

    loss, best_perm = pit_loss(outputs, targets)
    print(f"N=3 planted perm={planted}: loss={float(loss):.2f} "
          f"best_perm={best_perm}")
    # best_perm[i] must recover which target each output copied.
    assert best_perm == planted, (best_perm, planted)
    # Exact match: SI-SNR is huge, so loss is very negative.
    assert float(loss) < -50.0, float(loss)

    # A wrong permutation scores worse (higher loss) than the best.
    wrong = (0, 1, 2)  # not the planted shuffle
    wrong_value = None
    for i in range(3):
        pair = si_snr(outputs[i], targets[wrong[i]])
        wrong_value = pair if wrong_value is None else wrong_value + pair
    wrong_loss = float(-(wrong_value) / 3.0)
    print(f"N=3 wrong perm={wrong}:   loss={wrong_loss:.2f}")
    assert wrong_loss > float(loss), (wrong_loss, float(loss))

    # ---- N = 2, plant a swap ----
    targets2 = torch.randn(2, t)
    planted2 = (1, 0)
    outputs2 = torch.stack([targets2[planted2[i]] for i in range(2)], dim=0)
    loss2, best_perm2 = pit_loss(outputs2, targets2)
    print(f"N=2 planted perm={planted2}: loss={float(loss2):.2f} "
          f"best_perm={best_perm2}")
    assert best_perm2 == planted2, (best_perm2, planted2)
    assert float(loss2) < -50.0, float(loss2)

    # ---- batch helper ----
    batch_outputs = torch.stack([outputs, outputs[[1, 2, 0]]], dim=0)
    batch_targets = torch.stack([targets, targets], dim=0)
    mean_loss, best_perms = batch_pit_loss(batch_outputs, batch_targets)
    print(f"batch mean loss={float(mean_loss):.2f} best_perms={best_perms}")
    assert best_perms[0] == planted, best_perms
    assert float(mean_loss) < -50.0, float(mean_loss)

    print("All pit_loss self-tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Self-test the fixed-N utterance-level PIT loss.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic self-test (default action).")
    parser.parse_args()
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
