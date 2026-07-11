"""Fast smoke test for the Phase 3 baseline tracks (no LibriSpeech, no long run).

Checks, on RANDOM data only:
  1. pit_loss.py self-test passes (imported and run in-process).
  2. The cached sepformer-libri3mix loads; sepformer_forward on a random
     [1, 16000] mixture with num_spks=3 gives est shaped [1, ~16000, 3].
  3. Three optimizer steps of train_pit's loss path on a tiny random 3-source
     batch keep the loss finite and decreasing.
  4. ConvTasNet(num_sources=2) runs three optimizer steps on a tiny random
     2-source batch with the loss finite and decreasing.

Run:
  python scripts/phase3_baseline_smoke_test.py
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "train"))

from pit_loss import _self_test as pit_self_test  # noqa: E402
from pit_loss import batch_pit_loss  # noqa: E402
from train_pit import load_warm_start, sepformer_forward  # noqa: E402


def section(title: str) -> None:
    print("")
    print("=" * 60)
    print(title)
    print("=" * 60)


def make_random_batch(batch: int, num_speakers: int, samples: int):
    """Random [B, num_speakers, T] sources summed to a [B, T] mixture."""
    sources = torch.randn(batch, num_speakers, samples)
    mixture = sources.sum(dim=1)
    return mixture, sources


def check_decreasing(losses, label: str) -> None:
    finite = all(torch.isfinite(torch.tensor(x)) for x in losses)
    print(f"{label} losses: " + ", ".join(f"{x:.4f}" for x in losses))
    assert finite, f"{label}: non-finite loss encountered."
    assert losses[-1] < losses[0], (
        f"{label}: loss did not decrease ({losses[0]:.4f} -> "
        f"{losses[-1]:.4f}).")
    print(f"{label}: loss is finite and decreased "
          f"({losses[0]:.4f} -> {losses[-1]:.4f}).")


def main() -> int:
    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 1. pit_loss self-test ----
    section("1. pit_loss.py self-test")
    rc = pit_self_test()
    assert rc == 0, "pit_loss self-test returned nonzero."

    # ---- 2. sepformer_forward shape on random input ----
    section("2. sepformer-libri3mix forward shape (num_spks=3)")
    encoder, masknet, decoder = load_warm_start(
        "speechbrain/sepformer-libri3mix", 3, device)
    mix = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        est = sepformer_forward(encoder, masknet, decoder, mix, 3)
    fwd_shape = tuple(est.shape)
    print(f"sepformer_forward est shape: {fwd_shape} "
          f"(expected [1, ~16000, 3])")
    assert est.shape[0] == 1 and est.shape[2] == 3, fwd_shape
    assert abs(est.shape[1] - 16000) <= 64, est.shape[1]

    # ---- 3. three optimizer steps of train_pit's loss path (3 sources) ----
    section("3. train_pit loss path, 3 optimizer steps (3 sources)")
    params = (list(encoder.parameters())
              + list(masknet.parameters())
              + list(decoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-4)
    # Small T to keep this fast; batch of 1 (3-output SepFormer is heavy).
    samples = 8000
    mixture, sources = make_random_batch(1, 3, samples)
    mixture = mixture.to(device)
    targets = sources.to(device)
    sep_losses = []
    for _ in range(3):
        est = sepformer_forward(encoder, masknet, decoder, mixture, 3)
        outputs = est.permute(0, 2, 1).contiguous()
        length = min(outputs.shape[-1], targets.shape[-1])
        loss, _ = batch_pit_loss(outputs[..., :length], targets[..., :length])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        sep_losses.append(float(loss))
    check_decreasing(sep_losses, "sepformer uPIT")

    # ---- 4. ConvTasNet three optimizer steps (2 sources) ----
    section("4. ConvTasNet from scratch, 3 optimizer steps (2 sources)")
    from torchaudio.models import ConvTasNet
    model = ConvTasNet(num_sources=2).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mixture, sources = make_random_batch(4, 2, samples)
    mixture = mixture.to(device)
    targets = sources.to(device)
    conv_losses = []
    for _ in range(3):
        est = model(mixture.unsqueeze(1))     # [B, 2, T]
        length = min(est.shape[-1], targets.shape[-1])
        loss, _ = batch_pit_loss(est[..., :length], targets[..., :length])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        conv_losses.append(float(loss))
    check_decreasing(conv_losses, "convtasnet PIT")

    section("SUMMARY")
    print(f"sepformer_forward shape: {fwd_shape} confirmed on a random "
          f"16000-sample mixture.")
    print(f"sepformer uPIT losses (3 steps): "
          + ", ".join(f"{x:.4f}" for x in sep_losses))
    print(f"convtasnet PIT losses  (3 steps): "
          + ", ".join(f"{x:.4f}" for x in conv_losses))
    print("All Phase 3 baseline smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
