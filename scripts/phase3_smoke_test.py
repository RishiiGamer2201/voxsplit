"""Fast smoke test for VoxSplit Phase 3 OR-PIT fine-tuning (no LibriSpeech).

Verifies three things without any real data or long training:
  1. The cached sepformer-wsj02mix forward produces the expected shapes and
     exactly two speaker heads.
  2. The orpit_loss path runs through a real batch.
  3. A handful of optimizer steps on a tiny RANDOM synthetic batch, run
     through train_orpit's forward and loss path, produce a finite loss that
     DECREASES across steps.

Run:
  python scripts/phase3_smoke_test.py
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "train"))

from orpit_loss import batch_orpit_loss  # noqa: E402
from train_orpit import (  # noqa: E402
    load_warm_start,
    separate_forward,
)


def make_random_batch(batch_size: int, seg: int, device: str):
    """Build a tiny random batch: mixture = sum of random per-item sources.

    Ks alternate 2 and 3 so both the K=2 (rest is one speaker) and K=3 (rest
    is a sum) paths are exercised.
    """
    gen = torch.Generator().manual_seed(0)
    mixtures = []
    sources_list = []
    for b in range(batch_size):
        k = 2 if b % 2 == 0 else 3
        src = 0.3 * torch.randn(k, seg, generator=gen)
        sources_list.append(src.to(device))
        mixtures.append(src.sum(dim=0))
    mixture = torch.stack(mixtures, dim=0).to(device)
    return mixture, sources_list


def main() -> int:
    device = "cpu"
    print(f"Device: {device}")

    print("")
    print("== Step 1: load cached wsj02mix and check forward shapes ==")
    encoder, masknet, decoder = load_warm_start(
        "speechbrain/sepformer-wsj02mix", device)

    mix = torch.randn(2, 16000)
    with torch.no_grad():
        mix_w = encoder(mix)
        est_mask = masknet(mix_w)
        est = separate_forward(encoder, masknet, decoder, mix)
    print(f"mix        shape: {tuple(mix.shape)}")
    print(f"mix_w      shape: {tuple(mix_w.shape)}  (expect [B, F, L])")
    print(f"est_mask   shape: {tuple(est_mask.shape)}  "
          f"(expect [num_spk=2, B, F, L])")
    print(f"est        shape: {tuple(est.shape)}  (expect [2, ~16000, 2])")
    assert est_mask.shape[0] == 2, "masknet must output 2 speakers"
    assert est.shape[0] == 2 and est.shape[2] == 2, est.shape
    print("Shape checks passed: masknet gives 2 speakers.")

    print("")
    print("== Step 2: three optimizer steps on a tiny random batch ==")
    encoder.train()
    masknet.train()
    decoder.train()
    params = (list(encoder.parameters())
              + list(masknet.parameters())
              + list(decoder.parameters()))
    # A small lr keeps the tiny random-data run stable; the point is only to
    # confirm the loss path is differentiable and the loss goes down.
    optimizer = torch.optim.Adam(params, lr=5e-5)

    seg = 8000  # 1 second at 8 kHz, small for a fast CPU smoke run.
    mixture, sources = make_random_batch(batch_size=2, seg=seg, device=device)

    losses = []
    num_steps = 3
    for step in range(1, num_steps + 1):
        est = separate_forward(encoder, masknet, decoder, mixture)
        outputs = est.permute(0, 2, 1).contiguous()  # [B, 2, T2]
        loss, best_js = batch_orpit_loss(outputs, sources)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        val = float(loss)
        losses.append(val)
        print(f"step {step}: loss={val:.4f}  best_js={best_js}")
        assert torch.isfinite(loss), "loss is not finite"

    print("")
    print(f"loss values: {[round(v, 4) for v in losses]}")
    assert losses[-1] < losses[0], (
        f"loss did not decrease: first={losses[0]:.4f} "
        f"last={losses[-1]:.4f}")
    print(f"Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("")
    print("PHASE 3 SMOKE TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
