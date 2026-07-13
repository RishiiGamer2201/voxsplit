"""Compact TF-GridNet for VoxSplit Phase 3 (time-frequency separation baseline).

A faithful-but-small implementation of TF-GridNet (Wang et al. 2023,
arXiv:2209.03952): operate on the complex STFT, alternate an intra-frame
(across frequency) BLSTM with an inter-frame (across time) BLSTM inside each
grid block, then predict a complex ratio mask per speaker and inverse-STFT
back to waveforms. Plugs straight into the existing PIT pipeline (the forward
returns [B, num_spks, T] time-domain, like Conv-TasNet).

ponytail: this drops the paper's full-band self-attention module inside each
block (intra/inter BLSTMs only) and uses small channel/hidden dims, so it is a
learning-scale baseline, not the 23 dB SOTA config. Upgrade path if a real
TF-domain contender is wanted: add the MHSA sub-module and widen D/H.

Run this file directly for a self-test (shape + one loss-reducing step, no
real data):
  python src/models/tfgridnet.py
"""
from typing import Tuple

import torch
import torch.nn as nn


class GridNetBlock(nn.Module):
    """One TF-GridNet block: intra-frequency then inter-time BLSTM, residual."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.intra_norm = nn.GroupNorm(1, dim)
        self.intra_rnn = nn.LSTM(dim, hidden, batch_first=True,
                                 bidirectional=True)
        self.intra_fc = nn.Linear(2 * hidden, dim)
        self.inter_norm = nn.GroupNorm(1, dim)
        self.inter_rnn = nn.LSTM(dim, hidden, batch_first=True,
                                 bidirectional=True)
        self.inter_fc = nn.Linear(2 * hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, F]
        b, d, t, f = x.shape

        # Intra-frame: sequence over frequency, batch over (B, T).
        y = self.intra_norm(x)
        y = y.permute(0, 2, 3, 1).reshape(b * t, f, d)
        y, _ = self.intra_rnn(y)
        y = self.intra_fc(y).reshape(b, t, f, d).permute(0, 3, 1, 2)
        x = x + y

        # Inter-frame: sequence over time, batch over (B, F).
        z = self.inter_norm(x)
        z = z.permute(0, 3, 2, 1).reshape(b * f, t, d)
        z, _ = self.inter_rnn(z)
        z = self.inter_fc(z).reshape(b, f, t, d).permute(0, 3, 2, 1)
        x = x + z
        return x


class TFGridNet(nn.Module):
    """Compact TF-GridNet monaural separator returning time-domain sources."""

    def __init__(self, num_spks: int = 3, n_fft: int = 256, hop: int = 128,
                 dim: int = 48, hidden: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.num_spks = num_spks
        self.n_fft = n_fft
        self.hop = hop
        self.dim = dim
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.register_buffer("window", torch.hann_window(n_fft))

        self.input = nn.Conv2d(2, dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [GridNetBlock(dim, hidden) for _ in range(num_blocks)])
        # Two channels (real, imag) per speaker.
        self.output = nn.Conv2d(dim, 2 * num_spks, kernel_size=3, padding=1)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.stft(wav, self.n_fft, self.hop, window=self.window,
                          return_complex=True)  # [B, F, Tf]

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(spec, self.n_fft, self.hop, window=self.window,
                           length=length)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T] -> [B, num_spks, T]
        length = wav.shape[-1]
        spec = self.stft(wav)                       # [B, F, Tf] complex
        re, im = spec.real, spec.imag               # [B, F, Tf]

        x = torch.stack([re, im], dim=1)            # [B, 2, F, Tf]
        x = x.permute(0, 1, 3, 2)                   # [B, 2, Tf, F]
        x = self.input(x)                           # [B, D, Tf, F]
        for block in self.blocks:
            x = block(x)
        m = self.output(x)                          # [B, 2*S, Tf, F]
        m = m.permute(0, 1, 3, 2)                   # [B, 2*S, F, Tf]

        outs = []
        for s in range(self.num_spks):
            mask = torch.complex(m[:, 2 * s], m[:, 2 * s + 1])  # [B, F, Tf]
            est_spec = mask * spec
            outs.append(self.istft(est_spec, length))
        return torch.stack(outs, dim=1)             # [B, num_spks, T]


def _self_test() -> int:
    """Forward shape check plus one PIT step that reduces the loss on a toy."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from pit_loss import batch_pit_loss  # noqa: E402

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFGridNet(num_spks=2, num_blocks=2, dim=32, hidden=32).to(device)

    b, t = 2, 8000
    wav = torch.randn(b, t, device=device)
    out = model(wav)
    assert out.shape == (b, 2, t), out.shape
    print(f"forward ok: in {tuple(wav.shape)} -> out {tuple(out.shape)}")

    # One overfitting step on a fixed toy 2-source mixture must reduce loss.
    tone = torch.sin(2 * torch.pi * 220 * torch.arange(t, device=device) / 8000)
    noise = torch.randn(t, device=device) * 0.3
    targets = torch.stack([tone, noise], dim=0).unsqueeze(0)  # [1, 2, T]
    mix = targets.sum(dim=1)                                  # [1, T]

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        est = model(mix)
        n = min(est.shape[-1], targets.shape[-1])
        loss, _ = batch_pit_loss(est[..., :n], targets[..., :n])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print(f"toy PIT loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0] - 1.0, (losses[0], losses[-1])
    print("All TFGridNet self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
