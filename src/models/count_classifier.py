"""Speaker-count / stop classifier for VoxSplit Phase 4 (unknown count).

A small CNN over a log-mel spectrogram that predicts how many speakers are
active in a waveform, as logits over K in {1,2,3,4,5}. It serves two Phase 4
jobs at once:
  - Stop criterion for recursive OR-PIT separation: P(>=2 speakers) on the
    residual "rest" head decides whether another speaker remains.
  - Head selection: of the two OR-PIT heads, recurse on the one with the
    higher P(>=2 speakers) (the mixture of the rest) and keep the other (the
    extracted single speaker).
The predicted count also comes for free from the recursion depth, which
Takahashi et al. (2019) report is more accurate than reading it off this
classifier directly; this net is the driver plus a standalone cross-check.

Run this file directly for a self-test (shape + a toy 1-vs-3 learnability
check, no real data):
  python src/models/count_classifier.py
"""
import torch
import torch.nn as nn
import torchaudio.transforms as T

MAX_SPEAKERS = 5


class SpeakerCountCNN(nn.Module):
    """Log-mel CNN emitting count logits over K in {1..num_classes}."""

    def __init__(self, sample_rate: int = 8000, n_mels: int = 64,
                 num_classes: int = MAX_SPEAKERS) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels)

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T] -> logits [B, num_classes]
        mel = self.melspec(wav)                      # [B, n_mels, frames]
        x = torch.log(mel + 1e-6).unsqueeze(1)       # [B, 1, n_mels, frames]
        x = self.features(x).flatten(1)              # [B, 64]
        return self.head(x)

    @torch.no_grad()
    def prob_multi(self, wav: torch.Tensor) -> torch.Tensor:
        """P(>=2 speakers) = 1 - P(K==1) for each item in the batch."""
        probs = torch.softmax(self.forward(wav), dim=-1)
        return 1.0 - probs[:, 0]


def _self_test() -> int:
    """Forward shape plus a toy check that it learns 1 vs 3 overlapped tones."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeakerCountCNN(num_classes=MAX_SPEAKERS).to(device)

    b, t = 4, 8000
    wav = torch.randn(b, t, device=device)
    logits = model(wav)
    assert logits.shape == (b, MAX_SPEAKERS), logits.shape
    print(f"forward ok: {tuple(wav.shape)} -> {tuple(logits.shape)}")

    # Toy task: single tone (K=1, label 0) vs sum of 3 tones (K=3, label 2).
    tt = torch.arange(t, device=device) / 8000.0
    freqs = [180.0, 330.0, 520.0]

    def sample(k: int) -> torch.Tensor:
        sig = sum(torch.sin(2 * torch.pi * freqs[i] * tt) for i in range(k))
        return (sig / k + 0.01 * torch.randn(t, device=device))

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    losses = []
    for _ in range(40):
        xs = torch.stack([sample(1), sample(3), sample(1), sample(3)])
        ys = torch.tensor([0, 2, 0, 2], device=device)
        loss = ce(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print(f"toy CE loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0] - 0.3, (losses[0], losses[-1])

    model.eval()
    p1 = float(model.prob_multi(sample(1).unsqueeze(0)))
    p3 = float(model.prob_multi(sample(3).unsqueeze(0)))
    print(f"P(>=2) single={p1:.2f} triple={p3:.2f}")
    assert p3 > p1, (p1, p3)
    print("All SpeakerCountCNN self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
