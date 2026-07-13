"""Train the Phase 4 speaker-count / stop classifier.

Trains SpeakerCountCNN (src/models/count_classifier.py) to predict the number
of active speakers (K in 1..5) from a waveform, using the same on-the-fly
mixtures as the separators (mix_dataset.py) but including K=1 (a single clean
utterance). The label is num_speakers-1; cross-entropy loss. The default
speaker-count distribution over-samples low counts because the decision that
matters most for recursive stopping is 1 vs 2 speakers.

Note: this trains on clean mixtures. At inference the classifier is applied to
OR-PIT residual "rest" heads, which carry separation artifacts, so there is a
domain gap; blind-recursion results quantify how much it costs. Refining on
real residuals is a follow-up if the gap is large.

Example:
  python src/train/train_count_classifier.py \
      --source-dir data/LibriSpeech/train-clean-100 \
      --out-dir checkpoints/count_clf --max-steps 6000 \
      --batch-size 8 --lr 1e-3 --device auto --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from mix_dataset import MixDataset, collate  # noqa: E402
from orpit_loss import si_snr  # noqa: E402
from train_orpit import (resolve_device, separate_forward,  # noqa: E402
                         neutralize_lazy_modules)
from wandb_logger import WandbLogger  # noqa: E402
from count_classifier import SpeakerCountCNN, MAX_SPEAKERS  # noqa: E402


def parse_counts(text: str):
    counts = tuple(int(x) for x in text.split(",") if x.strip())
    if not counts:
        raise ValueError("speaker-counts must list at least one integer.")
    return counts


def load_orpit(init_model: str, ckpt_path: Path, device: str):
    """Load a converged OR-PIT model in eval mode for residual generation."""
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy
    savedir = Path("pretrained_models") / init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()
    ck = torch.load(str(ckpt_path), map_location=device)
    enc, mn, dec = sep.mods.encoder, sep.mods.masknet, sep.mods.decoder
    enc.load_state_dict(ck["encoder"])
    mn.load_state_dict(ck["masknet"])
    dec.load_state_dict(ck["decoder"])
    for m in (enc, mn, dec):
        m.eval()
    return enc, mn, dec


def _forward_one(signal, enc, mn, dec):
    """OR-PIT forward on a single [T] waveform -> heads [2, T2]."""
    with torch.no_grad():
        est = separate_forward(enc, mn, dec, signal.unsqueeze(0))  # [1,T2,2]
    return est[0].permute(1, 0).contiguous()                        # [2, T2]


def _pick_single_head(heads, sources, remaining):
    """Of the 2 heads pick the single extracted speaker among remaining refs.

    Returns (single_head_idx, matched_ref_idx). The "one" is chosen over the
    still-present references only, and the head assignment that best fits it.
    """
    n = min(heads.shape[-1], sources.shape[-1])
    heads = heads[:, :n]
    src = sources[:, :n]
    total = sum(src[r] for r in remaining)
    best = None
    single_idx, matched = 0, remaining[0]
    for r in remaining:
        one = src[r]
        rest = total - one
        a = si_snr(heads[0], one) + si_snr(heads[1], rest)  # head0 = one
        b = si_snr(heads[1], one) + si_snr(heads[0], rest)  # head1 = one
        score, assign = (float(a), 0) if a >= b else (float(b), 1)
        if best is None or score > best:
            best, single_idx, matched = score, assign, r
    return single_idx, matched


def residualize(mixture, sources_list, nums, enc, mn, dec, rng):
    """Turn a clean-mixture batch into OR-PIT-domain (waveform, count) pairs.

    For each item, emit one of: the raw mixture (count = M); a single extracted
    head at a random recursion depth (count = 1); or the residual "rest" head
    after a random number of oracle extractions (count = remaining speakers).
    Recursing to a random depth exposes the classifier to the artifact-heavy
    DEEP residuals that a single forward pass never produces, which is what
    made shallow-trained classifiers over-count 3- and 4-speaker mixtures.
    Head roles are identified with the references (oracle), fine at train time.
    """
    xs = []
    ys = []
    for i in range(mixture.shape[0]):
        m = int(nums[i])
        src = sources_list[i]
        choice = ["mix", "single", "residual"][rng.integers(0, 3)]
        if choice == "mix" or m < 2:
            xs.append(mixture[i])
            ys.append(m - 1)
            continue

        # Oracle recursion to a random depth d in 1..M-1.
        depth = int(rng.integers(1, m))
        residual = mixture[i]
        remaining = list(range(m))
        last_single = None
        for _ in range(depth):
            heads = _forward_one(residual, enc, mn, dec)
            si, matched = _pick_single_head(heads, src, remaining)
            last_single = heads[si]
            residual = heads[1 - si]
            remaining.remove(matched)

        if choice == "single" and last_single is not None:
            xs.append(last_single)
            ys.append(0)                                 # 1 speaker
        else:
            xs.append(residual)
            ys.append(len(remaining) - 1)                # remaining speakers
    length = min(x.shape[-1] for x in xs)
    x = torch.stack([xt[..., :length] for xt in xs], dim=0)
    y = torch.tensor(ys, device=x.device)
    return x, y


def save_checkpoint(out_dir: Path, step: int, model, optimizer) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_step{step}.pt"
    torch.save({"step": step, "num_classes": model.num_classes,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the speaker-count / stop classifier.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/count_clf"))
    parser.add_argument("--speaker-counts", default="1,1,2,2,3,4,5",
                        help="Counts sampled per item; low counts repeated to "
                             "balance the crucial 1-vs-2 boundary.")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If > 0 stop after this many steps. 0 means "
                             "--epochs * --steps-per-epoch.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--orpit-ckpt", type=Path, default=None,
                        help="If set, train on OR-PIT head outputs (residual "
                             "domain) instead of clean mixtures, so the "
                             "classifier matches blind-recursion inputs.")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-wsj02mix",
                        help="Architecture for the OR-PIT checkpoint.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="voxsplit")
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Source directory not found: {args.source_dir}")
        return 1

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    total_steps = (args.max_steps if args.max_steps > 0
                   else args.epochs * args.steps_per_epoch)
    print(f"Planned total steps: {total_steps}")

    model = SpeakerCountCNN(sample_rate=args.sample_rate,
                            num_classes=MAX_SPEAKERS).to(device)
    model.train()

    orpit = None
    if args.orpit_ckpt is not None:
        if not args.orpit_ckpt.is_file():
            print(f"OR-PIT checkpoint not found: {args.orpit_ckpt}")
            return 1
        print(f"Residual-domain training on OR-PIT outputs "
              f"({args.orpit_ckpt}).")
        orpit = load_orpit(args.init_model, args.orpit_ckpt, device)
    rng = np.random.default_rng(args.seed)

    counts = parse_counts(args.speaker_counts)
    ds_length = max(total_steps * args.batch_size * 2, args.batch_size * 4)
    dataset = MixDataset(
        source_dir=args.source_dir, sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds, speaker_counts=counts,
        seed=args.seed, length=ds_length)
    print(f"Dataset scanned {len(dataset.speakers)} speakers; "
          f"counts sampled from {counts}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "train_log.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    if write_header:
        csv_writer.writerow(["step", "loss", "running_loss", "running_acc"])
        csv_fh.flush()

    wandb_log = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, mode=args.wandb_mode,
        config=vars(args), name=f"count_clf_to{total_steps}")

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    step = 0
    stop = False

    while not stop:
        for batch in loader:
            step += 1
            mixture = batch["mixture"].to(device)                 # [B, T]
            if orpit is not None:
                sources = [s.to(device) for s in batch["sources"]]
                inputs, labels = residualize(
                    mixture, sources, batch["num_speakers"], *orpit, rng)
            else:
                inputs = mixture
                labels = torch.tensor(batch["num_speakers"],
                                      device=device) - 1

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                running_correct += int((pred == labels).sum())
                running_total += labels.shape[0]
            running_loss += float(loss)

            if step % args.log_every == 0:
                mean_loss = running_loss / (args.log_every)
                acc = running_correct / max(running_total, 1)
                print(f"step {step}/{total_steps}  loss={float(loss):.4f}  "
                      f"running_loss={mean_loss:.4f}  acc={acc:.3f}")
                csv_writer.writerow([step, f"{float(loss):.6f}",
                                     f"{mean_loss:.6f}", f"{acc:.6f}"])
                csv_fh.flush()
                wandb_log.log({"loss": float(loss), "running_loss": mean_loss,
                               "acc": acc}, step=step)
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                p = save_checkpoint(args.out_dir, step, model, optimizer)
                print(f"Saved checkpoint: {p}")

            if step >= total_steps:
                stop = True
                break

    p = save_checkpoint(args.out_dir, step, model, optimizer)
    print(f"Saved final checkpoint: {p}")
    csv_fh.close()
    wandb_log.finish()
    print(f"Training complete. Log at {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
